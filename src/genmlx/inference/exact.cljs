(ns genmlx.inference.exact
  "Exact tensor enumeration for models with finite discrete support.
   Computes exact posteriors by running the model ONCE with tensor axes
   for each free discrete variable. MLX broadcasting builds the joint
   probability table implicitly — no Cartesian product loop needed.

   This is the GenMLX-native equivalent of memo's array-programming backend:
   each trace site becomes a tensor axis, log-probabilities accumulate via
   broadcasting, and the result is the exact joint distribution.

   Usage:
     (exact-posterior model args observations)
     ;; => {:marginals {:coin {0 0.1, 1 0.9}} :log-ml -0.693 ...}

     (exact-joint model args observations)
     ;; => {:log-probs [2 2] tensor :axes [...] :retval ...}

     (enumerate model)   ;; handler substitution — returns a GF with full GFI

   Constraints:
   - All free trace sites must have finite discrete support (dist-support)
   - Model must not use mx/item or scalar if/when on traced values
   - Same constraints as vsimulate (shape-based batching)"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.dist.core :as dc]
            [genmlx.runtime :as rt]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.dist :as dist]
            [genmlx.dispatch :as dispatch]))

;; ---------------------------------------------------------------------------
;; Axis layout convention
;; ---------------------------------------------------------------------------
;; Each enumerated site gets a :dim value (0 = first traced, 1 = second, ...).
;; In the tensor, newest axis (highest dim) is leftmost (position 0).
;; Tensor position = ndim - 1 - dim.

(defn- dim->pos
  "Convert axis :dim to tensor position. Newest axis is leftmost."
  [ndim dim]
  (- ndim 1 dim))

;; ---------------------------------------------------------------------------
;; Enumerate handler transition
;; ---------------------------------------------------------------------------

(defn enumerate-transition
  "Pure enumerate transition: [state addr dist] -> [value state'].

   Free sites: expands all support values as a new tensor axis (leftmost).
   Constrained sites: uses scalar constraint value, no new axis.

   The values tensor has shape [k, 1, 1, ...] with ndim trailing 1s so it
   broadcasts with all previously enumerated axes. Log-probs are computed
   per support value and padded to the same dimensionality."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (if (cm/has-value? constraint)
      ;; Constrained: scalar value, add log-prob, no new axis
      (let [value (cm/get-value constraint)
            lp (dc/dist-log-prob dist value)]
        [value (-> state
                   (update :choices cm/set-value addr value)
                   (update :score #(mx/add % lp)))])
      ;; Free: enumerate all support values as new leftmost axis
      (let [support (dc/dist-support dist)
            k (count support)]
        (if (= k 1)
          ;; Deterministic (support size 1): use the value directly, no new axis
          (let [value (first support)
                lp (dc/dist-log-prob dist value)]
            [value (-> state
                       (update :choices cm/set-value addr value)
                       (update :score #(mx/add % lp)))])
          ;; Multiple values: create new tensor axis
          (let [ndim (:ndim state)
                ;; Values: [k, 1, 1, ...] with ndim trailing 1s
                val-shape (into [k] (repeat ndim 1))
                values-nd (mx/reshape (mx/stack support) val-shape)
                ;; Log-probs: bulk computation — one op, not K separate calls.
                ;; Returns [K, param_shape...] where param_shape comes from dist params.
                ;; Insert 1s BETWEEN K and param_shape (not trailing) so param dims
                ;; align at the rightmost positions of the score tensor.
                lp-all (dc/dist-log-prob-support dist)
                target-ndim (inc ndim)
                current-ndim (count (mx/shape lp-all))
                lp-nd (if (< current-ndim target-ndim)
                        (let [sh (vec (mx/shape lp-all))
                              k-dim (first sh)
                              param-dims (rest sh)
                              gap (- target-ndim current-ndim)]
                          (mx/reshape lp-all
                                      (into [k-dim] (concat (repeat gap 1) param-dims))))
                        lp-all)]
            [values-nd (-> state
                           (update :choices cm/set-value addr values-nd)
                           (update :score #(mx/add % lp-nd))
                           (update :axes conj {:addr addr :size k :dim ndim
                                               :support support})
                           (update :ndim inc))]))))))

;; ---------------------------------------------------------------------------
;; Exact — inference-mode annotation (lightweight, enumerate-context only)
;; ---------------------------------------------------------------------------

(defn Exact
  "Annotate a gen function for exact enumeration when spliced.
   Lightweight version — only works inside enumerate mode (run-enumerate).
   For full GFI support, use enumerate instead.
   Usage: (splice :agent (Exact agent-model) args)"
  [gf]
  (vary-meta gf assoc ::inference-strategy :exact))

(declare run-enumerate)

(defn- enumerate-and-normalize
  "Run exact enumeration on kernel, return {:probs :log-ml :retval :axes}."
  [kernel args constraints]
  (let [gf (dyn/auto-key kernel)
        constraints (or constraints cm/EMPTY)
        result (run-enumerate gf args constraints)
        score (:score result)
        axes (:axes result)
        log-ml (if (empty? axes)
                 score
                 (mx/logsumexp score (vec (range (count axes)))))
        log-probs (mx/subtract score log-ml)
        probs (mx/exp log-probs)]
    (mx/eval! probs log-ml)
    {:probs probs :log-ml log-ml :retval (:retval result) :axes axes}))

;; ---------------------------------------------------------------------------
;; enumerate — handler substitution for exact enumeration
;; ---------------------------------------------------------------------------

(defn- enumerate-run
  "Run function for the enumerate dispatcher. Handles all GFI ops by running
   the model with enumerate-transition and post-processing the result.
   Replaces the former ExactGF record with dispatch-based execution.
   Called as (enumerate-run op gf args key opts) by custom-transition-dispatcher."
  [op gf args key {:keys [constraints trace selection]}]
  (case op
    :simulate
    (let [{:keys [probs]} (enumerate-and-normalize gf args nil)]
      (tr/make-trace {:gen-fn gf :args (vec args) :choices cm/EMPTY
                      :retval probs :score (mx/scalar 0.0)}))

    :generate
    (let [{:keys [probs log-ml]} (enumerate-and-normalize gf args constraints)]
      {:trace (tr/make-trace {:gen-fn gf :args (vec args) :choices cm/EMPTY
                              :retval probs :score log-ml})
       :weight log-ml})

    :update
    (let [old-score (:score trace)
          {:keys [probs log-ml]} (enumerate-and-normalize gf (:args trace) constraints)]
      {:trace (tr/make-trace {:gen-fn gf :args (:args trace) :choices cm/EMPTY
                              :retval probs :score log-ml})
       :weight (mx/subtract log-ml old-score)
       :discard cm/EMPTY})

    :regenerate
    (let [{:keys [probs]} (enumerate-and-normalize gf (:args trace) nil)]
      {:trace (tr/make-trace {:gen-fn gf :args (:args trace) :choices cm/EMPTY
                              :retval probs :score (mx/scalar 0.0)})
       :weight (mx/scalar 0.0)})

    :assess
    (let [result (run-enumerate (dyn/auto-key gf) args constraints)
          score (:score result)]
      (mx/eval! score)
      {:retval (:retval result) :weight score})

    :propose
    (let [{:keys [probs]} (enumerate-and-normalize gf args nil)]
      {:choices cm/EMPTY
       :weight (mx/scalar 0.0)
       :retval probs})

    :project
    (mx/scalar 0.0)

    (throw (ex-info (str "enumerate: unsupported op " op) {:op op}))))

(defn enumerate
  "Wrap a gen function for exact enumeration via handler substitution.
   Returns a DynamicGF that implements the full GFI by running the model
   with enumerate-transition — all discrete latent variables are analytically
   marginalized. Traces have empty choices; score is the marginal likelihood.

   Usage:
     (p/simulate (enumerate model) args)
     (p/generate (enumerate model) args observations)
     (splice :agent (enumerate agent-model) args)"
  [gf]
  (dispatch/with-dispatch (dyn/auto-key gf)
    (with-meta enumerate-run {:score-type :collapsed})))

(defn categorical-argmax
  "Categorical distribution that deterministically picks the argmax.
   Equivalent to memo's to_maximize. Ties are broken uniformly."
  [values]
  (let [max-val (mx/amax values)
        indicator (.astype (mx/equal values max-val) mx/float32)
        logits (mx/log (mx/divide indicator (mx/sum indicator)))]
    (dist/categorical logits)))

(defn with-cache
  "Wrap a function with atom-based memoization. Cache key = args (JS primitives).
   Results materialized before caching."
  [f]
  (let [cache (atom {})]
    (with-meta
      (fn [& args]
        (let [key (vec args)]
          (or (get @cache key)
              (let [result (apply f args)]
                (when (mx/array? result) (mx/eval! result))
                (swap! cache assoc key result)
                result))))
      {::cache cache})))

(defn clear-cache
  "Clear the cache on a with-cache wrapped function."
  [f]
  (when-let [cache (::cache (meta f))] (reset! cache {})))

(defn- execute-exact
  "Execute a sub-GF in exact enumerate mode. Returns result map compatible
   with merge-sub-result. Only supports simulate/generate contexts — throws
   on update/regenerate (which require trace-level operations incompatible
   with full marginalization)."
  [gf args {:keys [constraints key old-choices selection param-store]}]
  (when (or (and old-choices (not= old-choices cm/EMPTY))
            selection)
    (throw (ex-info "Exact sub-models do not support update/regenerate"
                    {:has-old-choices (and (some? old-choices) (not= old-choices cm/EMPTY))
                     :has-selection (some? selection)})))
  (let [gf (cond-> (dyn/auto-key gf)
             param-store (vary-meta assoc :genmlx.dynamic/param-store param-store))
        constraints (or constraints cm/EMPTY)
        result (run-enumerate gf args constraints)
        score (:score result)
        axes (:axes result)
        ;; log-ml = logsumexp over sub-model's own axes (marginalizes latents)
        sub-positions (vec (range (count axes)))
        log-ml (if (empty? sub-positions)
                 score
                 (mx/logsumexp score sub-positions))
        ;; Normalized probability tensor over sub-model's axes
        log-probs (mx/subtract score log-ml)
        probs (mx/exp log-probs)]
    {:choices cm/EMPTY
     :score log-ml
     :retval probs}))

(defn- execute-sub-default
  "Default executor for non-Exact sub-GFs spliced inside an enumerate model."
  [gf args {:keys [constraints key param-store]}]
  (let [gf (cond-> (dyn/auto-key gf)
             key (vary-meta assoc :genmlx.dynamic/key key)
             param-store (vary-meta assoc :genmlx.dynamic/param-store param-store))]
    (if (and constraints (not= constraints cm/EMPTY))
      (let [{:keys [trace weight]} (p/generate gf args constraints)]
        {:choices (:choices trace) :score (:score trace)
         :retval (:retval trace) :weight weight})
      (let [trace (p/simulate gf args)]
        {:choices (:choices trace) :score (:score trace)
         :retval (:retval trace)}))))

(defn- enumerate-executor
  "Executor for splice calls within enumerate mode. Dispatches on sub-GF
   metadata: ::inference-strategy :exact or ::dispatch/custom-dispatch
   uses exact enumeration, otherwise falls back to standard GFI dispatch."
  [gf args opts]
  (cond
    ;; Exact metadata annotation (lightweight, inside enumerate mode)
    (= :exact (::inference-strategy (meta gf)))
    (execute-exact gf args opts)
    ;; enumerate-wrapped model (via dispatch/with-dispatch)
    (::dispatch/custom-dispatch (meta gf))
    (execute-exact gf args opts)
    ;; Default: standard GFI dispatch
    :else
    (execute-sub-default gf args opts)))

;; ---------------------------------------------------------------------------
;; Run enumerate
;; ---------------------------------------------------------------------------

(defn run-enumerate
  "Run a gen function in enumerate mode. Returns final handler state.

   model: DynamicGF (created by gen macro)
   args:  arguments to the model (vector)
   constraints: choicemap of observations (or nil)

   Returns state map with:
     :score  — joint log-prob tensor, shape [|D_n|, ..., |D_0|]
     :axes   — [{:addr :size :dim} ...] in enumeration order
     :ndim   — number of enumerated axes
     :choices — choicemap with tensor-valued leaves
     :retval — model return value (tensor-shaped)"
  [model args constraints]
  (let [key (rng/fresh-key)
        body-fn (:body-fn model)
        constraints (or constraints cm/EMPTY)
        init {:choices cm/EMPTY
              :score (mx/scalar 0.0)
              :key key
              :constraints constraints
              :axes []
              :ndim 0
              :executor enumerate-executor}
        result (rt/run-handler enumerate-transition init
                               (fn [runtime] (apply body-fn runtime args)))]
    ;; No mx/eval! here — keep the graph lazy.
    ;; Evaluate at API boundaries (exact-posterior, exact-joint, etc.)
    result))

;; ---------------------------------------------------------------------------
;; Post-processing: normalize, marginalize, extract
;; ---------------------------------------------------------------------------

(defn normalize-joint
  "Normalize joint log-score tensor to log-probabilities.
   Subtracts logsumexp over all elements."
  [log-score]
  (let [log-z (mx/logsumexp (mx/reshape log-score [-1]))]
    (mx/subtract log-score log-z)))

(defn marginal
  "Compute marginal log-probs for a target address.
   Marginalizes (logsumexp) over all axes except the target.

   axes: vector of {:addr :size :dim} from run-enumerate
   target-addr: keyword address to keep
   Returns: 1D log-prob tensor of size |D_target|"
  [log-probs axes target-addr]
  (let [target-axis (first (filter #(= (:addr %) target-addr) axes))
        _ (when-not target-axis
            (throw (ex-info (str "Unknown address: " target-addr)
                            {:addr target-addr :axes axes})))
        ndim (count axes)
        target-pos (dim->pos ndim (:dim target-axis))
        other-positions (vec (remove #{target-pos} (range ndim)))]
    (if (empty? other-positions)
      log-probs
      (mx/logsumexp log-probs other-positions))))

(defn- extract-marginal-probs
  "Extract marginal probabilities for a target address.
   Returns map of {value-as-number probability-as-number}."
  [log-probs axes target-addr support]
  (let [m-lp (marginal log-probs axes target-addr)
        m-p (mx/exp m-lp)
        _ (mx/eval! m-p)
        k (count support)]
    (into {} (map (fn [i sv]
                    [(mx/item sv) (mx/item (mx/slice m-p i (inc i)))])
                  (range k) support))))

;; ---------------------------------------------------------------------------
;; Expectation, entropy, variance
;; ---------------------------------------------------------------------------

(defn- all-positions
  "Compute tensor positions for all axes."
  [axes]
  (let [ndim (count axes)]
    (mapv #(dim->pos ndim (:dim %)) axes)))

(defn- addrs->positions
  "Convert address keywords to tensor positions for summation.
   If sum-addrs is nil, returns all positions (sum over everything)."
  [axes sum-addrs]
  (if (nil? sum-addrs)
    (all-positions axes)
    (let [ndim (count axes)
          addr-set (set sum-addrs)]
      (vec (keep (fn [{:keys [addr dim]}]
                   (when (addr-set addr) (dim->pos ndim dim)))
                 axes)))))

(defn expectation
  "Compute E[f(x)] over the joint distribution.

   log-probs: normalized joint log-prob tensor
   axes: axes metadata from run-enumerate
   f-values: tensor of f(x) values, broadcastable with log-probs
   sum-addrs: set of address keywords to sum over, or nil for all"
  [log-probs axes f-values sum-addrs]
  (let [p (mx/exp log-probs)
        weighted (mx/multiply p f-values)
        positions (addrs->positions axes sum-addrs)]
    (if (empty? positions)
      weighted
      (mx/sum weighted positions))))

(defn entropy
  "Compute H[target-addrs] — entropy of marginal over specified addresses.

   log-probs: normalized joint log-prob tensor
   axes: axes metadata from run-enumerate
   target-addrs: set of address keywords to compute joint entropy over"
  [log-probs axes target-addrs]
  (let [target-set (set target-addrs)
        non-target-addrs (into #{} (comp (remove #(target-set (:addr %))) (map :addr)) axes)
        non-target-positions (addrs->positions axes non-target-addrs)
        marginal-lp (if (empty? non-target-positions)
                      log-probs
                      (mx/logsumexp log-probs non-target-positions))
        p (mx/exp marginal-lp)
        ;; H = -sum(p * log(p)), with 0*log(0) = 0
        p-log-p (mx/multiply p marginal-lp)
        p-log-p (mx/where (mx/isnan p-log-p) (mx/scalar 0.0) p-log-p)]
    (mx/negative (mx/sum p-log-p))))

(defn variance
  "Compute Var[f(x)] = E[f(x)^2] - E[f(x)]^2 over the joint distribution.

   log-probs: normalized joint log-prob tensor
   axes: axes metadata from run-enumerate
   f-values: tensor of values, broadcastable with log-probs
   sum-addrs: set of address keywords to sum over, or nil for all"
  [log-probs axes f-values sum-addrs]
  (let [positions (addrs->positions axes sum-addrs)
        p (mx/exp log-probs)
        e-x (mx/sum (mx/multiply p f-values) positions)
        e-x2 (mx/sum (mx/multiply p (mx/multiply f-values f-values)) positions)]
    (mx/subtract e-x2 (mx/multiply e-x e-x))))

;; ---------------------------------------------------------------------------
;; Conditioning and joint marginals (Phase 2)
;; ---------------------------------------------------------------------------

(defn condition-on
  "Condition the joint on addr = value. Removes that axis and renormalizes.

   log-probs: normalized joint log-prob tensor
   axes: axes metadata from run-enumerate
   addr: keyword of the variable to fix
   value: the value to condition on (JS number or MLX scalar)

   Returns {:log-probs <tensor> :axes <updated>}

   Note: does not force mx/eval! — the result stays lazy. For long chains
   of condition-on calls, insert mx/eval! on intermediate results if the
   lazy graph grows too large."
  [log-probs axes addr value]
  (let [target (first (filter #(= (:addr %) addr) axes))
        _ (when-not target
            (throw (ex-info (str "condition-on: unknown address " addr)
                            {:addr addr})))
        ndim (count axes)
        tensor-pos (dim->pos ndim (:dim target))
        ;; Find support index matching value
        value-num (if (mx/array? value) (mx/item value) value)
        idx (first (keep-indexed
                    (fn [i sv] (when (= (mx/item sv) value-num) i))
                    (:support target)))
        _ (when (nil? idx)
            (throw (ex-info (str "condition-on: value " value-num " not in support")
                            {:addr addr :value value-num
                             :support (mapv mx/item (:support target))})))
        ;; Select slice, squeeze out the dim, renormalize
        sliced (mx/take-idx log-probs (mx/array #js [idx] mx/int32) tensor-pos)
        squeezed (mx/squeeze sliced [tensor-pos])
        log-z (mx/logsumexp (mx/reshape squeezed [-1]))
        normalized (mx/subtract squeezed log-z)
        ;; Update axes: remove target, decrement :dim for axes above
        target-dim (:dim target)
        new-axes (into []
                       (comp
                        (remove #(= (:addr %) addr))
                        (map (fn [ax]
                               (if (> (:dim ax) target-dim)
                                 (update ax :dim dec)
                                 ax))))
                       axes)]
    {:log-probs normalized :axes new-axes}))

(defn joint-marginal
  "Keep only keep-addrs, marginalize (logsumexp) over everything else.

   log-probs: normalized joint log-prob tensor
   axes: axes metadata
   keep-addrs: set or collection of address keywords to keep

   Returns {:log-probs <tensor> :axes <filtered>}

   Note: does not force mx/eval! — the result stays lazy. Insert
   mx/eval! on intermediate results if chaining many operations."
  [log-probs axes keep-addrs]
  (let [keep-set (set keep-addrs)
        ndim (count axes)
        marg-positions (vec (keep (fn [{:keys [addr dim]}]
                                    (when-not (keep-set addr)
                                      (dim->pos ndim dim)))
                                  axes))
        ;; Remap :dim values: after marginalization, remaining axes get
        ;; contiguous dims based on their relative order
        kept-sorted (sort-by :dim (filterv #(keep-set (:addr %)) axes))
        kept-axes (into [] (map-indexed (fn [i ax] (assoc ax :dim i))) kept-sorted)
        margd (if (empty? marg-positions)
                log-probs
                (mx/logsumexp log-probs marg-positions))]
    {:log-probs margd :axes kept-axes}))

(defn extract-table
  "Build a probability table by normalizing along condition-addr's axis.

   Returns a tensor where the condition-addr axis moves to position 0
   and entries are conditional probabilities P(other | cond = v).
   Single tensor operation — no per-value loop.

   joint-result: return value from exact-joint
   condition-addr: keyword of the address to normalize over

   Returns MLX tensor of shape [|D_cond|, |D_remaining|...]"
  [joint-result condition-addr]
  (let [axes (:axes joint-result)
        log-probs (:log-probs joint-result)
        cond-axis (first (filter #(= (:addr %) condition-addr) axes))
        _ (when-not cond-axis
            (throw (ex-info (str "extract-table: unknown address " condition-addr)
                            {:addr condition-addr})))
        ndim (count axes)
        cond-pos (dim->pos ndim (:dim cond-axis))
        ;; Normalize along all axes EXCEPT cond-axis (= conditional distribution)
        other-positions (vec (remove #{cond-pos} (range ndim)))
        ;; logsumexp over other axes gives log P(cond=v) for each v (keepdims for broadcasting)
        log-marginal (if (empty? other-positions)
                       log-probs
                       (mx/logsumexp log-probs other-positions true))
        ;; P(other | cond=v) = P(other, cond=v) / P(cond=v)
        table (mx/exp (mx/subtract log-probs log-marginal))
        ;; Move cond-axis to position 0 if not already there
        table (if (zero? cond-pos)
                table
                (let [perm (into [cond-pos] (remove #{cond-pos} (range ndim)))]
                  (mx/transpose table perm)))]
    table))

(defn conditional-marginal
  "Compute P(target-addr | given-addr = given-value).
   Composes condition-on + marginal.

   Returns 1D log-prob tensor for target-addr."
  [log-probs axes target-addr given-addr given-value]
  (let [{cond-lp :log-probs cond-axes :axes}
        (condition-on log-probs axes given-addr given-value)]
    (marginal cond-lp cond-axes target-addr)))

(defn agent-marginal
  "Joint marginal over all addresses in a given namespace.
   E.g., (agent-marginal lp axes \"monty\") keeps all :monty/* axes.

   Returns {:log-probs <tensor> :axes <filtered>}"
  [log-probs axes agent-ns]
  (let [agent-addrs (into #{}
                          (comp (filter #(= (namespace (:addr %)) agent-ns))
                                (map :addr))
                          axes)]
    (joint-marginal log-probs axes agent-addrs)))

(defn observe-constraint
  "Build a choicemap constraining condition-addrs to (mx/scalar 1).
   Used with the bernoulli conditioning trick: (trace addr (dist/bernoulli mask)).
   Merges with existing constraints."
  [constraints condition-addrs]
  (reduce (fn [cm addr]
            (cm/merge-cm cm (cm/choicemap addr (mx/scalar 1))))
          (or constraints cm/EMPTY)
          condition-addrs))

;; ---------------------------------------------------------------------------
;; High-level API
;; ---------------------------------------------------------------------------

(defn exact-posterior
  "Compute exact posterior marginals over all free discrete variables.

   model: gen function (DynamicGF)
   args: model arguments (vector or list)
   observations: choicemap of observed data (or nil)

   Returns:
     :marginals — {addr {value prob}} for each free address
     :log-ml    — log marginal likelihood (JS number)
     :axes      — axes metadata
     :joint-log-probs — normalized joint log-prob tensor"
  [model args observations]
  (let [result (run-enumerate model args observations)
        log-score (:score result)
        axes (:axes result)
        log-ml (mx/logsumexp (mx/reshape log-score [-1]))
        log-probs (mx/subtract log-score log-ml)
        _ (mx/eval! log-probs log-ml)
        marginals
        (into {}
              (map (fn [{:keys [addr support]}]
                     [addr (extract-marginal-probs log-probs axes addr support)])
                   axes))]
    {:marginals marginals
     :log-ml (mx/item log-ml)
     :axes axes
     :joint-log-probs log-probs}))

(defn exact-joint
  "Compute exact joint distribution over all free discrete variables.
   Returns the full joint probability tensor and axes metadata.

   model: gen function (DynamicGF)
   args: model arguments
   observations: choicemap of observed data (or nil)

   Returns:
     :log-probs — normalized joint log-prob tensor
     :probs     — joint probability tensor (exp of log-probs)
     :log-ml    — log marginal likelihood
     :axes      — axes metadata
     :retval    — model return value (tensor-shaped)"
  [model args observations]
  (let [result (run-enumerate model args observations)
        log-score (:score result)
        log-ml (mx/logsumexp (mx/reshape log-score [-1]))
        log-probs (mx/subtract log-score log-ml)
        probs (mx/exp log-probs)
        _ (mx/eval! log-probs probs log-ml)]
    {:log-probs log-probs
     :probs probs
     :log-ml (mx/item log-ml)
     :axes (:axes result)
     :retval (:retval result)}))

(defn exact-marginal-likelihood
  "Compute exact marginal likelihood log p(observations).

   model: gen function (DynamicGF)
   args: model arguments
   observations: choicemap of observed data (or nil)

   Returns: JS number (log marginal likelihood)"
  [model args observations]
  (let [result (run-enumerate model args observations)
        log-ml (mx/logsumexp (mx/reshape (:score result) [-1]))]
    (mx/eval! log-ml)
    (mx/item log-ml)))

;; ---------------------------------------------------------------------------
;; High-level helpers for readable model composition
;; ---------------------------------------------------------------------------

(defn thinks
  "Wrap a gen function for exact enumeration — one agent modeling another.
   Use with splice inside a gen body:

     (gen []
       (let [probs (splice :audience (exact/thinks (belief-model w)))]
         ...))

   Reads as: this agent THINKS about the belief-model, getting back
   the exact probability table over all the model's discrete choices."
  [model]
  (enumerate model))

(defn observes
  "Observe a discrete choice and get the posterior.
   Shorthand for exact-joint + condition-on.

     (exact/observes agent-model :action 5)
     ;; → probability vector over remaining choices given action=5

   With target address:

     (exact/observes agent-model :action 5 :wall)
     ;; → P(wall | action=5)"
  ([model observed-addr observed-value]
   (let [joint (exact-joint model [] nil)
         c (condition-on (:log-probs joint) (:axes joint)
                         observed-addr observed-value)
         p (mx/exp (:log-probs c))
         _ (mx/eval! p)]
     p))
  ([model observed-addr observed-value target-addr]
   (let [joint (exact-joint model [] nil)
         c (condition-on (:log-probs joint) (:axes joint)
                         observed-addr observed-value)
         m (joint-marginal (:log-probs c) (:axes c) #{target-addr})
         p (mx/exp (:log-probs m))
         _ (mx/eval! p)]
     p)))

(defn pr
  "Probability of a specific value, optionally conditioned.
   The closest GenMLX equivalent to memo's Pr[agent.x == v].

   Unconditional:
     (exact/pr model :wall 1)
     ;; → P(wall=1) as a JS number

   Conditional:
     (exact/pr model :wall 1 :given :action 5)
     ;; → P(wall=1 | action=5) as a JS number"
  ([model addr value]
   (let [joint (exact-joint model [] nil)
         m (joint-marginal (:log-probs joint) (:axes joint) #{addr})
         p (mx/exp (:log-probs m))
         _ (mx/eval! p)]
     (mx/item (mx/idx p value))))
  ([model addr value given-kw given-addr given-value]
   (mx/item (mx/idx (observes model given-addr given-value addr) value))))

(defn mutual-info
  "Mutual information I(A;B) between two sets of traced variables.
   Computed as H(A) + H(B) - H(A,B) from the exact joint.

     (exact/mutual-info model #{:x} #{:y})
     ;; → I(X;Y) in nats as a JS number

   Optionally divide by (Math/log 2) for bits."
  [model addr-set-a addr-set-b]
  (let [joint (exact-joint model [] nil)
        lp (:log-probs joint)
        ax (:axes joint)
        h-a (entropy lp ax addr-set-a)
        h-b (entropy lp ax addr-set-b)
        h-ab (entropy lp ax (into addr-set-a addr-set-b))
        mi (mx/subtract (mx/add h-a h-b) h-ab)
        _ (mx/eval! mi)]
    (mx/item mi)))
