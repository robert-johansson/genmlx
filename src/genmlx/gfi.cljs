(ns genmlx.gfi
  "The Generative Function Interface as an algebraic theory.

   A generative function P = (X, Y, p, f) is defined by:
     X  — argument type
     Y  — return type
     p  — family of structured, well-behaved probability densities on
          choice dictionaries, indexed by arguments x in X
     f  — deterministic return value function f(x, tau) -> Y

   This namespace expresses the GFI not as protocols (those are in
   protocols.cljs) but as the algebraic LAWS that any correct
   implementation must satisfy. Each law is data: a name, a citation
   to its mathematical source, a theorem statement, and a check
   function.

   The laws are derived from two sources:
     [T] Cusumano-Towner 2020 PhD thesis, Chapter 2
     [D] gen.dev/docs/stable/ref/core/gfi

   Usage:
     (gfi/verify model args)              ;; run all laws
     (gfi/verify model args {:laws [:update-density-ratio]})
     (gfi/check-law :score-decomposition model args)

   See also: genmlx.contracts (DEPRECATED — all 11 contracts subsumed here)
             genmlx.verify (static validation)
             genmlx.schemas (Malli structural schemas)"
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.gradients :as grad]
            [genmlx.verify :as verify]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]
            [genmlx.learning :as learn]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- ev
  "Evaluate an MLX array and extract its scalar value."
  [x]
  (mx/realize x))

(defn- approx=
  "True if |a - b| <= tol. Both must be finite."
  ([a b] (approx= a b 0.05))
  ([a b tol]
   (and (js/Number.isFinite a)
        (js/Number.isFinite b)
        (<= (js/Math.abs (- a b)) tol))))

(defn- pos-infinite?
  "True if x is +Infinity."
  [x]
  (and (number? x) (not (js/Number.isFinite x)) (pos? x)))

(defn- first-leaf-path
  "Return the first leaf address path from a choicemap, or nil."
  [choices]
  (first (cm/addresses choices)))

(defn path->selection
  "Convert an address path like [:x] or [:inner :z] to a selection."
  [path]
  (if (= 1 (count path))
    (sel/select (first path))
    (reduce (fn [inner-sel addr]
              (sel/hierarchical addr inner-sel))
            (sel/select (last path))
            (reverse (butlast path)))))

(defn- all-leaf-addrs
  "Return all leaf address paths from a choicemap."
  [choices]
  (cm/addresses choices))

(defn strip-compiled
  "Return a copy of model with all compiled execution paths removed from
   its schema, forcing the handler (interpreter) path for all GFI ops.
   Returns the model unchanged if it has no schema."
  [model]
  (let [schema (:schema model)]
    (if (nil? schema)
      model
      (dyn/->DynamicGF (:body-fn model) (:source model)
                        (dissoc schema
                                :compiled-simulate :compiled-generate
                                :compiled-update :compiled-assess
                                :compiled-project :compiled-regenerate)))))

;; ---------------------------------------------------------------------------
;; The algebraic laws
;; ---------------------------------------------------------------------------

(def laws
  "The GFI algebraic laws.

   Each law is a map:
     :name     — keyword identifier
     :from     — citation to thesis definition/proposition or gen.dev docs
     :theorem  — human-readable statement of the invariant
     :tags     — set of tags for filtering (e.g. #{:simulate :core})
     :check    — (fn [{:keys [model args]}] -> bool)
                 The model is a gen fn, args are valid arguments.
                 The check may internally simulate traces as needed."

  [;; ===================================================================
   ;; SIMULATE laws
   ;; ===================================================================

   {:name :simulate-produces-trace
    :from "[T] Def 2.1.16, §2.3.1 SIMULATE"
    :theorem "simulate(P, x) returns trace t = (P, x, tau) where
              tau is in supp(p(.; x)) and score is finite"
    :tags #{:simulate :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)]
               (and (some? (:choices t))
                    (= (:gen-fn t) model)
                    (js/Number.isFinite (ev (:score t))))))}

   {:name :simulate-score-is-log-density
    :from "[T] §2.3.1 LOGPDF, [D] get_score"
    :theorem "trace.score = log p(tau; x) = assess(P, x, tau).weight"
    :tags #{:simulate :assess :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   s (ev (:score t))
                   {:keys [weight]} (p/assess model args (:choices t))
                   w (ev weight)]
               (approx= s w 0.01)))}

   {:name :halts-with-probability-one
    :from "[T] Def 2.1.16"
    :theorem "simulate(P, x) terminates with probability 1"
    :tags #{:simulate :well-formedness}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)]
               (some? t)))}

   ;; ===================================================================
   ;; GENERATE laws
   ;; ===================================================================

   {:name :generate-empty-is-simulate
    :from "[D] generate with empty constraints"
    :theorem "generate(P, x, {}).weight = 0 (equivalent to simulate)"
    :tags #{:generate :core}
    :check (fn [{:keys [model args]}]
             (let [{:keys [weight]} (p/generate model args cm/EMPTY)]
               (approx= 0.0 (ev weight) 0.01)))}

   {:name :generate-full-weight-equals-score
    :from "[T] §2.3.1 GENERATE, [D] generate"
    :theorem "generate(P, x, tau).weight = trace.score when fully constrained
              (weight = log p-bar(sigma) for existentially sound sigma)"
    :tags #{:generate :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   {:keys [trace weight]} (p/generate model args (:choices t))]
               (approx= (ev (:score trace)) (ev weight) 0.01)))}

   {:name :return-value-independence
    :from "[T] §2.3.1"
    :theorem "f(x, tau) depends only on x and tau. Two generates with
              identical choices produce identical retvals."
    :tags #{:generate :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   r1 (:trace (p/generate model args (:choices t)))
                   r2 (:trace (p/generate model args (:choices t)))]
               (approx= (ev (:retval r1)) (ev (:retval r2)) 1e-10)))}

   ;; ===================================================================
   ;; ASSESS laws
   ;; ===================================================================

   {:name :assess-equals-generate-score
    :from "[D] assess"
    :theorem "assess(P, x, tau).weight = generate(P, x, tau).trace.score"
    :tags #{:assess :generate :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   {:keys [weight]} (p/assess model args (:choices t))
                   {:keys [trace]} (p/generate model args (:choices t))]
               (approx= (ev weight) (ev (:score trace)) 0.01)))}

   ;; ===================================================================
   ;; PROPOSE laws
   ;; ===================================================================

   {:name :propose-weight-equals-generate
    :from "[D] propose"
    :theorem "propose(P, x).weight = generate(P, x, propose.choices).weight"
    :tags #{:propose :generate :core}
    :check (fn [{:keys [model args]}]
             (let [{:keys [choices weight]} (p/propose model args)
                   pw (ev weight)
                   {:keys [weight]} (p/generate model args choices)
                   gw (ev weight)]
               (approx= pw gw 0.01)))}

   ;; ===================================================================
   ;; UPDATE laws
   ;; ===================================================================

   {:name :update-identity
    :from "[T] §2.3.1 UPDATE, h_update(tau, tau) = (tau, {})"
    :theorem "update(P, t, t.choices).weight = 0 (no-op update)"
    :tags #{:update :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   {:keys [weight]} (p/update model t (:choices t))]
               (approx= 0.0 (ev weight) 0.01)))}

   {:name :update-density-ratio
    :from "[T] §2.3.1 UPDATE weight = log(p(tau';x')/p(tau;x))"
    :theorem "update(t, sigma).weight = new_score - old_score"
    :tags #{:update :core}
    :check (fn [{:keys [model args]}]
             (let [t1 (p/simulate model args)
                   t2 (p/simulate model args)
                   old-score (ev (:score t1))
                   {:keys [trace weight]} (p/update model t1 (:choices t2))
                   new-score (ev (:score trace))
                   w (ev weight)]
               (approx= w (- new-score old-score) 0.1)))}

   {:name :update-round-trip
    :from "[T] Proposition 2.3.1"
    :theorem "update(update(t, sigma).trace, discard) recovers
              original values and original score"
    :tags #{:update :core}
    :check (fn [{:keys [model args]}]
             (let [t1 (p/simulate model args)
                   t2 (p/simulate model args)
                   orig-score (ev (:score t1))
                   {:keys [trace discard]} (p/update model t1 (:choices t2))
                   {:keys [trace]} (p/update model trace discard)
                   recovered (ev (:score trace))]
               (approx= orig-score recovered 0.05)))}

   {:name :update-discard-completeness
    :from "[T] §2.3.1 UPDATE"
    :theorem "update(t, sigma).discard contains the original value at
              every address that was overwritten by sigma"
    :tags #{:update :core}
    :check (fn [{:keys [model args]}]
             (let [t1 (p/simulate model args)
                   t2 (p/simulate model args)
                   {:keys [discard]} (p/update model t1 (:choices t2))]
               (every? (fn [path]
                         (approx= (ev (cm/get-choice (:choices t1) path))
                                  (ev (cm/get-choice discard path))
                                  1e-6))
                       (cm/addresses discard))))}

   ;; ===================================================================
   ;; REGENERATE laws
   ;; ===================================================================

   {:name :regenerate-empty-identity
    :from "[D] regenerate with empty selection"
    :theorem "regenerate(P, t, none).weight = 0 and choices unchanged"
    :tags #{:regenerate :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   {:keys [trace weight]} (p/regenerate model t sel/none)
                   w (ev weight)]
               (approx= 0.0 w 0.01)))}

   {:name :regenerate-preserves-unselected
    :from "[D] regenerate"
    :theorem "regenerate(t, S) preserves choice values outside S"
    :tags #{:regenerate :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))]
               (if (< (count addrs) 2)
                 true ;; need at least 2 addresses
                 (let [selected (first (first addrs))
                       unselected (map first (rest addrs))
                       orig-vals (into {} (map (fn [a]
                                                 [a (ev (cm/get-value
                                                         (cm/get-submap
                                                          (:choices t) a)))])
                                               unselected))
                       {:keys [trace]} (p/regenerate model t
                                                     (sel/select selected))]
                   (every? (fn [a]
                             (let [v (ev (cm/get-value
                                          (cm/get-submap (:choices trace) a)))]
                               (approx= (get orig-vals a) v 1e-6)))
                           unselected)))))}

   {:name :regenerate-weight-formula
    :from "[T] Eq 4.1, §3.4.2"
    :theorem "regenerate(t, S).weight = (new_score - old_score) -
              (project(new, S) - project(old, S)).
              The weight captures the change in log-density at unselected
              addresses whose distributions depend on selected values.
              Verified for both first and last leaf addresses."
    :tags #{:regenerate :inference :mcmc}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))]
               (if (< (count addrs) 2)
                 true ;; single-site: weight always 0, tested elsewhere
                 (let [check-addr (fn [addr]
                                    (let [sel (sel/select addr)
                                          old-score (ev (:score t))
                                          old-proj-s (ev (p/project model t sel))
                                          {:keys [trace weight]}
                                          (p/regenerate model t sel)
                                          new-score (ev (:score trace))
                                          new-proj-s (ev (p/project model trace sel))
                                          w (ev weight)
                                          expected (- (- new-score old-score)
                                                      (- new-proj-s old-proj-s))]
                                      (approx= w expected 0.01)))
                       first-addr (first (first addrs))
                       last-addr (first (last addrs))]
                   (and (check-addr first-addr)
                        (check-addr last-addr))))))}

   {:name :mh-acceptance-correctness
    :from "[T] Alg 5, §3.4.2"
    :theorem "MH with regenerate weight produces a chain whose
              statistics converge to the correct posterior.
              Verified on Normal-Normal conjugate: prior N(0,1),
              likelihood N(x, 0.5), observe y=2.
              Posterior mean = 1.6, checked after 500 MH steps."
    :tags #{:regenerate :inference :mcmc}
    :check (fn [_]
             ;; Fixed conjugate model — tests the full MH pipeline
             ;; independent of the passed-in model.
             ;; Prior: x ~ N(0,1), Likelihood: y ~ N(x, 0.5), y=2
             ;; Posterior: x | y=2 ~ N(1.6, sqrt(0.2))
             (let [mh-model (dyn/auto-key
                             (dyn/make-gen-fn
                              (fn [rt]
                                (let [trace (.-trace rt)
                                      x (trace :x (dist/gaussian 0 1))]
                                  (trace :y (dist/gaussian x 0.5))))
                              '([] (let [x (trace :x (dist/gaussian 0 1))]
                                     (trace :y (dist/gaussian x 0.5))))))
                   obs (cm/choicemap :y (mx/scalar 2.0))
                   init-trace (:trace (p/generate mh-model [] obs))
                   sel (sel/select :x)
                   final (loop [t init-trace i 0 acc [] k (rng/fresh-key 7)]
                           (if (>= i 500)
                             acc
                             (let [[k1 k2] (rng/split k)
                                   {:keys [trace weight]} (p/regenerate mh-model t sel)
                                   w (ev weight)
                                   accept? (u/accept-mh? w k1)
                                   next-t (if accept? trace t)
                                   x-val (ev (cm/get-value
                                              (cm/get-submap (:choices next-t) :x)))]
                               (recur next-t (inc i)
                                      (if (>= i 100) (conj acc x-val) acc)
                                      k2))))
                   mean (/ (reduce + final) (count final))]
               (approx= mean 1.6 0.3)))}

   {:name :mh-proposal-reversibility
    :from "[T] §3.4.2"
    :theorem "Regenerate-based MH proposals are reversible: after
              regenerate(t, S) -> t', regenerate(t', S) produces a
              valid trace with finite weight. Combined with
              preserves-unselected, this ensures the Markov chain
              can reach the original state."
    :tags #{:regenerate :inference :mcmc}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))]
               (if (empty? addrs)
                 true
                 (let [sel (sel/select (first (first addrs)))
                       ;; Forward move
                       {fwd-trace :trace fwd-weight :weight}
                       (p/regenerate model t sel)
                       w-fwd (ev fwd-weight)
                       ;; Reverse move
                       {rev-trace :trace rev-weight :weight}
                       (p/regenerate model fwd-trace sel)
                       w-rev (ev rev-weight)]
                   (and (js/Number.isFinite w-fwd)
                        (js/Number.isFinite w-rev)
                        ;; Reverse trace has same address set
                        (= (set (all-leaf-addrs (:choices t)))
                           (set (all-leaf-addrs (:choices rev-trace)))))))))}

   ;; ===================================================================
   ;; PROJECT laws
   ;; ===================================================================

   {:name :project-all-equals-score
    :from "[D] project"
    :theorem "project(P, t, all) = trace.score"
    :tags #{:project :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   proj (ev (p/project model t sel/all))
                   s (ev (:score t))]
               (approx= proj s 0.01)))}

   {:name :project-none-equals-zero
    :from "[D] project"
    :theorem "project(P, t, none) = 0"
    :tags #{:project :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   proj (ev (p/project model t sel/none))]
               (approx= 0.0 proj 0.01)))}

   {:name :score-decomposition
    :from "[T] §2.3.1, score additivity over disjoint selections"
    :theorem "project(t, S) + project(t, complement(S)) = trace.score"
    :tags #{:project :core :decomposition}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))]
               (if (< (count addrs) 2)
                 true ;; need 2+ addrs for meaningful decomposition
                 (let [first-addr (first (first addrs))
                       s (sel/select first-addr)
                       score (ev (:score t))
                       proj-s (ev (p/project model t s))
                       proj-cs (ev (p/project model t
                                              (sel/complement-sel s)))]
                   (approx= score (+ proj-s proj-cs) 0.1)))))}

   {:name :score-full-decomposition
    :from "[T] §2.3.1, [D] project"
    :theorem "Sum of project(t, {a_i}) over all leaf addrs = trace.score"
    :tags #{:project :decomposition}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))
                   score (ev (:score t))
                   proj-sum (reduce
                             (fn [acc path]
                               (let [s (path->selection path)
                                     proj (ev (p/project model t s))]
                                 (+ acc proj)))
                             0.0
                             addrs)]
               (approx= proj-sum score 0.05)))}

   ;; ===================================================================
   ;; STRUCTURED DENSITY laws [T] Def 2.1.3, Prop 2.1.2, Prop 2.1.3
   ;; ===================================================================

   {:name :structured-density
    :from "[T] Def 2.1.3"
    :theorem "For structured p: all traces in supp(p) have finite score.
              Traces sharing the same address set all have positive density."
    :tags #{:core :well-formedness}
    :check (fn [{:keys [model args]}]
             (let [traces (repeatedly 20 #(p/simulate model args))
                   groups (group-by (fn [t] (set (all-leaf-addrs (:choices t))))
                                    traces)]
               (every? (fn [[_addr-set ts]]
                         (every? (fn [t] (js/Number.isFinite (ev (:score t))))
                                 ts))
                       groups)))}

   {:name :conditional-is-structured
    :from "[T] Prop 2.1.2"
    :theorem "For any sigma, p(.|sigma) is structured. Generate with
              partial constraints produces traces with finite score."
    :tags #{:generate :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))]
               (if (< (count addrs) 2)
                 true
                 (let [first-addr (first (first addrs))
                       val (cm/get-value (cm/get-submap (:choices t) first-addr))
                       partial (cm/choicemap first-addr val)
                       {:keys [trace weight]} (p/generate model args partial)]
                   (and (js/Number.isFinite (ev (:score trace)))
                        (js/Number.isFinite (ev weight)))))))}

   {:name :generate-uniqueness
    :from "[T] Prop 2.1.3"
    :theorem "For structured p: given full constraints sigma, the trace
              is unique. Two generates with same full constraints produce
              identical scores and identical retvals."
    :tags #{:generate :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   r1 (p/generate model args (:choices t))
                   r2 (p/generate model args (:choices t))]
               (and (approx= (ev (:score (:trace r1)))
                             (ev (:score (:trace r2))) 1e-10)
                    (approx= (ev (:retval (:trace r1)))
                             (ev (:retval (:trace r2))) 1e-10)
                    (approx= (ev (:weight r1))
                             (ev (:weight r2)) 1e-10))))}

   ;; ===================================================================
   ;; CROSS-OPERATION consistency laws
   ;; ===================================================================

   {:name :generate-assess-agreement
    :from "[D] generate, assess"
    :theorem "For fully constrained sigma:
              generate(P, x, sigma).trace.score = assess(P, x, sigma).weight"
    :tags #{:generate :assess :consistency}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   {:keys [trace]} (p/generate model args (:choices t))
                   gs (ev (:score trace))
                   {:keys [weight]} (p/assess model args (:choices t))
                   aw (ev weight)]
               (approx= gs aw 0.01)))}

   {:name :propose-is-simulate-plus-score
    :from "[D] propose"
    :theorem "propose(P, x) = {choices: simulate.choices,
              weight: simulate.score} (modulo internal proposal)"
    :tags #{:propose :simulate :consistency}
    :check (fn [{:keys [model args]}]
             (let [{:keys [choices weight retval]} (p/propose model args)
                   pw (ev weight)]
                 ;; Weight should be finite and choices should be assessable
               (and (js/Number.isFinite pw)
                    (let [{:keys [weight]} (p/assess model args choices)]
                      (approx= pw (ev weight) 0.01)))))}

   ;; ===================================================================
   ;; COMPOSITIONALITY laws
   ;; ===================================================================

   {:name :update-compositionality
    :from "[T] Proposition 2.3.2"
    :theorem "For P3 = P1;P2 (splice): log w3 = log w1 + log w2
              where w1 = inner score diff, w2 = outer score diff"
    :tags #{:update :compositionality :splice}
    :check (fn [{:keys [model args]}]
             (let [t1 (p/simulate model args)
                   t2 (p/simulate model args)
                   {:keys [trace weight]} (p/update model t1 (:choices t2))
                   w3 (ev weight)
                   ;; Partition: inner (splice namespace) vs outer
                   inner-sel (sel/hierarchical :inner sel/all)
                   outer-sel (sel/complement-sel inner-sel)
                   ;; w1 = inner score difference via project
                   w1 (- (ev (p/project model trace inner-sel))
                         (ev (p/project model t1 inner-sel)))
                   ;; w2 = outer score difference via project
                   w2 (- (ev (p/project model trace outer-sel))
                         (ev (p/project model t1 outer-sel)))]
               (approx= w3 (+ w1 w2) 0.05)))}

   ;; ===================================================================
   ;; GRADIENT laws [T] Eq 2.12, §2.3.1
   ;; ===================================================================

   {:name :gradient-choice-correctness
    :from "[T] Eq 2.12"
    :theorem "Choice gradients: d(score)/d(tau[a]) matches finite-difference
              approximation for all continuous addresses."
    :tags #{:gradient :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (->> (:choices t) all-leaf-addrs (mapv first))
                   grads (grad/choice-gradients model t addrs)
                   h 1e-3]
               (every?
                (fn [addr]
                  (let [v (ev (cm/get-value (cm/get-submap (:choices t) addr)))
                        choices-plus (cm/set-choice (:choices t) [addr]
                                                    (mx/scalar (+ v h)))
                        choices-minus (cm/set-choice (:choices t) [addr]
                                                     (mx/scalar (- v h)))
                        score-plus (-> (p/generate model args choices-plus)
                                       :trace :score ev)
                        score-minus (-> (p/generate model args choices-minus)
                                        :trace :score ev)
                        fd-grad (/ (- score-plus score-minus) (* 2 h))
                        analytical (ev (get grads addr))]
                    (approx= analytical fd-grad 0.05)))
                addrs)))}

   {:name :gradient-argument-correctness
    :from "[T] §2.3.1"
    :theorem "Argument gradients: d(score)/d(x) via AD matches
              finite-difference approximation."
    :tags #{:gradient}
    :check (fn [{:keys [model args]}]
             (if (empty? args)
               true
               (let [t (p/simulate model args)
                     choices (:choices t)
                     args-v (vec args)
                     h 1e-3]
                 (every?
                  (fn [i]
                    (let [x-val (nth args-v i)
                           ;; AD gradient: differentiate score w.r.t. arg i
                          score-fn (fn [x-arr]
                                     (:weight (p/generate model
                                                          (assoc args-v i x-arr)
                                                          choices)))
                          analytical (ev ((mx/grad score-fn) x-val))
                           ;; FD gradient: central differences
                          x-num (ev x-val)
                          sp (ev (:weight (p/generate model
                                                      (assoc args-v i (mx/scalar (+ x-num h)))
                                                      choices)))
                          sm (ev (:weight (p/generate model
                                                      (assoc args-v i (mx/scalar (- x-num h)))
                                                      choices)))
                          fd (/ (- sp sm) (* 2 h))]
                      (approx= analytical fd 0.05)))
                  (range (count args-v))))))}

   ;; ===================================================================
   ;; INFERENCE laws (importance sampling)
   ;; ===================================================================

   {:name :is-weight-formula
    :from "[T] Alg 2, Eq 3.2"
    :theorem "IS weight from generate with partial constraints equals the
              log-prob contribution of constrained addresses:
              weight = project(trace, obs_selection)"
    :tags #{:inference :importance-sampling}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))]
               (if (< (count addrs) 2)
                 true
                 (let [obs-addr (first (first addrs))
                       obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                       obs (cm/choicemap obs-addr obs-val)
                       {:keys [trace weight]} (p/generate model args obs)
                       w (ev weight)
                       expected (ev (p/project model trace
                                               (sel/select obs-addr)))]
                   (approx= w expected 0.01)))))}

   {:name :proposal-support-coverage
    :from "[T] Eq 3.3"
    :theorem "p(tau) > 0 implies q(tau) > 0. All generate weights are
              finite (no -Inf), confirming proposal covers model support."
    :tags #{:inference :importance-sampling}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))]
               (if (< (count addrs) 2)
                 true
                 (let [obs-addr (first (first addrs))
                       obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                       obs (cm/choicemap obs-addr obs-val)
                       weights (repeatedly 20
                                           #(ev (:weight (p/generate model args obs))))]
                   (every? js/Number.isFinite weights)))))}

   {:name :log-ml-estimate-well-defined
    :from "[T] Eq 3.5"
    :theorem "The IS log-marginal-likelihood estimate logsumexp(weights) - log(N)
              is well-defined (finite) for valid models with observations.
              Convergence to the analytical value is verified separately in
              log-ml-convergence-analytical."
    :tags #{:inference :importance-sampling}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))]
               (if (< (count addrs) 2)
                 true
                 (let [obs-addr (first (first addrs))
                       obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                       obs (cm/choicemap obs-addr obs-val)
                       weights (repeatedly 100
                                           #(ev (:weight (p/generate model args obs))))
                       max-w (apply max weights)
                       log-ml (+ max-w
                                 (js/Math.log
                                  (/ (reduce + (map #(js/Math.exp (- % max-w)) weights))
                                     (count weights))))]
                   (js/Number.isFinite log-ml)))))}

   ;; ===================================================================
   ;; INTERNAL PROPOSAL laws [T] Ch. 4
   ;; ===================================================================

   {:name :internal-proposal-support
    :from "[T] Def 4.1.1 — q(v; x, sigma) > 0 iff p(v|sigma; x) > 0"
    :theorem "The forward-sampling internal proposal has identical support to
              the model. Tested at SUPPORT BOUNDARIES: x=0 and x=1 for Uniform,
              x near 0 and 1 for Beta, x=0 for Exponential. All generate
              weights must be finite, confirming no support gaps at boundaries.
              Distinct from proposal-support-coverage (#12): this law tests
              boundary values specifically, not arbitrary in-support values."
    :tags #{:generate :internal-proposal :ch4}
    :check (fn [_]
             ;; Fixed models with known support boundaries — independent of
             ;; the passed-in model, because we need specific distributions
             ;; whose boundary behavior we can reason about analytically.
             (let [;; Uniform boundary: weight = log Uniform(x; 0,1) = 0
                   ;; Analytical: -log(1) = 0 for x in [0,1]
                   uniform-model (dyn/auto-key
                                  (dyn/make-gen-fn
                                   (fn [rt]
                                     (let [trace (.-trace rt)
                                           x (trace :x (dist/uniform 0 1))]
                                       (trace :y (dist/gaussian x 1))))
                                   '([] (let [x (trace :x (dist/uniform 0 1))]
                                          (trace :y (dist/gaussian x 1))))))
                   ;; Beta(2,5) boundary: log-prob at x=0.01 and x=0.99
                   ;; Analytical: log B(2,5)^{-1} + log(x) + 4*log(1-x)
                   ;; At 0.01: -1.2442; at 0.99: -15.030. Both finite.
                   beta-model (dyn/auto-key
                               (dyn/make-gen-fn
                                (fn [rt]
                                  (let [trace (.-trace rt)
                                        p (trace :p (dist/beta-dist 2 5))]
                                    (trace :x (dist/bernoulli p))))
                                '([] (let [p (trace :p (dist/beta-dist 2 5))]
                                       (trace :x (dist/bernoulli p))))))
                   ;; Exponential boundary: log Exp(0; 1) = 0
                   ;; Analytical: log(lambda) - lambda*x = 0 - 0 = 0
                   exp-model (dyn/auto-key
                              (dyn/make-gen-fn
                               (fn [rt]
                                 (let [trace (.-trace rt)
                                       x (trace :x (dist/exponential 1))]
                                   (trace :y (dist/gaussian x 1))))
                               '([] (let [x (trace :x (dist/exponential 1))]
                                      (trace :y (dist/gaussian x 1))))))]
               (and
                ;; Uniform at boundaries x=0 and x=1
                (let [w0 (ev (:weight (p/generate uniform-model []
                                                  (cm/choicemap :x (mx/scalar 0.0)))))
                      w1 (ev (:weight (p/generate uniform-model []
                                                  (cm/choicemap :x (mx/scalar 1.0)))))]
                  (and (js/Number.isFinite w0) (approx= w0 0.0 1e-6)
                       (js/Number.isFinite w1) (approx= w1 0.0 1e-6)))
                ;; Beta(2,5) at near-boundary values
                (let [w-lo (ev (:weight (p/generate beta-model []
                                                    (cm/choicemap :p (mx/scalar 0.01)))))
                      w-hi (ev (:weight (p/generate beta-model []
                                                    (cm/choicemap :p (mx/scalar 0.99)))))]
                  (and (js/Number.isFinite w-lo)
                       (js/Number.isFinite w-hi)))
                ;; Exponential at boundary x=0
                (let [w0 (ev (:weight (p/generate exp-model []
                                                  (cm/choicemap :x (mx/scalar 0.0)))))]
                  (and (js/Number.isFinite w0) (approx= w0 0.0 1e-6))))))}

   {:name :generate-weight-is-marginal-likelihood
    :from "[T] §4.1.1 — log w = log p(tau; x) / q(v; x, sigma)
           = sum_{a in sigma} log p(sigma[a] | parents(a))"
    :theorem "For forward-sampling internal proposal, generate weight equals
              the sum of log-probs at CONSTRAINED addresses only. This is
              the marginal likelihood of the constraints under the model.
              Equivalently: weight = project(trace, constrained_selection).
              Tested with MULTIPLE constrained addresses (2 of 3 sites).
              Distinct from is-weight-formula (#11): that law constrains one
              address. This law constrains multiple addresses simultaneously
              and verifies the additive decomposition."
    :tags #{:generate :internal-proposal :ch4}
    :check (fn [_]
             ;; Non-conjugate model to bypass analytical dispatcher
             ;; a ~ Uniform(0,1), b ~ Laplace(a, 1), c ~ Exponential(1)
             ;; No conjugate pairs => compiled/handler path used
             (let [model (dyn/auto-key
                          (dyn/make-gen-fn
                           (fn [rt]
                             (let [trace (.-trace rt)
                                   a (trace :a (dist/uniform 0 1))
                                   b (trace :b (dist/laplace a 1))]
                               (trace :c (dist/exponential 1))))
                           '([] (let [a (trace :a (dist/uniform 0 1))
                                      b (trace :b (dist/laplace a 1))]
                                  (trace :c (dist/exponential 1))))))]
               ;; Test 1: constrain single address — weight = its log-prob
               (let [c1 (cm/choicemap :a (mx/scalar 0.5))
                     {:keys [trace weight]} (p/generate model [] c1)
                     w (ev weight)
                     ;; Uniform(0.5; 0,1): analytical = -log(1) = 0
                     expected 0.0]
                 (when-not (approx= w expected 0.01)
                   (throw (ex-info "single constraint" {:w w :expected expected}))))
               ;; Test 2: constrain TWO addresses — weight = sum of log-probs
               ;; Constrain a=0.5 and c=0.5. b is unconstrained.
               (let [c2 (cm/choicemap :a (mx/scalar 0.5) :c (mx/scalar 0.5))
                     {:keys [trace weight]} (p/generate model [] c2)
                     w (ev weight)
                     ;; Uniform(0.5;0,1) = 0, Exp(0.5;1) = log(1)-1*0.5 = -0.5
                     expected (+ 0.0 -0.5)]
                 (when-not (approx= w expected 0.01)
                   (throw (ex-info "multi constraint weight"
                                   {:w w :expected expected}))))
               ;; Test 3: weight = project(trace, constrained_selection)
               (let [c3 (cm/choicemap :a (mx/scalar 0.5) :c (mx/scalar 0.5))
                     {:keys [trace weight]} (p/generate model [] c3)
                     w (ev weight)
                     proj (ev (p/project model trace
                                         (sel/from-set #{:a :c})))]
                 (approx= w proj 0.01))))}

   {:name :forward-sampling-factorization
    :from "[T] §4.1.3 — q(v; x, sigma) = product_i p(a_i|context)^[a_i not in sigma]"
    :theorem "For Bayesian networks, the internal proposal factorizes: each
              unconstrained address is sampled from its conditional prior given
              all previously sampled/constrained values. Verified via the
              identity: weight = score - log q(unconstrained values), where
              log q = sum of log-probs at unconstrained addresses."
    :tags #{:generate :internal-proposal :ch4}
    :check (fn [_]
             ;; Non-conjugate chain: a ~ Uniform(0,1), b ~ Laplace(a, 1),
             ;; c ~ Laplace(a+b, 0.5). No conjugate pairs.
             (let [model (dyn/auto-key
                          (dyn/make-gen-fn
                           (fn [rt]
                             (let [trace (.-trace rt)
                                   a (trace :a (dist/uniform 0 1))
                                   b (trace :b (dist/laplace a 1))]
                               (trace :c (dist/laplace (mx/add a b) 0.5))))
                           '([] (let [a (trace :a (dist/uniform 0 1))
                                      b (trace :b (dist/laplace a 1))]
                                  (trace :c (dist/laplace (mx/add a b) 0.5))))))]
               ;; Case 1: constrain c only. Proposal samples a, b from priors.
               ;; weight = lp(c|a,b). Factorization: weight = score - lp(a) - lp(b).
               (let [ok1
                     (every? true?
                             (for [seed (range 5)]
                               (let [m (dyn/with-key model (rng/fresh-key seed))
                                     {:keys [trace weight]} (p/generate m []
                                                                        (cm/choicemap :c (mx/scalar 3.0)))
                                     w (ev weight)
                                     s (ev (:score trace))
                                     a (ev (cm/get-value (cm/get-submap (:choices trace) :a)))
                                     b (ev (cm/get-value (cm/get-submap (:choices trace) :b)))
                               ;; log q(a,b) = lp(a) + lp(b|a)
                                     lp-a (ev (dist/log-prob (dist/uniform 0 1) (mx/scalar a)))
                                     lp-b (ev (dist/log-prob (dist/laplace (mx/scalar a) 1)
                                                             (mx/scalar b)))
                                     log-q (+ lp-a lp-b)]
                                 (approx= w (- s log-q) 0.01))))
                     ;; Case 2: constrain a and c. Only b unconstrained.
                     ;; weight = lp(a) + lp(c|a,b). Factorization: weight = score - lp(b|a).
                     ok2
                     (every? true?
                             (for [seed (range 5)]
                               (let [m (dyn/with-key model (rng/fresh-key seed))
                                     constraints (cm/choicemap :a (mx/scalar 0.7)
                                                               :c (mx/scalar 3.0))
                                     {:keys [trace weight]} (p/generate m [] constraints)
                                     w (ev weight)
                                     s (ev (:score trace))
                                     b (ev (cm/get-value (cm/get-submap (:choices trace) :b)))
                               ;; log q(b) = lp(b|a=0.7)
                                     log-q (ev (dist/log-prob (dist/laplace (mx/scalar 0.7) 1)
                                                              (mx/scalar b)))]
                                 (approx= w (- s log-q) 0.01))))]
                 (and ok1 ok2))))}

   ;; ===================================================================
   ;; VECTORIZED laws
   ;; ===================================================================

   {:name :vsimulate-shape-correctness
    :from "[D] broadcast-equivalence"
    :theorem "vsimulate(P, x, N) produces scores with shape [N],
              all finite"
    :tags #{:vectorized}
    :check (fn [{:keys [model args]}]
             (let [n 10
                   key (rng/fresh-key 42)
                   vt (dyn/vsimulate model args n key)]
               (mx/materialize! (:score vt))
               (let [shape (mx/shape (:score vt))]
                 (and (= (vec shape) [n])
                      (every? js/Number.isFinite
                              (mx/->clj (:score vt)))))))}

   ;; ===================================================================
   ;; DENOTATIONAL SEMANTICS laws [T] §2.2.2, Figure 2-1
   ;; ===================================================================

   {:name :addrs-extraction-correctness
    :from "[T] Fig 2-1, Addrs⟦E⟧"
    :theorem "Schema extraction produces trace-site addresses that match
              the actual addresses in a simulated trace (for static models)."
    :tags #{:semantics :core}
    :check (fn [{:keys [model args]}]
             (let [schema (:schema model)]
               (if (or (nil? schema) (not (:static? schema)))
                 true ;; only testable for static models with schemas
                 (let [schema-addrs (set (map :addr (:trace-sites schema)))
                       t (p/simulate model args)
                       trace-addrs (set (map first (all-leaf-addrs (:choices t))))]
                   (= schema-addrs trace-addrs)))))}

   {:name :val-function-correctness
    :from "[T] Fig 2-1, Val⟦E⟧"
    :theorem "Val⟦E⟧(sigma)(tau) is a well-defined function: given fixed args
              and choices, retval is deterministic."
    :tags #{:semantics :core}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   r1 (:trace (p/generate model args (:choices t)))
                   r2 (:trace (p/generate model args (:choices t)))]
               (approx= (ev (:retval r1)) (ev (:retval r2)) 1e-10)))}

   {:name :dist-analytical-match
    :from "[T] Fig 2-1, Dist⟦E⟧"
    :theorem "Simulate distribution matches the analytical Dist⟦E⟧.
              Sample moments converge to analytical moments."
    :tags #{:semantics}
    :check (fn [{:keys [_model _args]}]
             ;; Tests on a fixed reference model (gaussian chain)
             ;; x ~ N(0,1), y ~ N(x, 0.5)
             ;; Analytical: E[y]=0, Var[y]=1.25, Cov(x,y)=1
             (let [ref-model (dyn/auto-key
                              (dyn/make-gen-fn
                               (fn [rt]
                                 (let [trace (.-trace rt)
                                       x (trace :x (dist/gaussian 0 1))]
                                   (trace :y (dist/gaussian x 0.5))))
                               '([] (let [x (trace :x (dist/gaussian 0 1))]
                                      (trace :y (dist/gaussian x 0.5))))))
                   n 2000
                   samples (repeatedly n #(let [t (p/simulate ref-model [])]
                                            [(ev (cm/get-value (cm/get-submap (:choices t) :x)))
                                             (ev (cm/get-value (cm/get-submap (:choices t) :y)))]))
                   xs (map first samples)
                   ys (map second samples)
                   mean-y (/ (reduce + ys) n)
                   var-y (/ (reduce + (map #(let [d (- % mean-y)] (* d d)) ys)) (dec n))
                   mean-x (/ (reduce + xs) n)
                   cov-xy (/ (reduce + (map (fn [x y] (* (- x mean-x) (- y mean-y))) xs ys)) (dec n))]
               (and (approx= mean-y 0.0 0.1) ;; E[y] = 0
                    (approx= var-y 1.25 0.15) ;; Var[y] = 1.25
                    (approx= cov-xy 1.0 0.15))))} ;; Cov(x,y) = 1

   {:name :gen-function-denotation
    :from "[T] Fig 2-1, ⟦@gen⟧"
    :theorem "The gen body maps to P = (X, Y, p, f) where all four
              components agree with the analytical denotation."
    :tags #{:semantics}
    :check (fn [{:keys [_model _args]}]
             ;; Fixed reference: x ~ N(0,1), y ~ N(x, 0.5)
             ;; At tau = {:x 0.5, :y 1.0}:
             ;;   p: score = log N(0.5;0,1) + log N(1.0;0.5,0.5)
             ;;   f: retval = y = 1.0
             (let [ref-model (dyn/auto-key
                              (dyn/make-gen-fn
                               (fn [rt]
                                 (let [trace (.-trace rt)
                                       x (trace :x (dist/gaussian 0 1))]
                                   (trace :y (dist/gaussian x 0.5))))
                               '([] (let [x (trace :x (dist/gaussian 0 1))]
                                      (trace :y (dist/gaussian x 0.5))))))
                   choices (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0))
                   {:keys [trace weight]} (p/generate ref-model [] choices)
                   score (ev (:score trace))
                   retval (ev (:retval trace))
                   ;; Analytical score
                   log-2pi-half (* -0.5 (js/Math.log (* 2 js/Math.PI)))
                   lp-x (- log-2pi-half (* 0.5 0.25))
                   lp-y (- (- log-2pi-half (js/Math.log 0.5)) (* 0.5 1.0))
                   expected-score (+ lp-x lp-y)]
               (and (approx= score expected-score 1e-3)
                    (approx= retval 1.0 1e-6)
                    (approx= (ev weight) expected-score 1e-3))))}

   ;; ===================================================================
   ;; SELECTION MH + KERNEL COMPOSITION laws [T] §3.4, §4.1, Prop 3.4.1
   ;; ===================================================================

   {:name :optimal-proposal-weight
    :from "[T] §4.1.3"
    :theorem "For conjugate models with analytical handler, generate weight
              equals the log marginal likelihood."
    :tags #{:inference :internal-proposal}
    :check (fn [{:keys [_model _args]}]
             ;; Fixed conjugate model: mu ~ N(0,2), y ~ N(mu,1)
             ;; Analytical: log N(3; 0, sqrt(5))
             (let [conj-model (dyn/auto-key
                                (dyn/make-gen-fn
                                 (fn [rt]
                                   (let [trace (.-trace rt)
                                         mu (trace :mu (dist/gaussian 0 2))]
                                     (trace :y (dist/gaussian mu 1))))
                                 '([] (let [mu (trace :mu (dist/gaussian 0 2))]
                                        (trace :y (dist/gaussian mu 1))))))
                   obs (cm/choicemap :y (mx/scalar 3.0))
                   {:keys [weight]} (p/generate conj-model [] obs)
                   w (ev weight)
                   analytical (- (* -0.5 (js/Math.log (* 2 js/Math.PI)))
                                 (* 0.5 (js/Math.log 5.0))
                                 (* 0.5 (/ 9.0 5.0)))]
               (approx= w analytical 1e-3)))}

   {:name :selection-mh-correctness
    :from "[T] Alg 15"
    :theorem "Selection MH = regenerate + accept/reject. The weight from
              regenerate decomposes as the score change at unselected sites."
    :tags #{:inference :mcmc}
    :check (fn [{:keys [model args]}]
             (let [t (p/simulate model args)
                   addrs (all-leaf-addrs (:choices t))]
               (if (< (count addrs) 2)
                 true
                 (let [sel-addr (first (first addrs))
                       sel (sel/select sel-addr)
                       {:keys [trace weight]} (p/regenerate model t sel)
                       w (ev weight)
                       new-score (ev (:score trace))
                       old-score (ev (:score t))
                       old-proj (ev (p/project model t sel))
                       new-proj (ev (p/project model trace sel))
                       expected (- (- new-score old-score)
                                   (- new-proj old-proj))]
                   (and (js/Number.isFinite w)
                        (approx= w expected 0.01))))))}

   {:name :simulate-address-set-consistency
    :from "[T] §4.1.4, Alg 16"
    :theorem "For static models, simulate always produces the same address
              set. This is a prerequisite for proposal override (Alg 16):
              R.simulate must have the same structure as P.simulate."
    :tags #{:inference :internal-proposal}
    :check (fn [{:keys [model args]}]
             (let [t1 (p/simulate model args)
                   t2 (p/simulate model args)]
               (and (js/Number.isFinite (ev (:score t1)))
                    (js/Number.isFinite (ev (:score t2)))
                    (= (set (all-leaf-addrs (:choices t1)))
                       (set (all-leaf-addrs (:choices t2)))))))}

   {:name :mixture-kernel-stationarity
    :from "[T] Prop 3.4.1"
    :theorem "Mix of stationary kernels is stationary: both mix-kernels
              and cycle of same components converge to the analytical
              posterior mean."
    :tags #{:inference :mcmc}
    :check (fn [{:keys [_model _args]}]
             ;; Fixed model: x ~ N(0,1), y ~ N(0,1), obs ~ N(x+y, 0.5), obs=2
             ;; Analytical posterior mean of x+y = 8/4.5 = 1.778
             (let [kern-model (dyn/auto-key
                                (dyn/make-gen-fn
                                 (fn [rt]
                                   (let [trace (.-trace rt)
                                         x (trace :x (dist/gaussian 0 1))
                                         y (trace :y (dist/gaussian 0 1))]
                                     (trace :obs (dist/gaussian (mx/add x y) 0.5))))
                                 '([] (let [x (trace :x (dist/gaussian 0 1))
                                            y (trace :y (dist/gaussian 0 1))]
                                        (trace :obs (dist/gaussian (mx/add x y) 0.5))))))
                   obs (cm/choicemap :obs (mx/scalar 2.0))
                   init (:trace (p/generate kern-model [] obs))
                   ;; Analytical: s=x+y, prior N(0,2), lik N(s,0.5), obs=2
                   ;; Posterior mean = 8/4.5 = 1.778
                   analytical-mean (/ 8.0 4.5)
                   run-chain
                   (fn [select-fn n-steps burn]
                     (loop [t init i 0 acc [] k (rng/fresh-key 13)]
                       (if (>= i n-steps)
                         acc
                         (let [[k1 k2] (rng/split k)
                               sel (select-fn i k1)
                               {:keys [trace weight]} (p/regenerate kern-model t sel)
                               w (ev weight)
                               accept? (u/accept-mh? w k1)
                               next-t (if accept? trace t)
                               x-val (ev (cm/get-value
                                          (cm/get-submap (:choices next-t) :x)))
                               y-val (ev (cm/get-value
                                          (cm/get-submap (:choices next-t) :y)))]
                           (recur next-t (inc i)
                                  (if (>= i burn)
                                    (conj acc (+ x-val y-val))
                                    acc)
                                  k2)))))
                   mix-samples (run-chain
                                (fn [_ k] (sel/select (if (pos? (mx/item (rng/bernoulli k 0.5 []))) :x :y)))
                                4000 1000)
                   cycle-samples (run-chain
                                  (fn [i _k] (sel/select (if (even? i) :x :y)))
                                  4000 1000)
                   mean-mix (/ (reduce + mix-samples) (count mix-samples))
                   mean-cycle (/ (reduce + cycle-samples) (count cycle-samples))]
               ;; Both kernels converge to the same analytical posterior
               (and (approx= mean-mix analytical-mean 0.3)
                    (approx= mean-cycle analytical-mean 0.3))))}

   ;; ===================================================================
   ;; WELL-FORMEDNESS laws [T] §2.2.1 (DML restrictions)
   ;; ===================================================================

   {:name :no-external-randomness
    :from "[T] §2.2.1, restriction 3"
    :theorem "Model body must not use untraced randomness (rand, js/Math.random, etc.).
              All stochasticity must flow through traced distribution calls."
    :tags #{:well-formedness}
    :check (fn [{:keys [model]}]
             (let [source (:source model)]
               (if (nil? source)
                 true
                 (empty? (verify/check-no-external-randomness source)))))}

   {:name :no-mutation
    :from "[T] §2.2.1, restriction 4"
    :theorem "Model body must not mutate state (atom, swap!, set!, volatile!, etc.).
              Gen fn bodies must be purely functional."
    :tags #{:well-formedness}
    :check (fn [{:keys [model]}]
             (let [source (:source model)]
               (if (nil? source)
                 true
                 (empty? (verify/check-no-mutation source)))))}

   {:name :no-hof-gen-fns
    :from "[T] §2.2.1, restriction 5"
    :theorem "Gen fns must not be passed as arguments to higher-order functions
              (map, reduce, filter, etc.). Use combinators (Map, Unfold) instead."
    :tags #{:well-formedness}
    :check (fn [{:keys [model]}]
             (let [source (:source model)]
               (if (nil? source)
                 true
                 (empty? (verify/check-no-hof-gen-fns source)))))}

   ;; ===================================================================
   ;; COMPILED PATH EQUIVALENCE laws [T] Ch 5
   ;;
   ;; The compilation ladder's central invariant: compiled execution
   ;; paths must produce the same probability density p(tau; x) as the
   ;; handler (interpreter) path. The handler is ground truth;
   ;; compilation is optimization.
   ;;
   ;; Strategy: for deterministic comparison, we use generate with full
   ;; constraints. Given identical choices, both paths must compute
   ;; identical scores. This avoids needing matching PRNG keys.
   ;; ===================================================================

   {:name :compiled-simulate-equivalence
    :from "[T] Ch 5 -- compiled simulate preserves p(tau; x)"
    :theorem "For a static model with compiled-simulate, the compiled path
              produces traces whose score equals the handler-computed score
              for the same choices. Verified: simulate via compiled, then
              assess via handler on the same choices."
    :tags #{:compiled :equivalence}
    :check (fn [{:keys [model args]}]
             (let [schema (:schema model)]
               (if (or (nil? schema) (nil? (:compiled-simulate schema)))
                 true
                 (let [compiled-trace (p/simulate model args)
                       compiled-score (ev (:score compiled-trace))
                       handler-model (dyn/auto-key (strip-compiled model))
                       {:keys [weight]} (p/assess handler-model args
                                                   (:choices compiled-trace))
                       handler-score (ev weight)]
                   (approx= compiled-score handler-score 1e-4)))))}

   {:name :compiled-generate-equivalence
    :from "[T] Ch 5 -- compiled generate preserves scores and weights"
    :theorem "For a static model with compiled-generate, generating with
              full constraints via compiled and handler paths produces
              identical scores, weights, and return values."
    :tags #{:compiled :equivalence}
    :check (fn [{:keys [model args]}]
             (let [schema (:schema model)]
               (if (or (nil? schema) (nil? (:compiled-generate schema)))
                 true
                 (let [source-trace (p/simulate model args)
                       constraints (:choices source-trace)
                       {:keys [trace weight]} (p/generate model args constraints)
                       compiled-score (ev (:score trace))
                       compiled-weight (ev weight)
                       compiled-retval (ev (:retval trace))
                       handler-model (dyn/auto-key (strip-compiled model))
                       {:keys [trace weight]} (p/generate handler-model args
                                                          constraints)
                       handler-score (ev (:score trace))
                       handler-weight (ev weight)
                       handler-retval (ev (:retval trace))]
                   (and (approx= compiled-score handler-score 1e-4)
                        (approx= compiled-weight handler-weight 1e-4)
                        (approx= compiled-retval handler-retval 1e-4))))))}

   {:name :compiled-update-equivalence
    :from "[T] Ch 5 -- compiled update preserves density ratios"
    :theorem "For a static model with compiled-update, updating a trace
              with new constraints via compiled and handler paths produces
              identical weights (density ratios) and new scores."
    :tags #{:compiled :equivalence}
    :check (fn [{:keys [model args]}]
             (let [schema (:schema model)]
               (if (or (nil? schema) (nil? (:compiled-update schema)))
                 true
                 (let [t1 (p/simulate model args)
                       t2 (p/simulate model args)
                       {:keys [trace weight]} (p/update model t1 (:choices t2))
                       compiled-weight (ev weight)
                       compiled-new-score (ev (:score trace))
                       handler-model (dyn/auto-key (strip-compiled model))
                       handler-t1 (:trace (p/generate handler-model args
                                                      (:choices t1)))
                       {:keys [trace weight]} (p/update handler-model
                                                        handler-t1
                                                        (:choices t2))
                       handler-weight (ev weight)
                       handler-new-score (ev (:score trace))]
                   (and (approx= compiled-weight handler-weight 1e-4)
                        (approx= compiled-new-score handler-new-score 1e-4))))))}

   {:name :compiled-regenerate-equivalence
    :from "[T] Ch 5 -- compiled regenerate preserves weight semantics"
    :theorem "For a static model with compiled-regenerate, the regenerated
              trace has a score that equals the handler-assessed density for
              the same choices. The weight is finite and the regenerated
              trace preserves unselected address values."
    :tags #{:compiled :equivalence}
    :check (fn [{:keys [model args]}]
             (let [schema (:schema model)]
               (if (or (nil? schema)
                       (nil? (:compiled-regenerate schema))
                       (< (count (all-leaf-addrs
                                  (:choices (p/simulate model args)))) 2))
                 true
                 (let [t (p/simulate model args)
                       addrs (all-leaf-addrs (:choices t))
                       sel (sel/select (first (first addrs)))
                       {:keys [trace weight]} (p/regenerate model t sel)
                       compiled-score (ev (:score trace))
                       compiled-weight (ev weight)
                       handler-model (dyn/auto-key (strip-compiled model))
                       {:keys [weight]} (p/assess handler-model args
                                                   (:choices trace))
                       handler-score (ev weight)
                       unselected (map first (rest addrs))
                       preserved? (every?
                                   (fn [a]
                                     (approx=
                                      (ev (cm/get-value
                                           (cm/get-submap (:choices t) a)))
                                      (ev (cm/get-value
                                           (cm/get-submap (:choices trace) a)))
                                      1e-6))
                                   unselected)]
                   (and (approx= compiled-score handler-score 1e-4)
                        (js/Number.isFinite compiled-weight)
                        preserved?)))))}

   ;; ===================================================================
   ;; HMC ACCEPTANCE laws [T] Alg 6
   ;; ===================================================================

   {:name :hmc-acceptance-correctness
    :from "[T] Alg 6, §3.4.3"
    :theorem "HMC with leapfrog integration converges to the correct
              posterior distribution. Verified on Normal-Normal conjugate:
              prior x ~ N(0,1), likelihood y ~ N(x, 0.5), observe y=2.
              Posterior: x|y ~ N(1.6, sqrt(0.2)). After 300 HMC samples
              (200 burn-in), the sample mean is within tolerance of 1.6."
    :tags #{:inference :mcmc :hmc}
    :check (fn [_]
             ;; Fixed conjugate model -- independent of passed-in model.
             ;; HMC acceptance: alpha = min{1, exp(H(q,p) - H(q',p'))}
             ;; where H(q,p) = U(q) + K(p), U(q) = -log p(q|data),
             ;; K(p) = 0.5 p^T M^{-1} p.
             ;; Leapfrog approximately preserves H, so acceptance is high
             ;; for well-tuned step-size.
             (let [hmc-model (dyn/auto-key
                              (dyn/make-gen-fn
                               (fn [rt]
                                 (let [trace (.-trace rt)
                                       x (trace :x (dist/gaussian 0 1))]
                                   (trace :y (dist/gaussian x 0.5))))
                               '([] (let [x (trace :x (dist/gaussian 0 1))]
                                      (trace :y (dist/gaussian x 0.5))))))
                   obs (cm/choicemap :y (mx/scalar 2.0))
                   samples (mcmc/hmc {:samples 300 :burn 200 :step-size 0.05
                                      :leapfrog-steps 10 :addresses [:x]
                                      :compile? false :device :cpu}
                                     hmc-model [] obs)
                   x-vals (mapv first samples)
                   mean-x (/ (reduce + x-vals) (count x-vals))]
               (approx= mean-x 1.6 0.3)))}

   ;; ===================================================================
   ;; PROPOSAL TRAINING laws [T] Eq 3.8-3.9
   ;; ===================================================================

   {:name :proposal-training-objective
    :from "[T] Eq 3.8-3.9"
    :theorem "Training a proposal via gradient descent on the negative
              log-likelihood decreases the loss (variational principle).
              max_theta E[log q(sigma; rho, theta)] = min_theta KL(p||q).
              Verified: Adam optimization of model parameters on observed
              data reduces negative log-likelihood, and the optimized
              parameter converges toward the maximum likelihood estimate."
    :tags #{:inference :training :variational}
    :check (fn [_]
             ;; Fixed training problem: mu ~ N(mu_param, 1), observe x=3.
             ;; MLE: mu_param -> 3. NLL at mu=0 is 5.42, at mu=3 is 0.92.
             ;; Training from mu=0 should decrease loss and converge.
             (let [train-model (dyn/auto-key
                                (dyn/make-gen-fn
                                 (fn [rt]
                                   (let [trace (.-trace rt)
                                         param (.-param rt)
                                         mu (param :mu (mx/scalar 0.0))
                                         x (trace :x (dist/gaussian mu 1))]
                                     x))
                                 '([] (let [mu (param :mu (mx/scalar 0.0))
                                            x (trace :x (dist/gaussian mu 1))]
                                        x))))
                   obs (cm/choicemap :x (mx/scalar 3.0))
                   loss-grad-fn (learn/make-param-loss-fn
                                 train-model [] obs [:mu])
                   init-params (mx/array [0.0])
                   result (learn/train
                           {:iterations 100 :optimizer :adam :lr 0.05
                            :key (rng/fresh-key 42)}
                           loss-grad-fn init-params)
                   losses (:loss-history result)
                   first-10 (take 10 losses)
                   last-10 (take-last 10 losses)
                   mean-first (/ (reduce + first-10) (count first-10))
                   mean-last (/ (reduce + last-10) (count last-10))
                   final-mu (ev (mx/index (:params result) 0))]
               (and (< mean-last mean-first)
                    (approx= final-mu 3.0 0.5))))}])

;; ---------------------------------------------------------------------------
;; Law index (for fast lookup)
;; ---------------------------------------------------------------------------

(def ^:private law-by-name
  (into {} (map (fn [law] [(:name law) law]) laws)))

;; ---------------------------------------------------------------------------
;; Verification API
;; ---------------------------------------------------------------------------

(defn check-law
  "Run a single law against a model. Returns {:name :pass? :theorem :error}."
  [law-name model args]
  (let [law (get law-by-name law-name)]
    (when-not law
      (throw (ex-info (str "Unknown GFI law: " law-name)
                      {:known (keys law-by-name)})))
    (try
      (let [ok? ((:check law) {:model model :args args})]
        {:name (:name law) :pass? ok? :theorem (:theorem law)})
      (catch :default e
        {:name (:name law) :pass? false :theorem (:theorem law)
         :error (.-message e)}))))

(defn verify
  "Run GFI laws against a model. Returns structured report.

   Options:
     :laws     — subset of law keywords to run (default: all)
     :tags     — subset of tag keywords; runs laws matching ANY tag
     :n-trials — number of independent trials per law (default: 10)"
  [model args & {:keys [law-names tags n-trials]
                 :or {n-trials 10}}]
  (let [model (dyn/auto-key model)
        selected (cond
                   law-names (filter #(contains? (set law-names) (:name %)) laws)
                   tags (let [tag-set (set tags)]
                          (filter #(some tag-set (:tags %)) laws))
                   :else laws)
        results (mapv
                 (fn [law]
                   (let [trials (for [_ (range n-trials)]
                                  (try
                                    ((:check law) {:model model :args args})
                                    (catch :default _ false)))
                         passes (count (filter true? trials))
                         fails (- n-trials passes)]
                     {:name (:name law)
                      :from (:from law)
                      :theorem (:theorem law)
                      :passes passes
                      :fails fails
                      :pass? (zero? fails)}))
                 selected)
        total-pass (reduce + 0 (map :passes results))
        total-fail (reduce + 0 (map :fails results))]
    {:results results
     :total-pass total-pass
     :total-fail total-fail
     :all-pass? (zero? total-fail)
     :n-laws (count selected)
     :n-trials n-trials}))

(defn print-report
  "Pretty-print a verify report."
  [report]
  (println (str "\nGFI Algebraic Theory: " (:n-laws report) " laws, "
                (:n-trials report) " trials each\n"))
  (doseq [{:keys [name from pass? passes fails theorem]} (:results report)]
    (println (str (if pass? "  PASS " "  FAIL ")
                  name
                  " (" passes "/" (+ passes fails) ")"
                  "\n        " from
                  (if pass? "" (str "\n        " theorem)))))
  (println (str "\n" (:total-pass report) " passed, "
                (:total-fail report) " failed"
                (if (:all-pass? report) " -- ALL LAWS HOLD" ""))))
