(ns agents-llm-policy
  "FLAGSHIP — agents × llm (ROADMAP Phase 3, item 4, Direction 1: LLM-as-policy).

   The genmlx.agents thesis is that an agent is a generative function and its
   POLICY is a pluggable GF. This example shows that the pluggable policy can be
   an LLM: the per-action policy logits ARE the LLM generative function's own
   per-action log-probabilities, obtained through the GFI (`p/assess` on the LLM
   GF), and the action is then drawn by an ordinary categorical policy GF. The
   LLM GF (genmlx.llm.core/make-llm-gf) sits in policy position, composed with the
   genmlx.agents MDP machinery purely through the GFI — no engine change, no custom
   dispatch.

   Concretely, on a tiny 3×3 gridworld:
     1. Describe the current state in natural language ('You are at row R, col C.
        The goal is at row 2, col 2. ... Best move:').
     2. For each action word in {' left',' right',' up',' down'} (aligned to
        gridworld action indices), score it with `p/assess` on the LLM GF →
        log p(action word | state prompt). These four numbers are the policy logits.
     3. The policy is `(gen [logits] (trace :action (dist/categorical logits)))`,
        a generative function whose action distribution is the LLM's. p/simulate
        samples an action, p/assess scores one, p/generate constrains one — all
        through the standard handler.
     4. Roll the LLM-policy agent out with the MDP's transition (:ns-fn).

   HONESTY: a 0.6B base model is NOT a competent gridworld planner. This example
   proves the COMPOSITION (an LLM GF as a policy, scored through the GFI), not that
   the LLM plans well. It asserts only structural GFI correctness (valid action
   site; the policy's p/assess equals logsoftmax of the LLM logits; p/generate over
   the four actions has probabilities summing to 1; reproducibility). The LLM's
   action distribution is printed for inspection, never asserted to be optimal.

   Reuse, zero engine change: genmlx.llm.core/make-llm-gf (verified by
   llm_core_test, 36/36, qwen3-0.6b), genmlx.agents.gridworld/build-mdp, the GFI
   (p/simulate, p/assess, p/generate). Self-checking + skip-clean if the model is
   absent. Run:

     bun run --bun nbb examples/agents_llm_policy.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx.random :as rng]
            [genmlx.gen :refer [gen]]
            [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [promesa.core :as pr]
            ["fs" :as fs]))

;; ---------------------------------------------------------------------------
;; tiny self-check harness
;; ---------------------------------------------------------------------------

(def ^:private fails (atom 0))

(defn- ->num [v] (if (mx/array? v) (mx/item v) v))

(defn- check-close [label expected actual tol]
  (let [a (->num actual)
        ok (<= (abs (- expected a)) tol)]
    (println (str (if ok "  ✓ " "  ✗ FAIL ") label " — expected " expected ", got " (.toFixed a 6)))
    (when-not ok (swap! fails inc))
    ok))

(defn- check-true [label ok]
  (println (str (if ok "  ✓ " "  ✗ FAIL ") label))
  (when-not ok (swap! fails inc))
  ok)

;; ---------------------------------------------------------------------------
;; environment: a tiny 3×3 gridworld; goal at bottom-right
;; ---------------------------------------------------------------------------

(def grid
  [[:empty :empty :empty]
   [:empty :empty :empty]
   [:empty :empty :goal]])

(def mdp (gw/build-mdp {:grid grid
                        :utilities {:goal 1.0 :timeCost -0.05}
                        :start [0 0] :gamma 1.0 :noise 0.0}))

(def W (:W mdp))
;; action words ALIGNED to genmlx.agents.gridworld action-kw [:left :right :up :down]
(def action-words [" left" " right" " up" " down"])
(def goal-idx (some (fn [[idx kw]] (when (= kw :goal) idx)) (:terminals mdp)))

(defn s->rc [s] [(quot s W) (rem s W)])

(defn describe-state
  "A compact natural-language prompt asking the LLM for the best gridworld move."
  [s]
  (let [[r c]   (s->rc s)
        [gr gc] (s->rc goal-idx)]
    (str "You are an agent on a 3x3 grid. Rows are numbered 0 (top) to 2 (bottom) "
         "and columns 0 (left) to 2 (right). You are at row " r ", column " c ". "
         "The goal is at row " gr ", column " gc ". "
         "Choose the single best move to reach the goal. Options: left, right, up, down. "
         "Best move:")))

;; ---------------------------------------------------------------------------
;; the LLM in policy position: per-action log-probs THROUGH the GFI (p/assess)
;; ---------------------------------------------------------------------------

(defn action-logprob
  "log p(action tokens | state prompt) via the LLM GF's GFI assess op. `act-ids`
   is the pre-encoded token-id vector of the action word (encode is async, so it
   is awaited by the caller and threaded in here as plain ids)."
  [llm-gf prompt-ids act-ids]
  (let [n           (count act-ids)
        constraints (reduce (fn [cm i]
                              (cm/set-value cm (keyword (str "t" i))
                                            (mx/scalar (nth act-ids i) mx/int32)))
                            cm/EMPTY (range n))]
    (->num (:weight (p/assess llm-gf [prompt-ids n] constraints)))))

(defn llm-action-logits
  "The four policy logits at a state: each is the LLM GF's assess log-prob of the
   corresponding action word given the (pre-encoded) state prompt. Pure/sync —
   `prompt-ids` and `act-id-lists` are already-encoded token-id vectors."
  [llm-gf prompt-ids act-id-lists]
  {:logits (mapv #(action-logprob llm-gf prompt-ids %) act-id-lists)
   :ntoks  (mapv count act-id-lists)})

;; the policy: an ordinary categorical GF whose logits are the LLM's per-action scores
(def action-policy
  (gen [logits] (trace :action (dist/categorical logits))))

(defn- logsoftmax-at
  "Host-side log_softmax(logits)[a] for cross-checking the GFI scores."
  [logits a]
  (let [m   (apply max logits)
        lse (+ m (js/Math.log (reduce + (map #(js/Math.exp (- % m)) logits))))]
    (- (nth logits a) lse)))

(defn- softmax [logits]
  (let [m  (apply max logits)
        es (mapv #(js/Math.exp (- % m)) logits)
        z  (reduce + es)]
    (mapv #(/ % z) es)))

;; ---------------------------------------------------------------------------
;; rollout: state → LLM-policy action → MDP transition
;; ---------------------------------------------------------------------------

(defn encode-prompt
  "Async: encode the state-prompt for state `s` into a token-id vector."
  [tok s]
  (pr/let [ids (llm/encode tok (describe-state s))] (vec ids)))

(defn rollout
  "Async: roll the LLM-policy agent from `start` for ≤ k steps. Illustrative (fresh
   entropy per step). `act-id-lists` is the pre-encoded action vocabulary. Resolves
   to a vector of {:state :action :probs}."
  [llm-gf tok act-id-lists start k]
  (let [terms (set (keys (:terminals mdp)))
        ns-fn (:ns-fn mdp)]
    (letfn [(step [s i acc]
              (if (or (>= i k) (contains? terms s))
                (pr/resolved acc)
                (pr/let [pid    (encode-prompt tok s)
                         res    (llm-action-logits llm-gf pid act-id-lists)
                         logmx  (mx/array (:logits res) mx/float32)
                         a      (int (->num (:retval (p/simulate (dyn/auto-key action-policy) [logmx]))))
                         s'     (ns-fn s a)]
                  (step s' (inc i) (conj acc {:state s :action a :probs (softmax (:logits res))})))))]
      (step start 0 []))))

;; ---------------------------------------------------------------------------
;; run + self-check (async: model load + tokenizer encode are promises)
;; ---------------------------------------------------------------------------

(def model-path (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-mlx-bf16"))

(defn run [m]
  (pr/let [tok          (:tokenizer m)
           llm-gf       (llm-core/make-llm-gf m)
           act-id-lists (pr/all (mapv (fn [w] (pr/let [ids (llm/encode tok w)] (vec ids))) action-words))
           start        (:start-idx mdp)
           start-pid    (encode-prompt tok start)
           ;; three distinct cells for the state-dependence check (idx 0,4,7)
           demo-states  [0 4 7]
           demo-pids    (pr/all (mapv #(encode-prompt tok %) demo-states))]
    (let [optimal (agent/make-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 24})
          {:keys [logits ntoks]} (llm-action-logits llm-gf start-pid act-id-lists)
          probs (softmax logits)
          logmx (mx/array logits mx/float32)]
      (println "\n=== agents × llm flagship — LLM as an agent policy (GFI nesting) ===")
      (println (str "  grid 3x3, goal at idx " goal-idx " (row 2,col 2); actions "
                    (mapv (comp keyword #(subs % 1)) action-words)))

      (println "\n-- LLM policy logits at the start state (via p/assess on the LLM GF) --")
      (println (str "  prompt: " (pr-str (describe-state start))))
      (println (str "  action words: " action-words "  (token counts " ntoks ")"))
      (println (str "  LLM log p(action|state): " (mapv #(.toFixed % 3) logits)))
      (println (str "  policy π(action|state)  : " (mapv #(.toFixed % 3) probs)
                    "   optimal action = " ((:act optimal) start)))
      (check-true "all four LLM action log-probs are finite" (every? js/isFinite logits))
      (check-close "policy distribution sums to 1" 1.0 (reduce + probs) 1e-5)

      ;; --- GFI proof 1: the policy IS a generative function ---
      (println "\n-- GFI proof: the LLM policy is a generative function --")
      (let [sim   (p/simulate (dyn/auto-key action-policy) [logmx])
            a-sim (int (->num (:retval sim)))]
        (check-true "p/simulate yields a valid :action site in {0..3}"
                    (and (contains? #{0 1 2 3} a-sim)
                         (cm/has-value? (cm/get-submap (:choices sim) :action)))))

      ;; --- GFI proof 2: p/assess weight == logsoftmax(LLM logits)[a] ---
      (doseq [a (range 4)]
        (let [w (->num (:weight (p/assess (dyn/auto-key action-policy) [logmx]
                                          (cm/choicemap :action (mx/scalar a mx/int32)))))]
          (check-close (str "p/assess(:action=" a ") == logsoftmax(LLM logits)[" a "]")
                       (logsoftmax-at logits a) w 1e-3)))

      ;; --- GFI proof 3: p/generate forcing each action; Σ exp(weight) == 1 ---
      (let [ws (mapv (fn [a]
                       (->num (:weight (p/generate (dyn/auto-key action-policy) [logmx]
                                                   (cm/choicemap :action (mx/scalar a mx/int32))))))
                     (range 4))]
        (check-true "p/generate weights all finite" (every? js/isFinite ws))
        (check-close "Σ exp(p/generate weight over the 4 actions) == 1"
                     1.0 (reduce + (map js/Math.exp ws)) 1e-3))

      ;; --- GFI proof 4: reproducibility (same key → same action) ---
      (let [a1 (int (->num (:retval (p/simulate (dyn/with-key action-policy (rng/fresh-key 7)) [logmx]))))
            a2 (int (->num (:retval (p/simulate (dyn/with-key action-policy (rng/fresh-key 7)) [logmx]))))
            a3 (int (->num (:retval (p/simulate (dyn/with-key action-policy (rng/fresh-key 8)) [logmx]))))]
        (check-true "same key → same sampled action" (= a1 a2))
        (check-true "reproducible draws are valid actions" (every? #{0 1 2 3} [a1 a2 a3])))

      ;; --- GFI proof 5: the policy logit IS the LLM GF assess of that action word ---
      (let [llm-direct (action-logprob llm-gf start-pid (nth act-id-lists 1))]
        (check-close "policy logit[:right] == LLM GF p/assess of ' right'"
                     (nth logits 1) llm-direct 1e-4))

      ;; --- state-dependence: the LLM policy genuinely conditions on the state ---
      (println "\n-- the LLM policy conditions on state (distribution varies per cell) --")
      (let [dists (mapv (fn [s pid]
                          (let [d (softmax (:logits (llm-action-logits llm-gf pid act-id-lists)))]
                            (let [[r c] (s->rc s)]
                              (println (str "  cell (row " r ",col " c "): π=" (mapv #(.toFixed % 2) d))))
                            d))
                        demo-states demo-pids)]
        (check-true "the LLM policy distribution differs across at least two cells"
                    (some (fn [[a b]] (> (reduce + (map (comp abs -) a b)) 1e-3))
                          (for [i (range (count dists)) j (range (count dists)) :when (< i j)]
                            [(nth dists i) (nth dists j)]))))

      ;; --- a short LLM-policy rollout (illustrative; planning quality NOT asserted) ---
      (println "\n-- LLM-policy rollout (3 steps; illustrative, not asserted) --")
      (pr/let [traj (rollout llm-gf tok act-id-lists start 3)]
        (doseq [{:keys [state action probs]} traj]
          (let [[r c] (s->rc state)]
            (println (str "  at (row " r ",col " c ") → action "
                          (name (nth [:left :right :up :down] action))
                          "  π=" (mapv #(.toFixed % 2) probs)))))
        (check-true "rollout produced at least one step with a valid action"
                    (and (seq traj) (every? #(contains? #{0 1 2 3} (:action %)) traj)))

        (println (str "\n" (if (zero? @fails)
                             "ALL CHECKS PASSED ✓"
                             (str @fails " CHECK(S) FAILED ✗"))))
        (when (pos? @fails) (js/process.exit 1))))))

(defn run-or-skip
  "Load the model and run the self-checking flagship, or skip cleanly if the model
   is absent. Returns a promise. Used both by direct invocation and the test wrapper
   (which awaits it as its top-level promise)."
  []
  (if-not (.existsSync fs model-path)
    (do (println (str "\nSKIP: model not found at " model-path
                      " — agents×llm flagship needs qwen3-0.6b-mlx-bf16. (clean skip, exit 0)"))
        (pr/resolved :skipped))
    (-> (pr/let [m (llm/load-model model-path)] (run m))
        (pr/catch (fn [e]
                    (println (str "\n✗ FAIL: agents×llm flagship errored: " (.-message e)))
                    (js/process.exit 1))))))

;; Auto-run only when this file is the entry point (so `(:require ...)` from the
;; test wrapper does not double-run; the wrapper calls run-or-skip explicitly and
;; awaits it — top-level requires would not await this async self-check).
(def ^:private main?
  (boolean (some #(re-find #"agents_llm_policy\.cljs" (str %)) (array-seq js/process.argv))))

(when main? (run-or-skip))
