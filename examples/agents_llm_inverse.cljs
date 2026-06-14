(ns agents-llm-inverse
  "FLAGSHIP — agents × llm (ROADMAP Phase 3, item 4, Direction 2: LLM reasons OVER
   an agent model). The companion to agents_llm_policy.cljs (Direction 1, LLM as the
   policy). Here the nesting is the other way around: an AGENT GENERATIVE FUNCTION
   provides the likelihood and an LLM GENERATIVE FUNCTION provides a reasoning score,
   and the two are combined in a single Bayesian inference over the agent's hidden goal.

   The task is inverse planning: we watch an agent take one action from the start cell
   and infer which of several candidate destinations it is pursuing. Two evidence
   sources, both generative functions, both scored through the GFI:
     • AGENT MODEL likelihood — for each candidate goal g, build the MDP-optimal
       softmax agent for g and read P(observed action | start, g) from its policy.
       This is classic inverse planning (planning-as-inference).
     • LLM REASONER score — for each goal g, ask the LLM 'the agent at (r,c) moved
       <action>; is it heading to (gr,gc)? Answer yes/no:' and read P_LLM(yes | g)
       from the LLM GF's per-token log-probs (p/assess on ' yes' vs ' no').
   The joint posterior P(g | action) ∝ P(action | g) · P_LLM(yes | g) fuses the
   agent model and the LLM reasoner — 'an LLM reasoning over an agent model.'

   SAFETY/DESIGN: the inference is host-enumerated over a small fixed goal set with
   ONE sequential LLM assess per goal — there is NO inference engine running over the
   LLM GF, so the LLM KV cache is never shared across particles (the failure mode the
   scoping flagged). Each LLM assess is independent, exactly like Direction 1.

   HONESTY: the AGENT-model half is exact and asserted — observing g*'s own optimal
   action makes g* the agent-model MAP (the candidate goals are chosen to induce
   distinct optimal first actions, so the action is identifying). The LLM-reasoner
   half is a 0.6B base model: its P_LLM(yes|g) is printed and checked only for being
   a valid probability — NEVER asserted to reason correctly. Skip-clean if the model
   is absent.

   Reuse, zero engine change: genmlx.agents.gridworld/build-mdp + agent/make-mdp-agent
   (the agent GF + soft policy), genmlx.llm.core/make-llm-gf (the reasoner GF), the GFI.
   Run:  bun run --bun nbb examples/agents_llm_inverse.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
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
  (let [a (->num actual) ok (<= (abs (- expected a)) tol)]
    (println (str (if ok "  ✓ " "  ✗ FAIL ") label " — expected " expected ", got " (.toFixed a 6)))
    (when-not ok (swap! fails inc)) ok))

(defn- check-true [label ok]
  (println (str (if ok "  ✓ " "  ✗ FAIL ") label))
  (when-not ok (swap! fails inc)) ok)

;; ---------------------------------------------------------------------------
;; math helpers
;; ---------------------------------------------------------------------------

(defn- softmax [xs]
  (let [m (apply max xs) es (mapv #(js/Math.exp (- % m)) xs) z (reduce + es)]
    (mapv #(/ % z) es)))

(defn- normalize [xs] (let [z (reduce + xs)] (mapv #(/ % z) xs)))
(defn- argmax-idx [xs] (first (apply max-key second (map-indexed vector xs))))

;; ---------------------------------------------------------------------------
;; the agent model: per-goal gridworld MDP + soft policy (inverse planning)
;; ---------------------------------------------------------------------------

(def N 3)            ; 3×3 grid
(def ALPHA 3.0)      ; softmax rationality of the modelled agent
(def action-names [:left :right :up :down])

(defn- grid-with-goal [[gr gc]]
  (vec (for [r (range N)] (vec (for [c (range N)] (if (and (= r gr) (= c gc)) :goal :empty))))))

(defn- goal-mdp [goal-rc]
  (gw/build-mdp {:grid (grid-with-goal goal-rc)
                 :utilities {:goal 1.0 :timeCost -0.05}
                 :start [0 0] :gamma 1.0 :noise 0.0}))

(defn- goal-agent [goal-rc]
  (agent/make-mdp-agent {:mdp (goal-mdp goal-rc) :alpha ALPHA :gamma 1.0 :n-iters 24}))

(defn- q-row [ag s] (nth (vec (map vec (mx/->clj (:Q ag)))) s))

(defn- action-probs
  "Soft policy P(action | s, goal) = softmax(ALPHA · Q_goal[s])."
  [ag s]
  (softmax (mapv #(* ALPHA %) (q-row ag s))))

(defn- s->rc [mdp s] (let [w (:W mdp)] [(quot s w) (rem s w)]))

;; ---------------------------------------------------------------------------
;; the LLM reasoner: P_LLM(yes | observation, goal) through the GFI
;; ---------------------------------------------------------------------------

(defn- token-logprob
  "log p(word tokens | prompt) via the LLM GF's GFI assess op. `word-ids` is the
   pre-encoded token-id vector (encode is async, awaited by the caller)."
  [llm-gf prompt-ids word-ids]
  (let [n (count word-ids)
        constraints (reduce (fn [cm i] (cm/set-value cm (keyword (str "t" i))
                                                     (mx/scalar (nth word-ids i) mx/int32)))
                            cm/EMPTY (range n))]
    (->num (:weight (p/assess llm-gf [prompt-ids n] constraints)))))

(defn- reasoner-prompt [mdp s a goal-rc]
  (let [[sr sc] (s->rc mdp s) [gr gc] goal-rc]
    (str "An agent on a 3x3 grid (rows 0-2 top to bottom, cols 0-2 left to right) is at "
         "row " sr ", column " sc ". It just moved " (name (nth action-names a)) ". "
         "Is the agent heading toward the destination at row " gr ", column " gc "? "
         "Answer yes or no:")))

;; ---------------------------------------------------------------------------
;; run + self-check (async: model load + tokenizer encode are promises)
;; ---------------------------------------------------------------------------

(def model-path (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-mlx-bf16"))

(defn run [m]
  (pr/let [tok    (:tokenizer m)
           llm-gf (llm-core/make-llm-gf m)
           yes-ids (pr/let [v (llm/encode tok " yes")] (vec v))
           no-ids  (pr/let [v (llm/encode tok " no")] (vec v))]
    ;; choose candidate goals that induce DISTINCT optimal first actions (so the
    ;; observed action identifies the goal); g* is the first.
    (let [ref-mdp     (goal-mdp [2 2])
          start       (:start-idx ref-mdp)
          candidates  (for [r (range N) c (range N) :when (not= [r c] [0 0])] [r c])
          first-act   (fn [g] (argmax-idx (q-row (goal-agent g) start)))
          goals       (->> candidates (group-by first-act) vals (mapv first) (take 3) vec)
          agents      (mapv goal-agent goals)
          g*          (first goals)
          a*          (first-act g*)                ; observed action = g*'s optimal first move
          ;; AGENT MODEL: P(a* | start, g) for each candidate goal
          lik         (mapv (fn [ag] (nth (action-probs ag start) a*)) agents)
          agent-post  (normalize lik)]
      (println "\n=== agents × llm flagship — an LLM reasoning OVER an agent model ===")
      (println (str "  candidate goals (row,col): " goals
                    "   each induces optimal first action "
                    (mapv #(name (nth action-names (first-act %))) goals)))
      (println (str "  OBSERVED: agent at " (s->rc ref-mdp start) " moved "
                    (name (nth action-names a*)) " (the true goal g* = " g* ")"))

      ;; --- agent-model inverse planning (exact, asserted) ---
      (println "\n-- agent-model likelihood P(action | goal) and posterior --")
      (println (str "  P(a*|g):      " (mapv #(.toFixed % 3) lik)))
      (println (str "  agent post.:  " (mapv #(.toFixed % 3) agent-post)
                    "   MAP = goal " (nth goals (argmax-idx agent-post))))
      (check-true "agent-model likelihoods all finite" (every? js/isFinite lik))
      (check-close "agent-model posterior sums to 1" 1.0 (reduce + agent-post) 1e-6)
      (check-true "agent-model MAP is the true goal g* (the observed action identifies it)"
                  (= 0 (argmax-idx agent-post)))

      ;; --- LLM reasoner over the agent model (printed; validity asserted) ---
      (pr/let [prompt-ids (pr/all (mapv (fn [g] (pr/let [v (llm/encode tok (reasoner-prompt ref-mdp start a* g))] (vec v)))
                                        goals))]
        (let [p-yes (mapv (fn [pid]
                            (let [ly (token-logprob llm-gf pid yes-ids)
                                  ln (token-logprob llm-gf pid no-ids)
                                  mx* (max ly ln)]
                              (/ (js/Math.exp (- ly mx*))
                                 (+ (js/Math.exp (- ly mx*)) (js/Math.exp (- ln mx*))))))
                          prompt-ids)
              llm-post  (normalize p-yes)
              combined  (normalize (mapv * lik p-yes))]
          (println "\n-- LLM reasoner P_LLM(yes | observation, goal) (0.6B model; validity-only) --")
          (println (str "  P_LLM(yes|g): " (mapv #(.toFixed % 3) p-yes)))
          (println (str "  LLM post.:    " (mapv #(.toFixed % 3) llm-post)))
          (check-true "LLM P_LLM(yes|g) are all valid probabilities in [0,1]"
                      (every? #(<= 0.0 % 1.0) p-yes))
          (check-true "LLM P_LLM(yes|g) are finite" (every? js/isFinite p-yes))
          (check-close "LLM-only posterior sums to 1" 1.0 (reduce + llm-post) 1e-6)

          ;; --- the joint: agent model × LLM reasoner ---
          (println "\n-- joint posterior P(goal | action) ∝ P(action|goal) · P_LLM(yes|goal) --")
          (println (str "  combined:     " (mapv #(.toFixed % 3) combined)
                        "   MAP = goal " (nth goals (argmax-idx combined))))
          (check-close "combined posterior sums to 1" 1.0 (reduce + combined) 1e-6)
          (check-true "combined posterior is finite and non-negative"
                      (every? #(and (js/isFinite %) (>= % 0.0)) combined))

          (println (str "\n" (if (zero? @fails) "ALL CHECKS PASSED ✓"
                                 (str @fails " CHECK(S) FAILED ✗"))))
          (when (pos? @fails) (js/process.exit 1)))))))

(defn run-or-skip
  "Load the model and run the self-checking inverse-planning flagship, or skip
   cleanly if the model is absent. Returns a promise (the test wrapper awaits it)."
  []
  (if-not (.existsSync fs model-path)
    (do (println (str "\nSKIP: model not found at " model-path
                      " — agents×llm inverse flagship needs qwen3-0.6b-mlx-bf16. (clean skip, exit 0)"))
        (pr/resolved :skipped))
    (-> (pr/let [m (llm/load-model model-path)] (run m))
        (pr/catch (fn [e]
                    (println (str "\n✗ FAIL: agents×llm inverse flagship errored: " (.-message e)))
                    (js/process.exit 1))))))

(def ^:private main?
  (boolean (some #(re-find #"agents_llm_inverse\.cljs" (str %)) (array-seq js/process.argv))))

(when main? (run-or-skip))
