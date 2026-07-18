;; Heuristic-oracle reward plugin for grpo_sessions (genmlx-mvyp).
;;
;; WHY: the built-in match-administered reward saturates on the 35B — the
;; child reproduces its own administered actions, every completion in a
;; group scores ~1.0, advantages are ~0, and GRPO gets no signal. This
;; plugin grades FORM quality instead of self-imitation: signals that
;; discriminate between regenerations of the same turn without needing
;; oracle answer keys. The maithri2 downstream oracle replaces it through
;; the SAME seam when it lands.
;;
;; CONTRACT (genmlx.world.session-reward/resolve-reward): the file's LAST
;; FORM is (fn [ctx] -> (fn [prompt completion] -> number)) with ctx =
;; {:points [...] :toolset [...] :opts {...}}. The returned closure is
;; pure, deterministic, and always FINITE (floor -1.0; a NaN/Inf reward
;; poisons the whole GRPO group). Exceptions floor + one stderr line
;; (the genmlx-oi07 observability rule).
;;
;; SIGNALS (weights sum to 1.0; every component in [0,1]):
;;   :parse    completion's tool blocks parse cleanly (errors -> floor)
;;   :declared fraction of calls whose function is in the deployed toolset
;;   :args     mean per-call fraction of parameters that are declared
;;   :code     fraction of embedded code candidates (fenced blocks + "("-
;;             leading parameter values) that PARSE as Clojure (edamame;
;;             parse only — no eval at reward time); no code = neutral 0.5
;;   :brevity  budget/len decay past :brevity-budget chars (default 1200)
;;
;; USE: REWARD=scripts/rewards/heuristic_oracle_reward.cljs (from repo root)
(ns heuristic-oracle-reward
  (:require [genmlx.llm.toolcall :as tc]
            [edamame.core :as e]
            [clojure.string :as str]))

(defn- clamp01 [x] (max 0.0 (min 1.0 x)))

(defn- err-msg [e] (or (ex-message e) (pr-str e)))

(defn- code-candidates
  "Fenced code blocks in the completion text plus any tool-call parameter
   value that looks like a Clojure form."
  [text calls]
  (concat (map second (re-seq #"(?s)```[a-zA-Z]*\n(.*?)```" text))
          (for [c calls
                [_ v] (:args c)
                :when (str/starts-with? (str/trim (str v)) "(")]
            (str v))))

(defn- code-parse-frac
  "Fraction of code candidates that parse. 0.5 (neutral) when there is no
   code to judge."
  [text calls]
  (let [cands (code-candidates text calls)]
    (if (empty? cands)
      0.5
      (/ (count (filter (fn [s]
                          (try (e/parse-string-all s {:all true}) true
                               (catch :default _ false)))
                        cands))
         (count cands)))))

(fn [{:keys [toolset opts]}]
  (let [{:keys [reward-floor weights brevity-budget]
         :or {reward-floor -1.0
              weights {:parse 0.15 :declared 0.25 :args 0.25
                       :code 0.20 :brevity 0.15}
              brevity-budget 1200}} opts
        names (set (map :name toolset))
        params-of (into {} (map (fn [t] [(:name t)
                                         (set (map :name (:params t)))])
                                toolset))]
    (fn [_prompt completion]
      (try
        (let [text (str completion)
              {:keys [calls errors]} (tc/parse-tool-calls text)]
          (if (seq errors)
            reward-floor
            (let [n (count calls)
                  ;; With an empty toolset we cannot judge declaredness or
                  ;; parameter schemas — credit both fully rather than
                  ;; penalize tool use we cannot assess.
                  no-toolset? (empty? names)
                  declared-frac
                  (cond
                    (zero? n) 0.0
                    no-toolset? 1.0
                    :else (/ (count (filter #(contains? names (:name %)) calls)) n))
                  args-frac
                  (cond
                    (zero? n) 0.0
                    no-toolset? 1.0
                    :else
                    (/ (reduce
                        + 0.0
                        (map (fn [{:keys [name args]}]
                               (let [ps (get params-of name)]
                                 (cond
                                   (nil? ps) 0.0          ; undeclared call
                                   (empty? args) 1.0      ; declared, arg-free
                                   :else (/ (count (filter ps (keys args)))
                                            (count args)))))
                             calls))
                       n))]
              (+ (* (:parse weights) 1.0)
                 (* (:declared weights) declared-frac)
                 (* (:args weights) args-frac)
                 (* (:code weights) (code-parse-frac text calls))
                 (* (:brevity weights)
                    (clamp01 (/ brevity-budget
                                (max (count text) brevity-budget))))))))
        (catch :default e
          (js/console.warn (str "[heuristic-oracle-reward] exception -> floor: "
                                (err-msg e)))
          reward-floor)))))
