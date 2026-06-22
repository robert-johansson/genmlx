;; @tier slow
(ns genmlx.llm-proposer-test
  "Acceptance for genmlx.world.llm-proposer — the REAL-LLM proposer for the REPL loop
   (genmlx-0yv7 / genmlx-wpua). NATIVE-FREE and deterministic: the policy LLM is MOCKED
   (an injected :call-llm), so every assertion is reproducible and no model loads. The
   live big/small-model experiment lives in scripts/synth_llm_probe.cljs; this file pins
   the pure, unit-testable parts AND the end-to-end wiring (a scripted 'LLM' drives the
   real genmlx.world.synth driver).

   Run: bun run --bun nbb test/genmlx/llm_proposer_test.cljs

   PARTS:
     A — extract-form: pulls the (fn [trace] ...) form out of fenced / prosey / <think>'d
         completions, recovers a bare (let ...) body, returns nil on no-form, and KEEPS
         a DSL slip (mx/0) verbatim (the check node, not the extractor, judges it).
     B — parse-spec: round-trips render <-> parse for in-grammar models (arg FORMS
         preserved), and returns nil for off-grammar code (a dist bound without trace,
         a non-static loop).
     C — completions->candidates: dedups, attaches :code (raw) + best-effort :spec',
         labels in-grammar :llm vs off-grammar :llm-raw.
     D — prompt builders: the step prompt is FEEDBACK-CONDITIONED (the verifier verdict
         is in the text); the system prompt teaches the anti-cliff rules.
     E — make-proposer (mock): returns candidates AND passes the feedback into the prompt.
     F — INTEGRATION: a scripted feedback-sensitive 'LLM' drives the real synth driver —
         real garbage reaches the check node and is REJECTED, the clean improvement is
         ACCEPTED, the loop climbs exact evidence and self-terminates on plateau."
  (:require [genmlx.world.llm-proposer :as lp]
            [genmlx.world.synth :as s]
            [genmlx.codegen.eval :as ce]
            [clojure.string :as str]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println "  PASS" label))
      (do (swap! fail inc) (println "  FAIL" label))))
(defn assert-close [label expected actual tol]
  (let [ok (and (number? actual) (js/isFinite actual) (<= (js/Math.abs (- expected actual)) tol))]
    (if ok (do (swap! pass inc) (println "  PASS" label (str "(" actual " ~ " expected ")")))
        (do (swap! fail inc) (println "  FAIL" label (str "(" actual " vs " expected ")"))))))

;; ===========================================================================
(println "\n-- A. extract-form --")

(def good-form
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 5))] {:y0 (trace :y0 (dist/gaussian mu 1.0))}))")

(assert-true "bare form returned verbatim"
             (= good-form (lp/extract-form good-form)))
(assert-true "fenced ```clojure block extracted"
             (= good-form (lp/extract-form (str "Here:\n```clojure\n" good-form "\n```"))))
(assert-true "prose + bare form: the form is found"
             (= good-form (lp/extract-form (str "Sure, here is the model: " good-form " — done."))))
(assert-true "<think> block stripped before extraction"
             (= good-form (lp/extract-form (str "<think>let me reason (a b c</think>\n" good-form))))
(assert-true "no model form -> nil"
             (nil? (lp/extract-form "I cannot help with that. (just prose, no fn)")))
(assert-true "bare (let ...) body recovered by wrapping in (fn [trace] ..)"
             (let [r (lp/extract-form "(let [mu (trace :mu (dist/gaussian 0 5))] {:y0 (trace :y0 (dist/gaussian mu 1.0))})")]
               (and r (str/starts-with? r "(fn [trace]") (ce/valid-cljs? r))))
;; the slip is KEPT — the extractor never sanitizes; the check node judges it. (mx/zero
;; is a readable symbol that is UNBOUND at eval — a real cliff slip; mx/0 is not even
;; reader-valid, so it is caught one level earlier, at the :parses? gate.)
(assert-true "DSL slip (mx/zero unbound) preserved verbatim in the extracted form"
             (let [slip "(fn [trace] (let [mu (trace :mu (dist/gaussian mx/zero 1))] {:y0 (trace :y0 (dist/gaussian mu 1.0))}))"]
               (= slip (lp/extract-form (str "```clojure\n" slip "\n```")))))
(assert-true "parens inside a string literal don't break balancing"
             (let [c "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 5))] {:y0 (trace :y0 (dist/gaussian mu 1.0))}))  ;; note: \"a ( b\""]
               (some? (lp/extract-form c))))

;; ===========================================================================
(println "\n-- B. parse-spec (inverse of render) --")

(def spec1
  (s/spec [(s/latent 'slope "gaussian" [0 3])
           (s/latent 'intercept "gaussian" [0 5])]
          [(s/obs :y0 "gaussian" [(list 'mx/add (list 'mx/multiply 'slope (list 'mx/scalar 0.0)) 'intercept) 1.0])
           (s/obs :y1 "gaussian" [(list 'mx/add (list 'mx/multiply 'slope (list 'mx/scalar 1.0)) 'intercept) 1.0])]))

(assert-true "render -> parse-spec round-trips to an equal spec (arg FORMS preserved)"
             (= spec1 (lp/parse-spec (s/render spec1))))
(assert-true "round-trip re-renders identically"
             (= (s/render spec1) (s/render (lp/parse-spec (s/render spec1)))))
(assert-true "no-latent body (bare obs map) parses"
             (let [sp (lp/parse-spec "(fn [trace] {:y0 (trace :y0 (dist/gaussian 0 1))})")]
               (and sp (empty? (:latents sp)) (= 1 (count (:obs sp))))))
(assert-true "off-grammar: a dist bound WITHOUT trace -> nil (raw :code still checkable)"
             (nil? (lp/parse-spec "(fn [trace] (let [mu (dist/gaussian 0 5)] {:y0 (trace :y0 (dist/gaussian mu 1))}))")))
(assert-true "off-grammar: a non-static loop body -> nil"
             (nil? (lp/parse-spec "(fn [trace] (doseq [i (range 3)] (trace :y (dist/gaussian 0 1))))")))
(assert-true "an arg-form slip ((:p trace)) round-trips structurally (caught later at eval)"
             (let [sp (lp/parse-spec "(fn [trace] (let [p (trace :p (dist/beta-dist 1 1))] {:y0 (trace :y0 (dist/bernoulli (:p trace)))}))")]
               (and sp (= '(:p trace) (first (:args (first (:obs sp))))))))
;; the shape strong instruct models actually emit: EVERYTHING in the let, return a map of
;; symbol references. parse-spec classifies referenced sites as observations.
(assert-true "'everything in the let, map-of-refs' shape (the 35B's shape) parses correctly"
             (let [sp (lp/parse-spec
                       (str "(fn [trace] (let [slope (trace :slope (dist/gaussian 1 2)) "
                            "intercept (trace :intercept (dist/gaussian 0 2)) "
                            "y0 (trace :y0 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar 0.0)) intercept) 1.0)) "
                            "y1 (trace :y1 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar 1.0)) intercept) 1.0))] "
                            "{:y0 y0 :y1 y1}))"))]
               (and sp (= 2 (count (:latents sp))) (= 2 (count (:obs sp)))
                    (= #{'slope 'intercept} (set (map :sym (:latents sp))))
                    (= #{:y0 :y1} (set (map :addr (:obs sp)))))))
;; regression (review finding): a latent that is BOTH returned in the map AND referenced in
;; another site's args must STAY let-bound, or the re-render has an unbound symbol.
(assert-true "a returned latent also referenced in another site's args stays let-bound (no unbound re-render)"
             (let [code "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 5)) y0 (trace :y0 (dist/gaussian mu 1.0))] {:mu mu :y0 y0}))"
                   sp (lp/parse-spec code)
                   fb (s/check (s/render sp) {:y0 1.0} {})]
               (and sp
                    (= #{'mu} (set (map :sym (:latents sp))))
                    (some #(= :y0 (:addr %)) (:obs sp))
                    (:evals? fb))))

;; ===========================================================================
(println "\n-- C. completions->candidates --")

(let [comps [(str "```clojure\n" good-form "\n```")
             (str "```clojure\n" good-form "\n```")               ; duplicate -> deduped
             "(fn [trace] (doseq [i (range 3)] (trace :y (dist/gaussian 0 1))))" ; off-grammar
             "sorry, no code here"]                                ; no form -> dropped
      cands (lp/completions->candidates comps :CURRENT-SPEC)]
  (assert-true "dedup + drop-no-form -> 2 candidates" (= 2 (count cands)))
  (assert-true "every candidate carries raw :code" (every? :code cands))
  (assert-true "in-grammar candidate labeled :llm with a parsed :spec'"
               (some #(and (= :llm (:edit %)) (map? (:spec' %)) (not= :CURRENT-SPEC (:spec' %))) cands))
  (assert-true "off-grammar candidate labeled :llm-raw, :spec' falls back to current"
               (some #(and (= :llm-raw (:edit %)) (= :CURRENT-SPEC (:spec' %))) cands)))

;; ===========================================================================
(println "\n-- D. prompt builders (feedback-conditioning) --")

(let [obs {:y0 5.0 :y1 5.2}
      p-err (lp/step-prompt "fit y" obs "(fn [trace] ...)"
                            {:parses? true :schema-ok? true :covered? true :evals? false
                             :error "Unbound: mx/0" :evidence nil})
      p-ev  (lp/step-prompt "fit y" obs "(fn [trace] ...)"
                            {:parses? true :schema-ok? true :covered? true :evals? true
                             :error nil :evidence -7.25})]
  (assert-true "step prompt embeds the data" (str/includes? p-err ":y0 = 5"))
  (assert-true "step prompt embeds the current model" (str/includes? p-err "CURRENT MODEL"))
  (assert-true "ERROR feedback is conditioned on (the eval error is in the prompt)"
               (str/includes? p-err "Unbound: mx/0"))
  (assert-true "EVIDENCE feedback is conditioned on (the number to beat is in the prompt)"
               (str/includes? p-ev "-7.25"))
  (assert-true "system prompt teaches the anti-cliff rules (no mx/0, trace not a map)"
               (and (str/includes? lp/default-system "mx/0")
                    (str/includes? lp/default-system "(:slope trace)"))))

;; ===========================================================================
(println "\n-- E. make-proposer with a mock LLM --")

(let [seen (atom nil)
      mock (fn [req] (reset! seen req)
             {:completions [(str "```clojure\n" good-form "\n```")]})
      prop (lp/make-proposer {:call-llm mock :task-desc "fit y" :observations {:y0 5.0}
                              :k 1 :temperature 0.5})
      fb   {:code "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 9))] {:y0 (trace :y0 (dist/gaussian mu 9))}))"
            :parses? true :schema-ok? true :covered? true :evals? true :evidence -12.0}
      cands (prop {:dummy :spec} fb)]
  (assert-true "proposer returns >=1 candidate from the LLM" (pos? (count cands)))
  (assert-true "candidate carries the LLM's raw :code" (= good-form (:code (first cands))))
  (assert-true "proposer fed the verifier feedback (evidence -12.0) into the prompt"
               (str/includes? (:prompt @seen) "-12.0"))
  (assert-true "proposer rendered the CURRENT model (from :code feedback) into the prompt"
               (str/includes? (:prompt @seen) "dist/gaussian mu 9")))

;; one-shot control: no feedback, just task+data
(let [mock (fn [_] {:completions [(str "```clojure\n" good-form "\n```")
                                  (str "```clojure\n" good-form "\n```")]})
      cands (lp/one-shot-candidates {:call-llm mock :task-desc "fit y" :observations {:y0 5.0} :k 2})]
  (assert-true "one-shot-candidates dedups to 1 :code-carrying candidate"
               (and (= 1 (count cands)) (:code (first cands)))))

;; ===========================================================================
(println "\n-- F. INTEGRATION: a scripted 'LLM' drives the real synth driver --")

;; Tightly-clustered data near 5 -> a tighter obs noise has HIGHER evidence. Confirm the
;; direction with an INDEPENDENT closed-form Gaussian-Gaussian marginal (non-circular).
(def f-obs {:y0 5.0 :y1 5.2 :y2 4.8})
(defn gg-marginal [ys m0 s0 sn]
  (let [n (count ys) s0² (* s0 s0) sn² (* sn sn)
        ds (map #(- % m0) ys) sd (reduce + ds) sd² (reduce + (map #(* % %) ds))
        denom (+ sn² (* n s0²))
        logdet (+ (js/Math.log denom) (* (dec n) (js/Math.log sn²)))
        quad (* (/ 1.0 sn²) (- sd² (* (/ s0² denom) (* sd sd))))]
    (- (* -0.5 n (js/Math.log (* 2 js/Math.PI))) (* 0.5 logdet) (* 0.5 quad))))
(assert-true "direction check: tighter noise (1.0) out-scores loose (3.0) for tight data"
             (> (gg-marginal (vals f-obs) 5 3 1.0) (gg-marginal (vals f-obs) 5 3 3.0)))

;; crude (loose) init: mu ~ N(5,3), each :yj ~ N(mu, 3.0)
(def f-init (s/spec [(s/latent 'mu "gaussian" [5 3])]
                    (for [k [:y0 :y1 :y2]] (s/obs k "gaussian" ['mu 3.0]))))

;; scripted 'LLM': each propose returns TWO completions —
;;  (1) a real DSL slip (a dist bound WITHOUT trace — readable, but evals to garbage) the
;;      check node must REJECT, and
;;  (2) a clean improvement (the same model with obs noise tightened to 1.0).
(def f-tight
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 5 3))] {:y0 (trace :y0 (dist/gaussian mu 1.0)) :y1 (trace :y1 (dist/gaussian mu 1.0)) :y2 (trace :y2 (dist/gaussian mu 1.0))}))")
(def f-garbage
  "(fn [trace] (let [mu (dist/gaussian 0 5)] {:y0 (trace :y0 (dist/gaussian mu 1.0)) :y1 (trace :y1 (dist/gaussian mu 1.0)) :y2 (trace :y2 (dist/gaussian mu 1.0))}))")
(defn scripted-llm [_req]
  {:completions [(str "```clojure\n" f-garbage "\n```")
                 (str "```clojure\n" f-tight "\n```")]})

(let [prop (lp/make-proposer {:call-llm scripted-llm :task-desc "fit y" :observations f-obs :k 2})
      res  (s/synthesize {:init-spec f-init :observations f-obs :propose prop
                          :max-steps 4 :plateau-eps 0.05})
      traj (:trajectory res)
      start (:evidence (first traj))
      final (:evidence (last traj))]
  (assert-true "loop CLIMBS exact evidence (loose -> tight) via the LLM proposer"
               (> final (+ start 0.05)))
  (assert-true "loop self-terminates on plateau (not :stuck/:max-steps)"
               (= :plateau (:stop-reason res)))
  (assert-true "accepted edit came from the LLM (:llm), garbage rejected"
               (= :llm (:edit (last traj))))
  ;; the accepted trajectory row carries the LLM's RAW code (the top-level :code is
  ;; re-rendered from the spec, where CLJS prints 1.0 as 1 — so assert on the raw row).
  (assert-true "accepted (raw) model is the tightened one (obs noise 1.0)"
               (str/includes? (:code (last traj)) "mu 1.0"))
  (assert-true "the loose noise (3.0) is gone from the re-rendered final model"
               (not (str/includes? (:code res) "mu 3")))
  ;; the rejected garbage really WAS scored against the check node and failed (raison d'être)
  (let [code (lp/extract-form (first (:completions (scripted-llm nil))))
        fb   (s/check code f-obs {})]
    (assert-true "the real LLM garbage (dist bound w/o trace) reached check and was NOT scored"
                 (and code (not (s/scored? fb))))))

;; ===========================================================================
(println "\n-- G. proposer self-correction (the inner REPL revision loop) --")

(def g-obs {:y0 5.0 :y1 5.2 :y2 4.8})
;; coverage slip: the model copies the EXAMPLE's :o addresses instead of the observed :y
(def g-bad
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 5 3))] {:o0 (trace :o0 (dist/gaussian mu 1.0)) :o1 (trace :o1 (dist/gaussian mu 1.0)) :o2 (trace :o2 (dist/gaussian mu 1.0))}))")
(def g-good
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 5 3))] {:y0 (trace :y0 (dist/gaussian mu 1.0)) :y1 (trace :y1 (dist/gaussian mu 1.0)) :y2 (trace :y2 (dist/gaussian mu 1.0))}))")
;; a scripted 'LLM' that slips until it is told the coverage error, then fixes it
(defn g-llm [req]
  {:completions [(str "```clojure\n" (if (str/includes? (:prompt req) "Coverage problem") g-good g-bad) "\n```")]})

(let [prop  (lp/make-proposer {:call-llm g-llm :task-desc "fit y" :observations g-obs :k 1 :revise 2})
      cands (prop (s/spec [(s/latent 'mu "gaussian" [5 3])] [(s/obs :y0 "gaussian" ['mu 3.0])])
                  {:parses? true :schema-ok? true :covered? true :evals? true :evidence -20.0})
      scored (filter #(s/scored? (s/check (:code %) g-obs {})) cands)]
  (assert-true "with :revise=0 (default) the coverage slip is NOT corrected (control)"
               (let [p0 (lp/make-proposer {:call-llm g-llm :task-desc "fit y" :observations g-obs :k 1})
                     c0 (p0 (s/spec [(s/latent 'mu "gaussian" [5 3])] [(s/obs :y0 "gaussian" ['mu 3.0])])
                            {:parses? true :schema-ok? true :covered? true :evals? true :evidence -20.0})]
                 (empty? (filter #(s/scored? (s/check (:code %) g-obs {})) c0))))
  (assert-true "revision turns the coverage-slipped proposer into a SCORING candidate"
               (seq scored))
  (assert-true "the scoring candidate is the corrected (:y-address) model"
               (some #(str/includes? (:code %) ":y0") scored))
  (assert-true "the corrected candidate is flagged :revised?"
               (some :revised? cands)))

;; ===========================================================================
(println (str "\n==== llm_proposer_test: " @pass " passed, " @fail " failed ===="))
(when (pos? @fail) (js/process.exit 1))
