;; @tier slow
(ns genmlx.synth-test
  "Acceptance for genmlx.world.synth — the Phase-1 REPL-synthesis KERNEL (genmlx-n74t):
   the model-spec + form-level edit ops, the four-level self-check, and the greedy
   oracle-driven driver. Deterministic, NATIVE-FREE (no policy LLM): conjugate partial
   models score the EXACT analytical marginal, so every assertion is reproducible. The
   end-to-end loop-solves-an-advanced-model demonstration (experiment B) lives in
   scripts/synth_repl_probe.cljs; this file pins the pure, unit-testable parts.

   Run: bun run --bun nbb test/genmlx/synth_test.cljs

   PARTS:
     A — spec + render + edit ops: each move is a pure spec->spec transform whose
         render is valid ClojureScript with the expected structure.
     B — the check node: the four levels gate correctly (parse / schema / coverage /
         eval / fit), with the right :error, and a valid model's :evidence matches an
         INDEPENDENT closed-form Gaussian-Gaussian marginal (non-circular).
     C — the greedy driver: climbs exact evidence, rejects distractor + broken
         candidates (backtrack), self-terminates on plateau, and lands at the
         grid-optimal model the closed form predicts."
  (:require [genmlx.world.synth :as s]
            [genmlx.codegen.eval :as ce]
            [clojure.string :as str]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(defn assert-close [label expected actual tol]
  (let [ok (and (number? actual) (js/isFinite actual) (<= (js/Math.abs (- expected actual)) tol))]
    (if ok
      (do (swap! pass inc) (println "  PASS" label (str "(" actual " ~ " expected ")")))
      (do (swap! fail inc) (println "  FAIL" label (str "(" actual " vs " expected " tol " tol ")"))))))

;; Independent closed-form Gaussian-Gaussian marginal (DERIVED outside GenMLX —
;; mu ~ N(m0,s0), y_i ~ N(mu, sn) — so the evidence check is not circular).
(defn gaussian-gaussian-marginal [ys m0 s0 sn]
  (let [n (count ys) s0² (* s0 s0) sn² (* sn sn)
        ds (map #(- % m0) ys) sd (reduce + ds) sd² (reduce + (map #(* % %) ds))
        denom (+ sn² (* n s0²))
        logdet (+ (js/Math.log denom) (* (dec n) (js/Math.log sn²)))
        quad (* (/ 1.0 sn²) (- sd² (* (/ s0² denom) (* sd sd))))]
    (- (* -0.5 n (js/Math.log (* 2 js/Math.PI))) (* 0.5 logdet) (* 0.5 quad))))

(def ^:private obs4 {:y0 2.0 :y1 2.3 :y2 1.7 :y3 2.1})
(def ^:private obs-ys (mapv obs4 [:y0 :y1 :y2 :y3]))

;; A shared-mean spec helper: mu ~ N(m0, s0), each :yj ~ N(mu, noise).
(defn shared-mean-spec [m0 s0 noise]
  (s/spec [(s/latent 'mu "gaussian" [m0 s0])]
          (for [k [:y0 :y1 :y2 :y3]]
            (s/obs k "gaussian" ['mu noise]))))

;; ===========================================================================
;; PART A — spec + render + edit operations
;; ===========================================================================

(defn part-a []
  (println "\n== PART A: spec + render + edit ops ==")
  (let [sp (shared-mean-spec 2 1 0.5)
        code (s/render sp)]
    (assert-true "render produces a complete cljs form" (ce/valid-cljs? code))
    (assert-true "render reads as (fn [trace] ...)" (str/starts-with? code "(fn [trace] (let ["))
    (assert-true "render emits the latent let-binding"
                 (str/includes? code "mu (trace :mu (dist/gaussian 2 1))"))
    (assert-true "render emits an obs trace site in the returned map"
                 (str/includes? code ":y0 (trace :y0 (dist/gaussian mu 0.5))")))

  ;; add-latent / add-obs
  (let [sp (-> (s/spec [] [])
               (s/add-latent (s/latent 'slope "gaussian" [0 3]))
               (s/add-obs (s/obs :y0 "gaussian" ['slope 1])))]
    (assert-true "add-latent appends a latent" (= 1 (count (:latents sp))))
    (assert-true "add-obs appends an obs site" (= 1 (count (:obs sp))))
    (assert-true "add-latent + add-obs render to valid cljs" (ce/valid-cljs? (s/render sp)))
    (assert-true "add-obs REPLACES an existing addr (no duplicate)"
                 (= 1 (count (:obs (s/add-obs sp (s/obs :y0 "gaussian" ['slope 2])))))))

  ;; set-args (set-prior) / set-mean / set-noise
  (let [sp (shared-mean-spec 2 1 0.5)]
    (assert-true "set-args replaces a latent's prior"
                 (str/includes? (s/render (s/set-args sp :mu [0 10]))
                                "mu (trace :mu (dist/gaussian 0 10))"))
    (assert-true "set-mean re-points an obs at a latent-dependent mean"
                 (str/includes? (s/render (s/set-mean sp :y0 (list 'mx/multiply 'mu (list 'mx/scalar 2.0))))
                                ":y0 (trace :y0 (dist/gaussian (mx/multiply mu (mx/scalar 2)) 0.5))"))
    (assert-true "set-noise replaces an obs site's last arg"
                 (str/includes? (s/render (s/set-noise sp :y0 1.5))
                                ":y0 (trace :y0 (dist/gaussian mu 1.5))"))
    (assert-true "edit ops leave other sites untouched"
                 (str/includes? (s/render (s/set-noise sp :y0 1.5))
                                ":y1 (trace :y1 (dist/gaussian mu 0.5))")))

  ;; homogeneous-obs?: detects the Map-fold opportunity (same dist, same noise, same
  ;; single latent mean) without rewriting; false for heterogeneous / non-latent means.
  (let [homo     (s/spec [(s/latent 'mu "gaussian" [0 5])]
                         (for [k [:y0 :y1 :y2]] (s/obs k "gaussian" ['mu 1])))
        hetero   (s/spec [(s/latent 'a "gaussian" [0 5]) (s/latent 'b "gaussian" [0 5])]
                         [(s/obs :y0 "gaussian" ['a 1]) (s/obs :y1 "gaussian" ['b 1])])
        non-lat  (s/spec [(s/latent 'mu "gaussian" [0 5])]
                         [(s/obs :y0 "gaussian" [(list 'mx/scalar 0.0) 1])])]
    (assert-true "homogeneous-obs? true for same dist+noise+single latent mean"
                 (s/homogeneous-obs? homo))
    (assert-true "homogeneous-obs? false for distinct per-site latents"
                 (not (s/homogeneous-obs? hetero)))
    (assert-true "homogeneous-obs? false when the mean is not a latent symbol"
                 (not (s/homogeneous-obs? non-lat))))

  ;; set-noise is robust on a no-arg site (pure no-op, never throws).
  (let [sp (s/spec [] [(s/obs :y0 "gaussian" [])])]
    (assert-true "set-noise on an empty-args site is a no-op (no throw)"
                 (= sp (s/set-noise sp :y0 1.0)))))

;; ===========================================================================
;; PART B — the four-level self-check
;; ===========================================================================

(defn part-b []
  (println "\n== PART B: the check node (four levels) ==")

  ;; level 1: syntax
  (let [fb (s/check "(fn [trace] (let [" obs4)]
    (assert-true "unparseable -> :parses? false" (false? (:parses? fb)))
    (assert-true "unparseable -> not evaluated, has an :error"
                 (and (false? (:evals? fb)) (string? (:error fb)))))

  ;; level 2: semantics — body does not return a map
  (let [fb (s/check "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 1))] mu))" obs4)]
    (assert-true "non-map return -> :schema-ok? false" (false? (:schema-ok? fb)))
    (assert-true "non-map return -> :returns-map? false" (false? (:returns-map? fb))))

  ;; level 2: semantics — a delta DISTRIBUTION CONSTRUCTOR is the point-mass hack
  (let [fb (s/check (str "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 1))] "
                         "{:y0 (trace :y0 (dist/delta mu))}))") {:y0 2.0})]
    (assert-true "(dist/delta ...) -> :uses-delta? true" (true? (:uses-delta? fb)))
    (assert-true "(dist/delta ...) -> :schema-ok? false (rejected)" (false? (:schema-ok? fb))))

  ;; but an honest latent merely NAMED `delta` (a common offset name) is NOT rejected
  (let [fb (s/check (str "(fn [trace] (let [delta (trace :delta (dist/gaussian 0 1))] "
                         "{:y0 (trace :y0 (dist/gaussian delta 1))}))") {:y0 2.0})]
    (assert-true "a variable named `delta` is NOT a point-mass hack" (false? (:uses-delta? fb)))
    (assert-true "a variable named `delta` -> :schema-ok? true (accepted + scored)"
                 (and (:schema-ok? fb) (s/scored? fb))))

  ;; empty observations is a misconfiguration, not a perfectly-explained (evidence-0) model
  (let [fb (s/check (s/render (shared-mean-spec 2 1 0.5)) {})]
    (assert-true "empty observations -> not scored (no vacuous evidence 0)"
                 (and (not (s/scored? fb)) (nil? (:evidence fb))))
    (assert-true "empty observations -> carries an explanatory :error" (string? (:error fb))))

  ;; level 3: coverage — declares a latent but ignores the data addresses
  (let [fb (s/check "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 5))] {:result (trace :result (dist/gaussian mu 1))}))" obs4)]
    (assert-true "uncovered -> :schema-ok? true but :covered? false"
                 (and (:schema-ok? fb) (false? (:covered? fb))))
    (assert-true "uncovered -> not scored (no evidence)" (nil? (:evidence fb))))

  ;; level 4: behavior — covered, well-formed, but errors at eval (unknown dist)
  (let [fb (s/check (str "(fn [trace] {:y0 (trace :y0 (dist/student-t 0 1))})") {:y0 2.0})]
    (assert-true "eval error -> :covered? true but :evals? false"
                 (and (:covered? fb) (false? (:evals? fb))))
    (assert-true "eval error -> captures an :error message" (string? (:error fb))))

  ;; level 5: fit — a valid model scores the EXACT marginal (matches the closed form)
  (let [m0 2 s0 1 sn 0.5
        fb (s/check (s/render (shared-mean-spec m0 s0 sn)) obs4)
        truth (gaussian-gaussian-marginal obs-ys m0 s0 sn)]
    (assert-true "valid model -> parses/schema-ok/covered/evals all true"
                 (and (:parses? fb) (:schema-ok? fb) (:covered? fb) (:evals? fb)))
    (assert-true "valid conjugate model scored by an EXACT method" (= :exact (:method fb)))
    (assert-close "check evidence == independent closed-form marginal" truth (:evidence fb) 1e-2))

  ;; scored? predicate
  (assert-true "scored? true for a finite-evidence verdict"
               (s/scored? (s/check (s/render (shared-mean-spec 2 1 0.5)) obs4)))
  (assert-true "scored? false for an unparseable verdict"
               (not (s/scored? (s/check "(fn [trace] (let [" obs4)))))

;; ===========================================================================
;; PART C — the greedy driver climbs, backtracks, self-terminates
;; ===========================================================================

(defn part-c []
  (println "\n== PART C: greedy REPL driver ==")
  (let [grid       [0.3 0.5 0.7 1 1.5 2.5]
        init       (shared-mean-spec 2 1 2.5)   ; loose noise: headroom to climb
        ;; proposer = noise-grid refinements (good) + a clearly-worse distractor
        ;; (noise 9.0) + a BROKEN candidate (drops :y3 -> uncovered). The oracle must
        ;; pick only improving valid moves; the distractor/broken never get accepted.
        broken     (update (shared-mean-spec 2 1 0.5) :obs (comp vec butlast))
        propose    (fn [sp _fb]
                     (concat (s/noise-refinements sp grid)
                             [{:edit :distractor :desc "noise -> 9.0 (worse)" :spec' (s/set-noise sp :y0 9.0)}
                              {:edit :broken :desc "drop :y3 (uncovered)" :spec' broken}]))
        res        (s/synthesize {:init-spec init :observations obs4 :propose propose})
        traj       (:trajectory res)
        init-ev    (:evidence (first traj))
        final-ev   (:evidence (last traj))
        grid-best  (apply max (map #(gaussian-gaussian-marginal obs-ys 2 1 %) grid))]
    (assert-true "driver self-terminates on plateau" (= :plateau (:stop-reason res)))
    (assert-true "driver makes >=1 accepted edit" (>= (count traj) 2))
    (assert-true "final evidence strictly improves on the crude start" (> final-ev init-ev))
    (assert-true "every accepted step has a positive delta (monotone climb)"
                 (every? #(or (nil? (:delta %)) (pos? (:delta %))) traj))
    ;; the per-site noise search STRICTLY CONTAINS the homogeneous-noise grid (setting
    ;; every site to the same sigma is reachable), so the driver does at least as well
    ;; as the best shared-sigma model the closed form predicts — here strictly better.
    (assert-true "driver does at least as well as the best shared-sigma model (closed-form max)"
                 (>= final-ev (- grid-best 1e-2)))
    (assert-true "no distractor/broken edit was ever accepted"
                 (every? #(not (contains? #{:distractor :broken} (:edit %))) traj))
    (assert-true "final model is valid + covered + scored" (s/scored? (:feedback res)))

    ;; a never-scored init (uncovered) the proposer cannot complete -> :stuck, NOT a
    ;; spurious :plateau (which would falsely read as a fitted model).
    (let [stuck-init (s/spec [(s/latent 'mu "gaussian" [0 1])] [(s/obs :result "gaussian" ['mu 1])])
          stuck (s/synthesize {:init-spec stuck-init :observations obs4 :propose (fn [_ _] [])})]
      (assert-true "never-scored init + no candidate -> :stop-reason :stuck (not :plateau)"
                   (= :stuck (:stop-reason stuck))))

    (println (str "  trajectory: "
                  (str/join " -> " (map (fn [r] (str (name (:edit r)) "@"
                                                     (.toFixed (js/Number (:evidence r)) 2))) traj))))))

;; ===========================================================================

(defn- summary []
  (println (str "\n== synth (genmlx-n74t): " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(part-a)
(part-b)
(part-c)
(summary)
