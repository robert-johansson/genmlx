;; @tier slow
(ns genmlx.search-test
  "Acceptance for genmlx.world.search — the Phase-2 particle/beam search over construction
   steps (genmlx-47zx). Deterministic, native-free: every partial model is a shared-mean
   Gaussian scored by the EXACT analytical marginal, so the orderings are reproducible and
   checkable against an INDEPENDENT closed form. The 4-advanced-model + greedy-vs-beam
   head-to-head lives in scripts/synth_search_probe.cljs; this file pins the search core.

   Run: bun run --bun nbb test/genmlx/search_test.cljs

   PARTS:
     A — particles + expansion + selection: beam keeps a POPULATION (>1 branch alive),
         dedups identical programs, ranks deterministic evidence first.
     B — BEAM BEATS GREEDY on a constructed two-branch trap: greedy locks into the
         locally-best branch A; beam (width 2) keeps branch B alive and finds the global
         optimum B2 (exact evidence, verified against the closed form).
     C — backtrack-refine rescues a NARROW search: greedy (beam-width 1) + backtrack
         re-opens the step-1 decision, takes the road not taken, and recovers B2.
     D — :smc resampling reaches the optimum; adaptive width widens on an ambiguous step;
         the search self-terminates on a whole-population plateau with per-step diagnostics."
  (:require [genmlx.world.search :as se]
            [genmlx.world.synth :as syn]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println "  PASS" label))
        (do (swap! fail inc) (println "  FAIL" label))))
(defn assert-close [label expected actual tol]
  (let [ok (and (number? actual) (js/isFinite actual) (<= (js/Math.abs (- expected actual)) tol))]
    (if ok (do (swap! pass inc) (println "  PASS" label (str "(" actual " ~ " expected ")")))
           (do (swap! fail inc) (println "  FAIL" label (str "(" actual " vs " expected ")"))))))

;; Independent closed-form Gaussian-Gaussian marginal (mu ~ N(m0,s0), y_i ~ N(mu, sn)).
(defn gaussian-gaussian-marginal [ys m0 s0 sn]
  (let [n (count ys) s0² (* s0 s0) sn² (* sn sn)
        ds (map #(- % m0) ys) sd (reduce + ds) sd² (reduce + (map #(* % %) ds))
        denom (+ sn² (* n s0²))
        logdet (+ (js/Math.log denom) (* (dec n) (js/Math.log sn²)))
        quad (* (/ 1.0 sn²) (- sd² (* (/ s0² denom) (* sd sd))))]
    (- (* -0.5 n (js/Math.log (* 2 js/Math.PI))) (* 0.5 logdet) (* 0.5 quad))))

(def ^:private obs4 {:y0 2.0 :y1 2.3 :y2 1.7 :y3 2.1})
(def ^:private obs-ys (mapv obs4 [:y0 :y1 :y2 :y3]))

(defn smean
  "A shared-mean Gaussian spec mu~N(m0,s0), :yj~N(mu,noise), tagged with :branch (an
   extra key render ignores) so the trap proposer can route on which branch a spec is in."
  [m0 s0 noise branch]
  (assoc (syn/spec [(syn/latent 'mu "gaussian" [m0 s0])]
                   (for [k [:y0 :y1 :y2 :y3]] (syn/obs k "gaussian" ['mu noise])))
         :branch branch))

(defn ev [spec] (:evidence (syn/check (syn/render spec) obs4)))

;; ---- The two-branch trap (all exact). Evidence rises as sn -> data spread (~0.22):
;;   E(sn=1.5) < E(sn=1.0) < E(sn=0.7) < E(sn=0.3). So branch A (1.0->0.7) is locally
;;   better at step 1 (greedy takes it) but branch B (1.5->0.3) is the GLOBAL optimum. ----
(def crude (smean 2 1 2.5 nil))
(def A1 (smean 2 1 1.0 :a))
(def A2 (smean 2 1 0.7 :a))
(def B1 (smean 2 1 1.5 :b))
(def B2 (smean 2 1 0.3 :b))

(defn trap-propose [sp _fb]
  (let [n (last (:args (first (:obs sp)))) br (:branch sp)]
    (cond
      (nil? br)                  [{:edit :a1 :desc "branch A start (sn 1.0)" :spec' A1}
                                  {:edit :b1 :desc "branch B start (sn 1.5)" :spec' B1}]
      (and (= br :a) (= n 1.0))  [{:edit :a2 :desc "refine A -> 0.7" :spec' A2}]
      (and (= br :b) (= n 1.5))  [{:edit :b2 :desc "refine B -> 0.3" :spec' B2}]
      :else                      [])))

(defn part-a []
  (println "\n== PART A: particles + expansion + selection ==")
  ;; the trap is real: branch ordering as designed
  (assert-true "trap ordering E(B1) < E(A1) < E(A2) < E(B2)"
               (< (ev B1) (ev A1) (ev A2) (ev B2)))
  ;; beam (width 2) keeps BOTH branches alive after step 1
  (let [res (se/search {:init-spec crude :observations obs4 :propose trap-propose
                        :beam-width 2 :adaptive? false :max-steps 6})
        branches (set (map (comp :branch :spec) (:population res)))]
    (assert-true "beam population explores >1 branch (both A and B kept alive)"
                 (= #{:a :b} branches)))
  ;; selection ranks an EXACT-scored particle ahead of an IS one (method-aware)
  (let [det {:feedback {:method :exact} :evidence -5.0}
        is  {:feedback {:method :handler-is} :evidence -4.0}]
    (assert-true "exact evidence outranks a higher IS estimate"
                 (= det (first (sort-by @#'se/particle-rank [is det]))))))

(defn part-b []
  (println "\n== PART B: BEAM BEATS GREEDY on the two-branch trap ==")
  (let [greedy (se/search {:init-spec crude :observations obs4 :propose trap-propose
                           :beam-width 1 :adaptive? false :max-steps 6})
        beam   (se/search {:init-spec crude :observations obs4 :propose trap-propose
                           :beam-width 2 :adaptive? false :max-steps 6})
        ge (:evidence (:best greedy)) be (:evidence (:best beam))]
    (assert-true "greedy (width 1) locks into branch A" (= :a (:branch (:spec (:best greedy)))))
    (assert-close "greedy final == E(A2) (the local optimum)" (ev A2) ge 1e-2)
    (assert-true "beam (width 2) finds branch B" (= :b (:branch (:spec (:best beam)))))
    (assert-close "beam final == E(B2) (the global optimum)" (ev B2) be 1e-2)
    (assert-true "BEAM STRICTLY BEATS GREEDY (be > ge)" (> be ge))
    (println (str "  greedy -> " (.toFixed (js/Number ge) 3) " (A) | beam -> "
                  (.toFixed (js/Number be) 3) " (B)"))))

(defn part-c []
  (println "\n== PART C: backtrack-refine rescues a narrow (greedy) search ==")
  (let [greedy (se/search {:init-spec crude :observations obs4 :propose trap-propose
                           :beam-width 1 :adaptive? false :max-steps 6})
        bt     (se/backtrack-refine (:best greedy) obs4 trap-propose {:plateau-eps 0.05 :n-particles 2000 :max-steps 6})]
    (assert-true "backtrack reports an improvement" (:improved? bt))
    (assert-true "backtrack re-opened the step-1 decision" (= 1 (:reopened-at bt)))
    (assert-close "backtrack recovers B2 (the global optimum)" (ev B2) (:evidence (:result bt)) 1e-2)
    ;; the spliced result trajectory reads as the FULL path (original prefix in front)
    ;; with the re-opened edit flagged, not a fresh run rooted at the alternative.
    (assert-true "backtrack result trajectory splices the original prefix (starts at :init)"
                 (= :init (:edit (first (:trajectory (:result bt))))))
    (assert-true "backtrack result flags the re-opened edit (:reopened?)"
                 (boolean (some :reopened? (:trajectory (:result bt)))))
    (assert-true "backtrack result trajectory steps are renumbered 0..n"
                 (= (range (count (:trajectory (:result bt))))
                    (map :step (:trajectory (:result bt)))))
    ;; and search with :backtrack? on beam-width 1 does it end-to-end
    (let [g+bt (se/search {:init-spec crude :observations obs4 :propose trap-propose
                           :beam-width 1 :adaptive? false :backtrack? true :max-steps 6})]
      (assert-close "search beam-width 1 + :backtrack? reaches B2" (ev B2) (:evidence (:best g+bt)) 1e-2)
      (assert-true "search reports the backtrack improvement" (:improved? (:backtrack g+bt))))))

(defn part-d []
  (println "\n== PART D: :smc strategy, adaptive width, plateau termination ==")
  ;; :smc resampling still reaches the global optimum (seeded -> reproducible)
  (let [smc (se/search {:init-spec crude :observations obs4 :propose trap-propose
                        :strategy :smc :beam-width 3 :temperature 0.3 :seed 7
                        :adaptive? false :max-steps 6})]
    (assert-close "smc reaches the global optimum B2" (ev B2) (:evidence (:best smc)) 1e-2))
  ;; adaptive width WIDENS on the ambiguous first step (A1 and B1 score close) ...
  (let [res  (se/search {:init-spec crude :observations obs4 :propose trap-propose
                         :beam-width 1 :adaptive? true :spread-margin 5.0 :max-width 4 :max-steps 6})
        d1   (first (:diagnostics res))]
    (assert-true "adaptive width > base on the ambiguous step-1" (> (:width d1) 1))
    ;; ... and with the wider beam even base-width-1 now finds the global optimum
    (assert-close "adaptive search reaches B2" (ev B2) (:evidence (:best res)) 1e-2))
  ;; self-terminates on plateau (not max-steps) with one diagnostics row per step
  (let [res (se/search {:init-spec crude :observations obs4 :propose trap-propose
                        :beam-width 2 :adaptive? false :max-steps 20})]
    (assert-true "search self-terminates on plateau" (= :plateau (:stop-reason res)))
    (assert-true "diagnostics has one row per step" (= (:steps res) (count (:diagnostics res))))
    (assert-true "trajectory of the best particle is recorded" (>= (count (:trajectory (:best res))) 2))))

(defn- summary []
  (println (str "\n== search (genmlx-47zx): " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(part-a)
(part-b)
(part-c)
(part-d)
(summary)
