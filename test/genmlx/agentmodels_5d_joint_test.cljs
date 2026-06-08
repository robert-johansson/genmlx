;; @tier slow
;; Headless tests for 5d procrastination JOINT inference (agentmodels Ch 5d).
;; Run: bun run --bun nbb test/genmlx/agentmodels_5d_joint_test.cljs

(ns genmlx.agentmodels-5d-joint-test
  (:require [agentmodels.joint-5d-inference :as j5d]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol)
    (do (vswap! passed inc) (println " PASS" msg "  =" actual))
    (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))

(println "\n== 5d: Optimal vs Possibly-Discounting (8 observed waits) ==")
(def opt  (j5d/analyze {:optimal? true}))
(def disc (j5d/analyze {:optimal? false}))
(println "  Optimal     : pwork" (:predict-work opt)  " E[reward]" (:E-reward opt)  " E[alpha]" (:E-alpha opt))
(println "  Discounting : pwork" (:predict-work disc) " E[reward]" (:E-reward disc) " E[discount]" (:E-discount disc))

(assert-close "Optimal predictWorkLastMinute ~= 0.0056"        0.0056 (:predict-work opt)  5e-4)
(assert-close "Discounting predictWorkLastMinute ~= 0.216"     0.216  (:predict-work disc) 5e-3)
(assert-true  "Discounting pwork > Optimal pwork"              (> (:predict-work disc) (:predict-work opt)))
(assert-true  "ratio Disc/Opt > 10"                            (> (/ (:predict-work disc) (:predict-work opt)) 10))
(assert-true  "both pwork < 0.3"                               (and (< (:predict-work opt) 0.3) (< (:predict-work disc) 0.3)))
(assert-close "Optimal explains waiting by low reward ~0.530"  0.530  (:E-reward opt)  0.02)
(assert-close "Optimal infers high noise E[alpha] ~457.5"      457.5  (:E-alpha opt)  2.0)
(assert-close "Discounting keeps reward higher ~2.849"         2.849  (:E-reward disc) 0.05)
(assert-close "Discounting E[discount] ~2.631"                 2.631  (:E-discount disc) 0.05)

(println "\n== 5d: posterior revision when the task COMPLETES (append work@W_8) ==")
(def opt2  (j5d/analyze {:optimal? true  :extra-work? true}))
(def disc2 (j5d/analyze {:optimal? false :extra-work? true}))
(println "  Optimal      reward" (:E-reward opt)  "->" (:E-reward opt2)  "  alpha" (:E-alpha opt)  "->" (:E-alpha opt2))
(println "  Discounting  reward" (:E-reward disc) "->" (:E-reward disc2) "  alpha" (:E-alpha disc) "->" (:E-alpha disc2))

(assert-true  "Optimal E[reward] rises by > 2.0 on completion"  (> (- (:E-reward opt2) (:E-reward opt)) 2.0))
(assert-true  "Optimal E[alpha] collapses (> 100x drop)"        (> (/ (:E-alpha opt) (:E-alpha opt2)) 100))
(assert-true  "Discounting alpha barely moves (< 3x change)"    (< (/ (max (:E-alpha disc) (:E-alpha disc2))
                                                                      (min (:E-alpha disc) (:E-alpha disc2))) 3))
(assert-true  "Discounting reward rises moderately (> 1.0)"     (> (- (:E-reward disc2) (:E-reward disc)) 1.0))
(assert-true  "Optimal alpha-revision >> Discounting alpha-revision"
              (> (Math/abs (- (:E-alpha opt) (:E-alpha opt2)))
                 (Math/abs (- (:E-alpha disc) (:E-alpha disc2)))))

(println "\n== 5d: online posterior sequence ==")
(def series (j5d/online-posteriors (j5d/demo-cfg {:optimal? false}) (j5d/build-agents (j5d/demo-cfg {:optimal? false})) 8 2))
(println "  prefix lengths:" (mapv :n series))
(assert-true  "online series has prior + one-per-observation"   (= (count series) 9))
(assert-true  "index 0 is the prior (E[discount] = prior mean 1.5)"
              (< (Math/abs (- (:E-discount (first series)) 1.5)) 1e-6))
(println "  predict-work :" (mapv #(.toFixed (:predict-work %) 3) series))
(println "  E[discount]  :" (mapv #(.toFixed (:E-discount %) 3) series))
;; Observing waiting => the model infers a procrastinator: P(work) FALLS, E[discount] RISES.
(assert-true  "predict-work FALLS as waiting accumulates (looks like a procrastinator)"
              (< (:predict-work (last series)) (:predict-work (first series))))
(assert-true  "E[discount] RISES as waiting accumulates (prior 1.5 -> ~2.63)"
              (> (:E-discount (last series)) (:E-discount (first series))))

(println "\n== 5d: normalization & structure ==")
(let [post (:joint (:posterior disc))]
  (assert-true "joint posterior sums to 1" (< (Math/abs (- 1.0 (reduce + (vals post)))) 1e-6)))
(let [m (:marginals (:posterior disc))]
  (assert-true "reward marginal sums to 1"   (< (Math/abs (- 1.0 (reduce + (vals (:reward m))))) 1e-6))
  (assert-true "discount marginal sums to 1" (< (Math/abs (- 1.0 (reduce + (vals (:discount m))))) 1e-6)))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
