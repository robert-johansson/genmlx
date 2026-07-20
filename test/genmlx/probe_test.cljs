;; @tier fast
(ns genmlx.probe-test
  "Pure-math tests for genmlx.world.probe (the GRPO probe-set eval,
   genmlx-lkt0): probe-set selection determinism, per-prompt summaries,
   cluster-pooled SE, and the paired pre/post delta t — verified against
   hand-computed values. No GPU, no engine.

   Run: bun run --bun nbb test/genmlx/probe_test.cljs"
  (:require [genmlx.world.probe :as probe]))

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [msg v]
  (if v
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg))))

(defn- assert-close [msg expected actual tol]
  (assert-true (str msg " (expected ~" expected ", got " actual ")")
               (and (number? actual)
                    (< (js/Math.abs (- expected actual)) tol))))

;; ============================================================
;; probe-indices
;; ============================================================

(println "\n== probe-indices ==")

(assert-true "evenly spaced over 37 (the night's kept corpus)"
             (= [0 6 12 18 24 30] (probe/probe-indices 37 6)))
(assert-true "deterministic" (= (probe/probe-indices 37 6) (probe/probe-indices 37 6)))
(assert-true "k > n degrades to all indices" (= [0 1 2] (probe/probe-indices 3 8)))
(assert-true "k = n is identity" (= [0 1 2 3] (probe/probe-indices 4 4)))
(assert-true "k = 0 is empty" (= [] (probe/probe-indices 5 0)))
(assert-true "distinct indices always" (apply distinct? (probe/probe-indices 10 7)))

;; ============================================================
;; summarize-prompt
;; ============================================================

(println "\n== summarize-prompt ==")

(let [s (probe/summarize-prompt "a/1"
                                [-1 0.4 0.9 0.9]
                                ["<tool_call>\nunfinished"
                                 "prose"
                                 "ok"
                                 "ok"])]
  (assert-true "carries key + n" (and (= "a/1" (:key s)) (= 4 (:n s))))
  (assert-close "reward-mean" 0.3 (:reward-mean s) 1e-9)
  ;; sample sd of [-1 0.4 0.9 0.9]: mean 0.3, devs [-1.3 .1 .6 .6] ->
  ;; ss = 1.69+.01+.36+.36 = 2.42, /3 = 0.80667, sqrt = 0.898146
  (assert-close "reward-std (sample, n-1)" 0.898146 (:reward-std s) 1e-4)
  (assert-close "floored-frac counts exact floor" 0.25 (:floored-frac s) 1e-9)
  (assert-close "truncated-frac via strip-truncated-tail" 0.25 (:truncated-frac s) 1e-9)
  (assert-close "mean-chars" (/ (+ 22 5 2 2) 4) (:mean-chars s) 1e-9))

(let [s (probe/summarize-prompt "b/1" [0.5] ["x"] {:floor 0.5})]
  (assert-true "std nil below 2 observations" (nil? (:reward-std s)))
  (assert-close "floor value is an opt" 1.0 (:floored-frac s) 1e-9))

;; ============================================================
;; pool — cluster (prompt-level) SE
;; ============================================================

(println "\n== pool ==")

(let [mk (fn [k m] {:key k :n 8 :reward-mean m :reward-std 0.1
                    :floored-frac 0.25 :truncated-frac 0.5 :mean-chars 100})
      pooled (probe/pool [(mk "a" 0.2) (mk "b" 0.4) (mk "c" 0.6)])]
  (assert-true "counts prompts + rollouts" (and (= 3 (:n-prompts pooled))
                                                (= 24 (:n-rollouts pooled))))
  (assert-close "pooled mean = mean of prompt means" 0.4 (:mean pooled) 1e-9)
  ;; sd of [.2 .4 .6] = 0.2, / sqrt 3 = 0.11547
  (assert-close "clustered SE = sd(prompt means)/sqrt(P)" 0.11547 (:se pooled) 1e-4)
  (assert-close "fractions pass through" 0.25 (:floored-frac pooled) 1e-9))

(assert-true "single-prompt pool has nil SE"
             (nil? (:se (probe/pool [{:key "a" :n 8 :reward-mean 0.3}]))))

;; ============================================================
;; pass-record
;; ============================================================

(println "\n== pass-record ==")

(let [r (probe/pass-record "pre" 0 [{:key "a" :n 2 :reward-mean 0.5}])]
  (assert-true "record shape" (and (= "pre" (:probe r)) (= 0 (:step r))
                                   (= 1 (count (:prompts r)))
                                   (some? (:pooled r)))))

;; ============================================================
;; delta — the paired pre/post test
;; ============================================================

(println "\n== delta ==")

;; A uniform +0.1 shift on 4 prompts: mean 0.1, sd 0 -> sem 0 -> t nil
;; (degenerate: no variance). Perturb one delta to get a real t.
(let [pre  [{:key "a" :reward-mean 0.2} {:key "b" :reward-mean 0.4}
            {:key "c" :reward-mean 0.6} {:key "d" :reward-mean 0.1}]
      post [{:key "a" :reward-mean 0.3} {:key "b" :reward-mean 0.5}
            {:key "c" :reward-mean 0.7} {:key "d" :reward-mean 0.3}]
      d (probe/delta pre post)]
  (assert-true "pairs matched by key" (= 4 (:n-pairs d)))
  ;; deltas [.1 .1 .1 .2]: mean .125, sd .05, sem .025, t 5
  (assert-close "mean paired delta" 0.125 (:mean d) 1e-9)
  (assert-close "sem" 0.025 (:sem d) 1e-6)
  (assert-close "t" 5.0 (:t d) 1e-6))

(let [d (probe/delta [{:key "a" :reward-mean 0.2} {:key "gone" :reward-mean 0.9}]
                     [{:key "a" :reward-mean 0.2} {:key "new" :reward-mean 0.1}])]
  (assert-true "unmatched keys are dropped, not mispaired" (= 1 (:n-pairs d)))
  (assert-close "single pair: delta 0" 0.0 (:mean d) 1e-9)
  (assert-true "single pair: no sem/t" (and (nil? (:sem d)) (nil? (:t d)))))

;; MDE sanity: the design point the bean commits to. P=6 prompts of N=64
;; rollouts with per-rollout sd 0.8 -> per-prompt SE 0.1; if prompt means
;; scatter with sd ~0.1 the pooled/paired SEM ~ 0.04 -> ~0.08-0.12 MDE at
;; t=2..3, an order of magnitude below the night's ±0.25.
(let [pre  (mapv (fn [i] {:key (str i) :reward-mean (* 0.1 i)}) (range 6))
      post (mapv (fn [i] {:key (str i) :reward-mean (+ 0.09 (* 0.1 i)
                                                       (* 0.01 (- i 2.5)))})
                 (range 6))
      d (probe/delta pre post)]
  (assert-true "a +0.09 shift with small scatter is resolvable (t > 3)"
               (> (:t d) 3)))

;; ============================================================
;; Summary
;; ============================================================

(let [p @pass-count f @fail-count]
  (println (str "\n=== " p "/" (+ p f) " PASS ==="))
  (when (pos? f)
    (println (str "!!! " f " FAILURES !!!"))
    (set! (.-exitCode js/process) 1)))
