(ns genmlx.m5-fused-loop-test
  "M5: Fused loop execution tests.
   Verifies that the accumulate-and-fuse handler produces identical
   scores/weights to the standard per-iteration handler."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vec]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.compiled :as compiled])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg "expected:" expected "got:" actual))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc) (println "  PASS:" msg))
      (do (swap! fail-count inc) (println "  FAIL:" msg "expected:" expected "got:" actual "diff:" diff)))))

;; ---------------------------------------------------------------------------
;; Shared models
;; ---------------------------------------------------------------------------

(def linreg
  (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

(def simple-loop
  (gen [n-obs]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (dotimes [i n-obs]
        (trace (keyword (str "x" i))
               (dist/gaussian mu 1)))
      mu)))

;; ---------------------------------------------------------------------------
;; Helper: run standard (non-fused) vgenerate for comparison
;; ---------------------------------------------------------------------------

(defn vgenerate-standard
  "Run vgenerate using standard handler (no fusion) for baseline comparison."
  [gf args constraints n key]
  (let [key (rng/ensure-key key)
        _ (rng/seed! key)
        result (rt/run-handler h/batched-generate-transition
                               {:choices cm/EMPTY :score (mx/scalar 0.0)
                                :weight (mx/scalar 0.0) :key key
                                :constraints constraints :batch-size n :batched? true
                                :loop-stacks (dyn/prepare-loop-stacks constraints (:schema gf))}
                               (fn [rt] (apply (:body-fn gf) rt args)))]
    (vec/->VectorizedTrace gf args (:choices result) (:score result)
                           (:weight result) n (:retval result))))

(defn vsimulate-standard
  "Run vsimulate using standard handler (no fusion) for baseline comparison."
  [gf args n key]
  (let [key (rng/ensure-key key)
        _ (rng/seed! key)
        result (rt/run-handler h/batched-simulate-transition
                               {:choices cm/EMPTY :score (mx/scalar 0.0)
                                :key key :batch-size n :batched? true}
                               (fn [rt] (apply (:body-fn gf) rt args)))]
    (vec/->VectorizedTrace gf args (:choices result) (:score result)
                           (mx/zeros [n]) n (:retval result))))

;; ---------------------------------------------------------------------------
;; 1. match-loop-addr (via prepare-loop-stacks proxy)
;; ---------------------------------------------------------------------------

(println "\n-- 1. Fusability detection --")

(let [schema (:schema linreg)]
  (assert-true "linreg has loop-sites" (seq (:loop-sites schema)))
  (assert-true "linreg loop is homogeneous" (:homogeneous? (first (:loop-sites schema))))
  (assert-true "linreg loop is rewritable" (:rewritable? (first (:loop-sites schema)))))

;; ---------------------------------------------------------------------------
;; 2. Fused generate numerical equivalence
;; ---------------------------------------------------------------------------

(println "\n-- 2. Fused generate numerical equivalence --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      obs (dyn/loop-obs "y" [3.0 5.0 7.0 9.0 11.0])
      n 100
      key (rng/fresh-key)
      vt-std (vgenerate-standard linreg [xs] obs n key)
      vt-fused (dyn/vgenerate linreg [xs] obs n key)]
  (assert-close "mean weights match"
                (mx/item (mx/mean (:weight vt-std)))
                (mx/item (mx/mean (:weight vt-fused)))
                0.01)
  (assert-equal "weight shapes match"
                (mx/shape (:weight vt-std))
                (mx/shape (:weight vt-fused)))
  ;; Per-element weight comparison
  (let [diff (mx/item (mx/sum (mx/abs (mx/subtract (:weight vt-std) (:weight vt-fused)))))]
    (assert-close "total weight diff near zero" 0.0 diff 0.01)))

;; ---------------------------------------------------------------------------
;; 3. Fused simulate numerical equivalence
;; ---------------------------------------------------------------------------

(println "\n-- 3. Fused simulate numerical equivalence --")

(let [xs [1.0 2.0 3.0]
      n 100
      key (rng/fresh-key)
      vt-std (vsimulate-standard linreg [xs] n key)
      vt-fused (dyn/vsimulate linreg [xs] n key)]
  (assert-close "mean scores match"
                (mx/item (mx/mean (:score vt-std)))
                (mx/item (mx/mean (:score vt-fused)))
                0.01)
  (assert-equal "score shapes match"
                (mx/shape (:score vt-std))
                (mx/shape (:score vt-fused)))
  (let [diff (mx/item (mx/sum (mx/abs (mx/subtract (:score vt-std) (:score vt-fused)))))]
    (assert-close "total score diff near zero" 0.0 diff 0.01)))

;; ---------------------------------------------------------------------------
;; 4. Shape correctness
;; ---------------------------------------------------------------------------

(println "\n-- 4. Shape correctness --")

(let [xs [1.0 2.0 3.0]
      obs (dyn/loop-obs "y" [10.0 20.0 30.0])
      n 50
      key (rng/fresh-key)
      vt (dyn/vgenerate linreg [xs] obs n key)]
  (assert-equal "weight shape [N]" [50] (mx/shape (:weight vt)))
  (assert-equal "score shape [N]" [50] (mx/shape (:score vt)))
  (assert-equal "n-particles" 50 (:n-particles vt))
  ;; Constrained sites should be scalar
  (let [y0-val (cm/get-value (cm/get-submap (:choices vt) :y0))]
    (assert-true "y0 is scalar (constrained)" (= [] (mx/shape y0-val)))))

;; ---------------------------------------------------------------------------
;; 5. Non-loop addresses unchanged
;; ---------------------------------------------------------------------------

(println "\n-- 5. Non-loop addresses unchanged --")

(let [xs [1.0 2.0 3.0]
      obs (dyn/merge-obs (cm/choicemap :slope 2.0 :intercept 1.0)
                         (dyn/loop-obs "y" [3.0 5.0 7.0]))
      n 50
      key (rng/fresh-key)
      vt-std (vgenerate-standard linreg [xs] obs n key)
      vt-fused (dyn/vgenerate linreg [xs] obs n key)]
  ;; Static sites should be identical
  (let [slope-std (cm/get-value (cm/get-submap (:choices vt-std) :slope))
        slope-fused (cm/get-value (cm/get-submap (:choices vt-fused) :slope))]
    (assert-true "slope values identical"
                 (if (and (mx/array? slope-std) (mx/array? slope-fused))
                   (zero? (mx/item (mx/sum (mx/abs (mx/subtract slope-std slope-fused)))))
                   (= slope-std slope-fused))))
  (let [int-std (cm/get-value (cm/get-submap (:choices vt-std) :intercept))
        int-fused (cm/get-value (cm/get-submap (:choices vt-fused) :intercept))]
    (assert-true "intercept values identical"
                 (if (and (mx/array? int-std) (mx/array? int-fused))
                   (zero? (mx/item (mx/sum (mx/abs (mx/subtract int-std int-fused)))))
                   (= int-std int-fused))))
  ;; Weights should still match
  (assert-close "weights match with static constraints"
                (mx/item (mx/mean (:weight vt-std)))
                (mx/item (mx/mean (:weight vt-fused)))
                0.01))

;; ---------------------------------------------------------------------------
;; 6. Fallback for non-fusable loops
;; ---------------------------------------------------------------------------

(println "\n-- 6. Fallback for non-fusable loops --")

;; Model with branch inside loop → not rewritable → standard handler
(let [branch-loop-model
      (gen [xs]
        (let [mu (trace :mu (dist/gaussian 0 10))]
          (doseq [[j x] (map-indexed vector xs)]
            (if (> x 0)
              (trace (keyword (str "y" j)) (dist/gaussian mu 1))
              (trace (keyword (str "y" j)) (dist/uniform -10 10))))
          mu))
      schema (:schema branch-loop-model)]
  (assert-true "branch loop not rewritable"
               (not (:rewritable? (first (:loop-sites schema)))))
  ;; Should still work (falls back to standard handler)
  (let [obs (cm/choicemap :y0 1.0 :y1 2.0)
        key (rng/fresh-key)
        vt (dyn/vgenerate branch-loop-model [[1.0 2.0]] obs 10 key)]
    (assert-equal "fallback produces correct n" 10 (:n-particles vt))))

;; ---------------------------------------------------------------------------
;; 7. Simple loop model (dotimes)
;; ---------------------------------------------------------------------------

(println "\n-- 7. Simple loop model (dotimes) --")

(let [obs (dyn/loop-obs "x" [1.0 2.0 3.0 4.0])
      n 100
      key (rng/fresh-key)
      vt-std (vgenerate-standard simple-loop [4] obs n key)
      vt-fused (dyn/vgenerate simple-loop [4] obs n key)]
  (assert-close "dotimes: mean weights match"
                (mx/item (mx/mean (:weight vt-std)))
                (mx/item (mx/mean (:weight vt-fused)))
                0.01)
  (let [diff (mx/item (mx/sum (mx/abs (mx/subtract (:weight vt-std) (:weight vt-fused)))))]
    (assert-close "dotimes: total weight diff near zero" 0.0 diff 0.01)))

;; ---------------------------------------------------------------------------
;; 8. Large T scaling
;; ---------------------------------------------------------------------------

(println "\n-- 8. Large T scaling (T=100) --")

(let [T 100
      obs (dyn/loop-obs "x" (repeat T 5.0))
      n 50
      key (rng/fresh-key)
      vt-std (vgenerate-standard simple-loop [T] obs n key)
      vt-fused (dyn/vgenerate simple-loop [T] obs n key)]
  (assert-close "T=100: mean weights match"
                (mx/item (mx/mean (:weight vt-std)))
                (mx/item (mx/mean (:weight vt-fused)))
                0.1)
  (assert-equal "T=100: shapes match"
                (mx/shape (:weight vt-std))
                (mx/shape (:weight vt-fused))))

;; ---------------------------------------------------------------------------
;; 9. Equivalence: loop-obs + fused ≡ manual choicemap + standard
;; ---------------------------------------------------------------------------

(println "\n-- 9. loop-obs + fused ≡ manual choicemap --")

(let [xs [1.0 2.0 3.0]
      manual (cm/choicemap :y0 10.0 :y1 20.0 :y2 30.0)
      loop-c (dyn/loop-obs "y" [10.0 20.0 30.0])
      n 100
      key (rng/fresh-key)
      vt-manual (dyn/vgenerate linreg [xs] manual n key)
      vt-loop (dyn/vgenerate linreg [xs] loop-c n key)]
  (assert-close "manual vs loop-obs weights"
                (mx/item (mx/mean (:weight vt-manual)))
                (mx/item (mx/mean (:weight vt-loop)))
                0.01)
  (assert-equal "same n-particles"
                (:n-particles vt-manual)
                (:n-particles vt-loop)))

;; ---------------------------------------------------------------------------
;; 10. Benchmark: fused vs standard
;; ---------------------------------------------------------------------------

(println "\n-- 10. Benchmark --")

(let [T 50
      obs (dyn/loop-obs "x" (repeat T 5.0))
      n 200
      key (rng/fresh-key)
      iters 20
      ;; Warm up
      _ (dyn/vgenerate simple-loop [T] obs n key)
      _ (vgenerate-standard simple-loop [T] obs n key)
      ;; Benchmark standard
      t0-std (js/Date.now)
      _ (dotimes [_ iters]
          (vgenerate-standard simple-loop [T] obs n key))
      t1-std (js/Date.now)
      ms-std (- t1-std t0-std)
      ;; Benchmark fused
      t0-fused (js/Date.now)
      _ (dotimes [_ iters]
          (dyn/vgenerate simple-loop [T] obs n key))
      t1-fused (js/Date.now)
      ms-fused (- t1-fused t0-fused)
      speedup (/ ms-std (max ms-fused 1))]
  (println (str "  Standard: " ms-std "ms (" iters " iters, T=" T ", N=" n ")"))
  (println (str "  Fused:    " ms-fused "ms"))
  (println (str "  Speedup:  " (.toFixed speedup 1) "x")))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n========================================")
(println (str "M5 Fused Loop Execution: " @pass-count "/" (+ @pass-count @fail-count)
              " passed" (when (pos? @fail-count) (str ", " @fail-count " FAILED"))))
(println "========================================")

(when (pos? @fail-count)
  (js/process.exit 1))
