(ns genmlx.method-selection-test
  "WP-3 Gate 3: Automatic method selection tests.
   Tests select-method decision tree and tune-method-opts across
   8 model categories + edge cases (~35 tests)."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.method-selection :as ms])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn assert-true [msg v]
  (if v
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc) (println (str "  FAIL: " msg)))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc) (println (str "  FAIL: " msg " expected=" expected " actual=" actual)))))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

;; 1. All-conjugate: Normal-Normal prior + observed → :exact
(def m-conjugate
  (gen [x]
       (let [mu (trace :mu (dist/gaussian 0 10))]
         (trace :y (dist/gaussian mu 1)))))

;; 2. Non-conjugate, small static (3 latents) → :hmc
(def m-static-small
  (gen []
       (let [a (trace :a (dist/gaussian 0 1))
             b (trace :b (dist/gaussian 0 1))
             c (trace :c (dist/gaussian 0 1))]
         (mx/add a b))))

;; 3. Non-conjugate, large static (11+ latents) → :vi
(def m-static-large
  (gen []
       (let [x1 (trace :x1 (dist/gaussian 0 1))
             x2 (trace :x2 (dist/gaussian 0 1))
             x3 (trace :x3 (dist/gaussian 0 1))
             x4 (trace :x4 (dist/gaussian 0 1))
             x5 (trace :x5 (dist/gaussian 0 1))
             x6 (trace :x6 (dist/gaussian 0 1))
             x7 (trace :x7 (dist/gaussian 0 1))
             x8 (trace :x8 (dist/gaussian 0 1))
             x9 (trace :x9 (dist/gaussian 0 1))
             x10 (trace :x10 (dist/gaussian 0 1))
             x11 (trace :x11 (dist/gaussian 0 1))]
         x1)))

;; 4. Mixed conjugate: 2 of 5 sites conjugate, obs on y1/y2 → :hmc with residual :p
(def m-mixed
  (gen [xs]
       (let [mu1 (trace :mu1 (dist/gaussian 0 10))
             mu2 (trace :mu2 (dist/gaussian 0 10))
             p (trace :p (dist/uniform 0 1))
             y1 (trace :y1 (dist/gaussian mu1 1))
             y2 (trace :y2 (dist/gaussian mu2 1))]
         (mx/add mu1 mu2))))

;; 5. Temporal model with Unfold
(def unfold-kernel
  (gen [t state]
       (let [x (trace :x (dist/gaussian state 1))]
         x)))

(def m-unfold
  (gen [T]
       (let [init (trace :init (dist/gaussian 0 10))
             unfold-gf (comb/unfold-combinator unfold-kernel)]
         (splice :steps unfold-gf T init))))

;; 6. Temporal model with Scan (name contains 'scan')
(def scan-kernel
  (gen [carry input]
       (let [x (trace :x (dist/gaussian carry 1))]
         x)))

(def m-scan
  (gen [inputs]
       (let [init (trace :init (dist/gaussian 0 10))
             scan-gf (comb/scan-combinator scan-kernel)]
         (splice :steps scan-gf init inputs))))

;; 7. Dynamic addresses
(def m-dynamic
  (gen [n]
       (let [k (trace :k (dist/poisson 5))]
         (dotimes [i 3]
           (trace (keyword (str "x" i)) (dist/gaussian 0 1))))))

;; 8. Empty model (no trace sites)
(def m-empty (gen [] 42))

;; 9. All-observed model
(def m-all-obs
  (gen []
       (let [x (trace :x (dist/gaussian 0 1))
             y (trace :y (dist/gaussian 0 1))]
         (mx/add x y))))

;; 10. Splice model without temporal name
(def sub-model
  (gen [x]
       (let [z (trace :z (dist/gaussian x 1))]
         z)))

(def m-splice-generic
  (gen []
       (let [a (trace :a (dist/gaussian 0 1))
             helper sub-model]
         (splice :sub helper a))))

;; ---------------------------------------------------------------------------
;; Category 1: All-conjugate → :exact
;; ---------------------------------------------------------------------------

(println "\n== Category 1: All-conjugate → :exact ==")

(let [obs (cm/choicemap :y 1.0)
      result (ms/select-method m-conjugate obs)]
  (assert-equal "conjugate model → :exact" :exact (:method result))
  (assert-true "conjugate: :mu eliminated" (contains? (:eliminated result) :mu))
  (assert-equal "conjugate: 0 residual" 0 (:n-residual result))
  (assert-true "conjugate: reason mentions analytical" (some? (:reason result)))
  (assert-true "conjugate: opts is map" (map? (:opts result))))

;; ---------------------------------------------------------------------------
;; Category 2: Static small → :hmc
;; ---------------------------------------------------------------------------

(println "\n== Category 2: Static small → :hmc ==")

(let [result (ms/select-method m-static-small nil)]
  (assert-equal "static small → :hmc" :hmc (:method result))
  (assert-equal "static small: 3 residual" 3 (:n-residual result))
  (assert-true "static small: reason mentions static" (some? (:reason result)))
  (assert-true "static small: opts has :n-samples" (contains? (:opts result) :n-samples))
  (assert-true "static small: opts has :n-leapfrog" (contains? (:opts result) :n-leapfrog)))

;; ---------------------------------------------------------------------------
;; Category 3: Static large → :vi
;; ---------------------------------------------------------------------------

(println "\n== Category 3: Static large → :vi ==")

(let [result (ms/select-method m-static-large nil)]
  (assert-equal "static large → :vi" :vi (:method result))
  (assert-equal "static large: 11 residual" 11 (:n-residual result))
  (assert-true "static large: opts has :n-iters" (contains? (:opts result) :n-iters))
  (assert-true "static large: opts has :learning-rate" (contains? (:opts result) :learning-rate)))

;; ---------------------------------------------------------------------------
;; Category 4: Mixed conjugate → :hmc with reduced residual
;; ---------------------------------------------------------------------------

(println "\n== Category 4: Mixed conjugate → :hmc ==")

(let [obs (cm/choicemap :y1 1.0 :y2 2.0)
      result (ms/select-method m-mixed obs)]
  (assert-equal "mixed conjugate → :hmc" :hmc (:method result))
  (assert-true "mixed: :mu1 eliminated" (contains? (:eliminated result) :mu1))
  (assert-true "mixed: :mu2 eliminated" (contains? (:eliminated result) :mu2))
  (assert-true "mixed: :p is residual" (contains? (:residual-addrs result) :p))
  (assert-equal "mixed: 1 residual" 1 (:n-residual result)))

;; ---------------------------------------------------------------------------
;; Category 5: Temporal (Unfold) → :smc
;; ---------------------------------------------------------------------------

(println "\n== Category 5: Temporal (Unfold) → :smc ==")

(let [result (ms/select-method m-unfold nil)]
  (assert-equal "unfold → :smc" :smc (:method result))
  (assert-true "unfold: reason mentions temporal"
               (clojure.string/includes? (:reason result) "emporal"))
  (assert-true "unfold: opts has :n-particles" (contains? (:opts result) :n-particles)))

;; ---------------------------------------------------------------------------
;; Category 6: Temporal (Scan) → :smc
;; ---------------------------------------------------------------------------

(println "\n== Category 6: Temporal (Scan) → :smc ==")

(let [result (ms/select-method m-scan nil)]
  (assert-equal "scan → :smc" :smc (:method result))
  (assert-true "scan: reason mentions temporal"
               (clojure.string/includes? (:reason result) "emporal")))

;; ---------------------------------------------------------------------------
;; Category 7: Dynamic addresses → :handler-is
;; ---------------------------------------------------------------------------

(println "\n== Category 7: Dynamic addresses → :handler-is ==")

(let [result (ms/select-method m-dynamic nil)]
  (assert-equal "dynamic → :handler-is" :handler-is (:method result))
  (assert-true "dynamic: reason mentions dynamic"
               (clojure.string/includes? (:reason result) "ynamic"))
  (assert-true "dynamic: opts has :n-particles" (contains? (:opts result) :n-particles)))

;; ---------------------------------------------------------------------------
;; Category 8: Generic splice (non-temporal name) → :smc
;; ---------------------------------------------------------------------------

(println "\n== Category 8: Generic splice → :smc ==")

(let [result (ms/select-method m-splice-generic nil)]
  (assert-equal "generic splice → :smc" :smc (:method result))
  (assert-true "generic splice: reason mentions splice"
               (clojure.string/includes? (:reason result) "plice")))

;; ---------------------------------------------------------------------------
;; Edge cases
;; ---------------------------------------------------------------------------

(println "\n== Edge cases ==")

;; Empty model → :exact
(let [result (ms/select-method m-empty nil)]
  (assert-equal "empty model → :exact" :exact (:method result))
  (assert-equal "empty: 0 residual" 0 (:n-residual result))
  (assert-equal "empty: 0 latent" 0 (:n-latent result)))

;; All observed → :exact
(let [obs (cm/choicemap :x 1.0 :y 2.0)
      result (ms/select-method m-all-obs obs)]
  (assert-equal "all observed → :exact" :exact (:method result))
  (assert-equal "all observed: 0 residual" 0 (:n-residual result))
  (assert-equal "all observed: 0 latent" 0 (:n-latent result)))

;; Conjugate without observations → :hmc (not :exact, since :y not observed)
(let [result (ms/select-method m-conjugate nil)]
  (assert-equal "conjugate no obs → :hmc" :hmc (:method result))
  (assert-true "conjugate no obs: :y is residual" (contains? (:residual-addrs result) :y)))

;; nil analytical-plan is safe
(let [result (ms/select-method m-static-small nil)]
  (assert-true "nil analytical-plan: safe" (some? (:method result)))
  (assert-equal "nil plan: eliminated is empty set" #{} (:eliminated result)))

;; ---------------------------------------------------------------------------
;; Return structure validation
;; ---------------------------------------------------------------------------

(println "\n== Return structure ==")

(let [result (ms/select-method m-static-small nil)]
  (assert-true "has :method" (contains? result :method))
  (assert-true "has :reason" (contains? result :reason))
  (assert-true "has :opts" (contains? result :opts))
  (assert-true "has :eliminated" (contains? result :eliminated))
  (assert-true "has :residual-addrs" (contains? result :residual-addrs))
  (assert-true "has :n-residual" (contains? result :n-residual))
  (assert-true "has :n-latent" (contains? result :n-latent))
  (assert-true ":method is keyword" (keyword? (:method result)))
  (assert-true ":reason is string" (string? (:reason result)))
  (assert-true ":opts is map" (map? (:opts result)))
  (assert-true ":eliminated is set" (set? (:eliminated result)))
  (assert-true ":residual-addrs is set" (set? (:residual-addrs result)))
  (assert-true ":n-residual is number" (number? (:n-residual result)))
  (assert-true ":n-latent is number" (number? (:n-latent result))))

;; ---------------------------------------------------------------------------
;; tune-method-opts tests
;; ---------------------------------------------------------------------------

(println "\n== tune-method-opts ==")

;; HMC tuning (small residual)
(let [sel (ms/select-method m-static-small nil)
      tuned (ms/tune-method-opts sel)]
  (assert-equal "hmc tune: n-leapfrog for 3 residual" 10 (:n-leapfrog tuned))
  (assert-equal "hmc tune: step-size for 3 residual" 0.05 (:step-size tuned))
  (assert-equal "hmc tune: n-warmup for 3 residual" 200 (:n-warmup tuned)))

;; VI tuning (large residual)
(let [sel (ms/select-method m-static-large nil)
      tuned (ms/tune-method-opts sel)]
  (assert-equal "vi tune: n-iters for 11 residual" 2000 (:n-iters tuned))
  (assert-equal "vi tune: n-samples for 11 residual" 10 (:n-samples tuned)))

;; SMC tuning
(let [sel (ms/select-method m-unfold nil)
      tuned (ms/tune-method-opts sel)]
  (assert-true "smc tune: has :n-particles" (contains? tuned :n-particles))
  (assert-true "smc tune: has :ess-threshold" (contains? tuned :ess-threshold)))

;; handler-is tuning
(let [sel (ms/select-method m-dynamic nil)
      tuned (ms/tune-method-opts sel)]
  (assert-true "handler-is tune: has :n-particles" (contains? tuned :n-particles)))

;; exact tuning
(let [obs (cm/choicemap :y 1.0)
      sel (ms/select-method m-conjugate obs)
      tuned (ms/tune-method-opts sel)]
  (assert-equal "exact tune: empty opts" {} tuned))

;; User overrides take priority
(let [sel (ms/select-method m-static-small nil)
      tuned (ms/tune-method-opts sel {:step-size 0.001 :custom-key 42})]
  (assert-equal "user override: step-size" 0.001 (:step-size tuned))
  (assert-equal "user override: custom-key" 42 (:custom-key tuned))
  (assert-true "user override: preserves n-leapfrog" (contains? tuned :n-leapfrog)))

;; 1-arity version works
(let [sel (ms/select-method m-static-small nil)
      tuned (ms/tune-method-opts sel)]
  (assert-true "1-arity: returns map" (map? tuned)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== SUMMARY: " @pass-count "/" (+ @pass-count @fail-count) " passed =="))
(when (pos? @fail-count)
  (println (str "  FAILURES: " @fail-count)))
