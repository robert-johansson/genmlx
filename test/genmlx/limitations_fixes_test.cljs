(ns genmlx.limitations-fixes-test
  "Tests for the 6 items that previously worked with limitations.
   1.2 Custom proposal MH — now uses update weight
   2.3 Involutive MCMC — now supports Jacobian determinant
   3.3 Contramap/dimap — now has update/regenerate
   3.5 Mix combinator — now has update/regenerate
   4.3 Programmable VI — dead code removed
   4.4 Wake-sleep — auto-discovers guide addresses"
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.trace :as tr]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.combinators :as comb]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.vi :as vi]
            [genmlx.learning :as learn]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:dynamic *pass-count* 0)
(def ^:dynamic *fail-count* 0)

(defn assert-true [msg pred]
  (if pred
    (do (set! *pass-count* (inc *pass-count*))
        (println (str "  PASS: " msg)))
    (do (set! *fail-count* (inc *fail-count*))
        (println (str "  FAIL: " msg)))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (assert-true (str msg " (expected " expected ", got " actual ", diff " diff ")")
                 (< diff tol))))

(defn assert-finite [msg val]
  (assert-true (str msg " (value: " val ")")
               (and (number? val) (js/isFinite val))))

(defn- to-num
  "Extract a JS number from either an MLX array or a JS number."
  [v]
  (if (number? v) v (mx/item v)))

;; ---------------------------------------------------------------------------
;; 1.2 Custom Proposal MH — uses update weight
;; ---------------------------------------------------------------------------

(println "\n=== 1.2 Custom Proposal MH (uses update weight) ===")

(let [;; Simple model: x ~ Normal(0, 1), y ~ Normal(x, 0.5)
      model (gen [_]
              (let [x (dyn/trace :x (dist/gaussian 0.0 1.0))]
                (dyn/trace :y (dist/gaussian x 0.5))
                x))
      ;; Simple symmetric proposal: propose new x from Normal(current_x, 0.3)
      proposal (gen [current-choices]
                 (let [old-x (mx/item (cm/get-choice current-choices [:x]))]
                   (dyn/trace :x (dist/gaussian old-x 0.3))))
      obs (cm/choicemap :y 2.0)
      key (rng/fresh-key 42)
      ;; Run custom MH — should work without errors
      samples (mcmc/mh-custom
                {:samples 50 :burn 20 :proposal-gf proposal :key key}
                model [] obs)]
  (assert-true "mh-custom produces samples" (= 50 (count samples)))
  (assert-true "mh-custom samples are traces" (every? #(some? (:choices %)) samples))
  ;; Check posterior concentration: with y=2 observed, x should be near ~1.6
  (let [x-vals (mapv #(mx/item (cm/get-choice (:choices %) [:x])) samples)
        mean-x (/ (reduce + x-vals) (count x-vals))]
    (assert-true "posterior x concentrates near 1.6" (< 0.5 mean-x 3.0))))

;; Test with backward-gf (asymmetric proposal)
(let [model (gen [_]
              (dyn/trace :x (dist/gaussian 0.0 1.0)))
      ;; Asymmetric proposal: always proposes from N(0.5, 0.5)
      forward (gen [current-choices]
                (dyn/trace :x (dist/gaussian 0.5 0.5)))
      backward (gen [current-choices]
                 (dyn/trace :x (dist/gaussian 0.5 0.5)))
      key (rng/fresh-key 99)
      samples (mcmc/mh-custom
                {:samples 30 :burn 10 :proposal-gf forward :backward-gf backward :key key}
                model [] cm/EMPTY)]
  (assert-true "asymmetric custom MH produces samples" (= 30 (count samples))))

;; ---------------------------------------------------------------------------
;; 2.3 Involutive MCMC — supports Jacobian determinant
;; ---------------------------------------------------------------------------

(println "\n=== 2.3 Involutive MCMC (Jacobian support) ===")

;; Test volume-preserving involution (2-tuple return, log|det J| = 0)
(let [model (gen [_]
              (let [x (dyn/trace :x (dist/gaussian 0.0 1.0))
                    y (dyn/trace :y (dist/gaussian 0.0 1.0))]
                [x y]))
      ;; Proposal: sample auxiliary u ~ Normal(0, 0.3)
      proposal (gen [current-choices]
                 (dyn/trace :u (dist/gaussian 0.0 0.3)))
      ;; Swap involution: x' = y + u, y' = x - u, u' = -u
      involution (fn [trace-cm aux-cm]
                   (let [x (cm/get-choice trace-cm [:x])
                         y (cm/get-choice trace-cm [:y])
                         u (cm/get-choice aux-cm [:u])
                         new-x (mx/add y u)
                         new-y (mx/subtract x u)]
                     [(cm/choicemap :x new-x :y new-y)
                      (cm/choicemap :u (mx/negative u))]))
      key (rng/fresh-key 123)
      samples (mcmc/involutive-mh
                {:samples 20 :burn 5 :proposal-gf proposal :involution involution :key key}
                model [] cm/EMPTY)]
  (assert-true "volume-preserving involutive MH works" (= 20 (count samples))))

;; Test non-volume-preserving involution (3-tuple return with log|det J|)
(let [model (gen [_]
              (dyn/trace :x (dist/gaussian 0.0 2.0)))
      ;; Proposal: sample scale factor
      proposal (gen [current-choices]
                 (dyn/trace :s (dist/gaussian 0.0 0.3)))
      ;; Scaling involution: x' = x * exp(s), s' = -s
      ;; Jacobian: dx'/dx = exp(s), so log|det J| = s
      involution (fn [trace-cm aux-cm]
                   (let [x (cm/get-choice trace-cm [:x])
                         s (cm/get-choice aux-cm [:s])
                         new-x (mx/multiply x (mx/exp s))
                         log-det-J s]  ;; log|det J| = s
                     [(cm/choicemap :x new-x)
                      (cm/choicemap :s (mx/negative s))
                      log-det-J]))
      key (rng/fresh-key 456)
      samples (mcmc/involutive-mh
                {:samples 30 :burn 10 :proposal-gf proposal :involution involution :key key}
                model [] cm/EMPTY)]
  (assert-true "non-volume-preserving involutive MH works" (= 30 (count samples)))
  (let [x-vals (mapv #(mx/item (cm/get-choice (:choices %) [:x])) samples)
        variance (let [m (/ (reduce + x-vals) (count x-vals))]
                   (/ (reduce + (map #(* (- % m) (- % m)) x-vals)) (count x-vals)))]
    (assert-true "samples have reasonable variance" (> variance 0.01))))

;; ---------------------------------------------------------------------------
;; 3.3 Contramap/Dimap — update/regenerate
;; ---------------------------------------------------------------------------

(println "\n=== 3.3 Contramap/Dimap (update/regenerate) ===")

(let [;; Inner model: x ~ Normal(mean, 1)
      inner-model (gen [mean]
                    (dyn/trace :x (dist/gaussian mean 1.0)))
      ;; Contramap: transforms [offset] -> [offset + 5] before passing to inner
      contra-model (comb/contramap-gf inner-model (fn [args] [(+ (first args) 5.0)]))
      ;; Simulate
      trace (p/simulate contra-model [3.0])
      _ (assert-true "contramap simulate works" (some? (:choices trace)))
      _ (assert-true "contramap args preserved" (= [3.0] (:args trace)))
      ;; Update with constraint
      constraint (cm/choicemap :x 10.0)
      {:keys [trace weight]} (p/update contra-model trace constraint)]
  (assert-true "contramap update works" (some? trace))
  (assert-finite "contramap update weight" (mx/item weight))
  (assert-close "contramap update value" 10.0 (to-num (cm/get-choice (:choices trace) [:x])) 0.001)
  ;; Score should be Normal(10; 8, 1) since mean = 3 + 5 = 8
  (let [expected-score (- (/ (* -1 (- 10.0 8.0) (- 10.0 8.0)) 2.0)
                          (* 0.5 (js/Math.log (* 2 js/Math.PI))))]
    (assert-close "contramap score correct" expected-score (mx/item (:score trace)) 0.01))
  ;; Regenerate
  (let [{:keys [trace weight]} (p/regenerate contra-model trace sel/all)]
    (assert-true "contramap regenerate works" (some? trace))
    (assert-finite "contramap regenerate weight" (mx/item weight))))

(let [;; MapRetval: transform return value
      inner-model (gen [_]
                    (dyn/trace :x (dist/gaussian 0.0 1.0)))
      retval-model (comb/map-retval inner-model #(* % 10))
      trace (p/simulate retval-model [])
      _ (assert-true "map-retval simulate works" (some? trace))
      ;; Update
      constraint (cm/choicemap :x 2.0)
      {:keys [trace weight]} (p/update retval-model trace constraint)]
  (assert-true "map-retval update works" (some? trace))
  (assert-close "map-retval update value" 2.0 (to-num (cm/get-choice (:choices trace) [:x])) 0.001)
  ;; Regenerate
  (let [{:keys [trace weight]} (p/regenerate retval-model trace sel/all)]
    (assert-true "map-retval regenerate works" (some? trace))
    (assert-finite "map-retval regenerate weight" (mx/item weight))))

;; Test dimap (both contramap + map-retval)
(let [inner-model (gen [x]
                    (dyn/trace :z (dist/gaussian x 1.0)))
      dimap-model (comb/dimap inner-model
                              (fn [args] [(* (first args) 2.0)])
                              (fn [retval] (+ retval 100)))
      trace (p/simulate dimap-model [3.0])
      _ (assert-true "dimap simulate works" (some? trace))
      {:keys [trace weight]} (p/update dimap-model trace (cm/choicemap :z 7.0))]
  (assert-true "dimap update works" (some? trace))
  (assert-close "dimap update value" 7.0 (to-num (cm/get-choice (:choices trace) [:z])) 0.001)
  (assert-close "dimap retval transformed" 107.0 (to-num (:retval trace)) 0.001))

;; ---------------------------------------------------------------------------
;; 3.5 Mix Combinator — update/regenerate
;; ---------------------------------------------------------------------------

(println "\n=== 3.5 Mix Combinator (update/regenerate) ===")

(let [;; Two component GFs
      comp-a (gen [_]
               (dyn/trace :val (dist/gaussian -5.0 1.0)))
      comp-b (gen [_]
               (dyn/trace :val (dist/gaussian 5.0 1.0)))
      log-w (mx/array [0.0 0.0])  ;; equal weights
      mix-model (comb/mix-combinator [comp-a comp-b] log-w)
      ;; Generate with component 0 constrained
      {:keys [trace]} (p/generate mix-model [] (cm/choicemap :component-idx (mx/scalar 0 mx/int32)
                                                             :val -4.0))
      old-idx (to-num (cm/get-choice (:choices trace) [:component-idx]))
      _ (assert-true "mix generate sets component" (= 0.0 old-idx))
      ;; Update within same component
      {:keys [trace weight]} (p/update mix-model trace (cm/choicemap :val (mx/scalar -3.0)))]
  (assert-true "mix update same component works" (some? trace))
  (assert-finite "mix update weight" (mx/item weight))
  (assert-close "mix update value" -3.0 (to-num (cm/get-choice (:choices trace) [:val])) 0.001)
  (assert-close "mix component unchanged" 0.0
                (to-num (cm/get-choice (:choices trace) [:component-idx])) 0.001)
  ;; Update: switch component
  (let [{:keys [trace weight]} (p/update mix-model trace
                                         (cm/choicemap :component-idx (mx/scalar 1 mx/int32)
                                                       :val (mx/scalar 4.0)))]
    (assert-true "mix component switch works" (some? trace))
    (assert-close "mix new component idx" 1.0
                  (to-num (cm/get-choice (:choices trace) [:component-idx])) 0.001)
    (assert-close "mix new value" 4.0 (to-num (cm/get-choice (:choices trace) [:val])) 0.001)
    (assert-finite "mix switch weight" (mx/item weight))
    ;; Regenerate within same component (component-idx NOT selected)
    (let [{:keys [trace weight]} (p/regenerate mix-model trace (sel/select :val))]
      (assert-true "mix regenerate within component works" (some? trace))
      (assert-close "mix regenerate keeps component" 1.0
                    (to-num (cm/get-choice (:choices trace) [:component-idx])) 0.001)
      (assert-finite "mix regenerate weight" (mx/item weight)))
    ;; Regenerate with component-idx selected
    (let [{:keys [trace weight]} (p/regenerate mix-model trace sel/all)]
      (assert-true "mix regenerate all works" (some? trace))
      (assert-finite "mix regenerate all weight" (mx/item weight)))))

;; ---------------------------------------------------------------------------
;; 4.3 Programmable VI — dead code removed
;; ---------------------------------------------------------------------------

(println "\n=== 4.3 Programmable VI (dead code removed) ===")

;; Verify programmable-vi still works after removing obj-builder
(let [;; Simple 1D target: N(3, 1)
      log-p (fn [z] (let [z-val (mx/index z 0)]
                       (mx/multiply (mx/scalar -0.5) (mx/square (mx/subtract z-val (mx/scalar 3.0))))))
      log-q (fn [z params]
              (let [mu (mx/index params 0)
                    log-sigma (mx/index params 1)
                    sigma (mx/exp log-sigma)
                    z-val (mx/index z 0)]
                (mx/multiply (mx/scalar -0.5)
                             (mx/add (mx/scalar (js/Math.log (* 2 js/Math.PI)))
                                     (mx/multiply (mx/scalar 2.0) log-sigma)
                                     (mx/square (mx/divide (mx/subtract z-val mu) sigma))))))
      sample-fn (fn [params key n]
                  (let [mu (mx/index params 0)
                        log-sigma (mx/index params 1)
                        sigma (mx/exp log-sigma)
                        eps (rng/normal (rng/ensure-key key) [n 1])]
                    (mx/add mu (mx/multiply sigma eps))))
      init-params (mx/array [0.0 0.0])
      result (vi/programmable-vi
               {:iterations 100 :learning-rate 0.05 :n-samples 5
                :objective :elbo :key (rng/fresh-key 42)}
               log-p log-q sample-fn init-params)]
  (assert-true "programmable-vi returns params" (some? (:params result)))
  (assert-true "programmable-vi returns loss history" (= 100 (count (:loss-history result))))
  ;; mu should be near 3
  (let [final-mu (mx/item (mx/index (:params result) 0))]
    (assert-true "programmable-vi mu converges toward target" (< 1.0 final-mu 5.0))))

;; Test IWELBO objective still works
(let [log-p (fn [z] (mx/multiply (mx/scalar -0.5) (mx/square (mx/index z 0))))
      log-q (fn [z params]
              (let [mu (mx/index params 0)]
                (mx/multiply (mx/scalar -0.5) (mx/square (mx/subtract (mx/index z 0) mu)))))
      sample-fn (fn [params key n]
                  (let [mu (mx/index params 0)
                        eps (rng/normal (rng/ensure-key key) [n 1])]
                    (mx/add mu eps)))
      result (vi/programmable-vi
               {:iterations 50 :learning-rate 0.05 :n-samples 10
                :objective :iwelbo :key (rng/fresh-key 99)}
               log-p log-q sample-fn (mx/array [2.0]))]
  (assert-true "IWELBO still works after dead code removal" (= 50 (count (:loss-history result)))))

;; ---------------------------------------------------------------------------
;; 4.4 Wake-Sleep — auto-discovers guide addresses
;; ---------------------------------------------------------------------------

(println "\n=== 4.4 Wake-Sleep (auto-discover guide addresses) ===")

(let [;; Simple model
      model (gen [_]
              (let [x (dyn/trace :x (dist/gaussian 0.0 1.0))]
                (dyn/trace :y (dist/gaussian x 0.5))))
      ;; Guide with two addresses
      guide (gen [_]
              (dyn/trace :x (dist/gaussian 0.0 1.0)))
      obs (cm/choicemap :y 2.0)
      ;; Test with explicit guide-addresses (existing behavior)
      result-explicit (learn/wake-sleep
                        {:iterations 5 :lr 0.01 :key (rng/fresh-key 42)}
                        model guide [] obs [:x] (mx/array [0.0]))]
  (assert-true "wake-sleep with explicit addresses works" (some? (:params result-explicit)))
  (assert-true "wake-sleep returns wake losses" (= 5 (count (:wake-losses result-explicit))))
  (assert-true "wake-sleep returns sleep losses" (= 5 (count (:sleep-losses result-explicit))))
  ;; Test with auto-discovery (guide-addresses = nil)
  (let [result-auto (learn/wake-sleep
                      {:iterations 5 :lr 0.01 :key (rng/fresh-key 42)}
                      model guide [] obs nil nil)]
    (assert-true "wake-sleep auto-discovery works" (some? (:params result-auto)))
    (assert-true "auto-discovered wake losses" (= 5 (count (:wake-losses result-auto))))
    (assert-true "auto-discovered sleep losses" (= 5 (count (:sleep-losses result-auto))))))

;; Test auto-discovery with multi-address guide
(let [model (gen [_]
              (let [a (dyn/trace :a (dist/gaussian 0.0 1.0))
                    b (dyn/trace :b (dist/gaussian 0.0 1.0))]
                (dyn/trace :obs (dist/gaussian (mx/add a b) 0.5))))
      guide (gen [_]
              (dyn/trace :a (dist/gaussian 0.0 1.0))
              (dyn/trace :b (dist/gaussian 0.0 1.0)))
      obs (cm/choicemap :obs 3.0)
      result (learn/wake-sleep
               {:iterations 3 :lr 0.01 :key (rng/fresh-key 77)}
               model guide [] obs nil nil)]
  (assert-true "multi-addr auto-discovery works" (some? (:params result)))
  ;; Auto-discovered params should have 2 elements (for :a and :b)
  (assert-true "auto-discovered correct number of params"
               (= 2 (first (mx/shape (:params result))))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Results: " *pass-count* " passed, " *fail-count* " failed ==="))
(when (pos? *fail-count*)
  (println "THERE WERE FAILURES"))
