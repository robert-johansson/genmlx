(ns genmlx.limitations-fixes-test
  "Tests for the 6 items that previously worked with limitations."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
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

(defn- to-num
  "Extract a JS number from either an MLX array or a JS number."
  [v]
  (if (number? v) v (mx/item v)))

;; ---------------------------------------------------------------------------
;; 1.2 Custom Proposal MH -- uses update weight
;; ---------------------------------------------------------------------------

(deftest custom-proposal-mh
  (testing "mh-custom with symmetric proposal"
    (let [model (gen [_]
                  (let [x (trace :x (dist/gaussian 0.0 1.0))]
                    (trace :y (dist/gaussian x 0.5))
                    x))
          proposal (gen [current-choices]
                     (let [old-x (mx/item (cm/get-choice current-choices [:x]))]
                       (trace :x (dist/gaussian old-x 0.3))))
          obs (cm/choicemap :y 2.0)
          key (rng/fresh-key 42)
          samples (mcmc/mh-custom
                    {:samples 50 :burn 20 :proposal-gf proposal :key key}
                    model [] obs)]
      (is (= 50 (count samples)) "mh-custom produces samples")
      (is (every? #(some? (:choices %)) samples) "mh-custom samples are traces")
      (let [x-vals (mapv #(mx/item (cm/get-choice (:choices %) [:x])) samples)
            mean-x (/ (reduce + x-vals) (count x-vals))]
        (is (< 0.5 mean-x 3.0) "posterior x concentrates near 1.6"))))

  (testing "mh-custom with backward-gf (asymmetric proposal)"
    (let [model (gen [_]
                  (trace :x (dist/gaussian 0.0 1.0)))
          forward (gen [current-choices]
                    (trace :x (dist/gaussian 0.5 0.5)))
          backward (gen [current-choices]
                     (trace :x (dist/gaussian 0.5 0.5)))
          key (rng/fresh-key 99)
          samples (mcmc/mh-custom
                    {:samples 30 :burn 10 :proposal-gf forward :backward-gf backward :key key}
                    model [] cm/EMPTY)]
      (is (= 30 (count samples)) "asymmetric custom MH produces samples"))))

;; ---------------------------------------------------------------------------
;; 2.3 Involutive MCMC -- supports Jacobian determinant
;; ---------------------------------------------------------------------------

(deftest involutive-mcmc
  (testing "volume-preserving involution"
    (let [model (gen [_]
                  (let [x (trace :x (dist/gaussian 0.0 1.0))
                        y (trace :y (dist/gaussian 0.0 1.0))]
                    [x y]))
          proposal (gen [current-choices]
                     (trace :u (dist/gaussian 0.0 0.3)))
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
      (is (= 20 (count samples)) "volume-preserving involutive MH works")))

  (testing "non-volume-preserving involution (3-tuple with log|det J|)"
    (let [model (gen [_]
                  (trace :x (dist/gaussian 0.0 2.0)))
          proposal (gen [current-choices]
                     (trace :s (dist/gaussian 0.0 0.3)))
          involution (fn [trace-cm aux-cm]
                       (let [x (cm/get-choice trace-cm [:x])
                             s (cm/get-choice aux-cm [:s])
                             new-x (mx/multiply x (mx/exp s))
                             log-det-J s]
                         [(cm/choicemap :x new-x)
                          (cm/choicemap :s (mx/negative s))
                          log-det-J]))
          key (rng/fresh-key 456)
          samples (mcmc/involutive-mh
                    {:samples 30 :burn 10 :proposal-gf proposal :involution involution :key key}
                    model [] cm/EMPTY)]
      (is (= 30 (count samples)) "non-volume-preserving involutive MH works")
      (let [x-vals (mapv #(mx/item (cm/get-choice (:choices %) [:x])) samples)
            variance (let [m (/ (reduce + x-vals) (count x-vals))]
                       (/ (reduce + (map #(* (- % m) (- % m)) x-vals)) (count x-vals)))]
        (is (> variance 0.01) "samples have reasonable variance")))))

;; ---------------------------------------------------------------------------
;; 3.3 Contramap/Dimap -- update/regenerate
;; ---------------------------------------------------------------------------

(deftest contramap-update-regenerate
  (testing "contramap update and regenerate"
    (let [inner-model (gen [mean]
                        (trace :x (dist/gaussian mean 1.0)))
          contra-model (comb/contramap-gf (dyn/auto-key inner-model) (fn [args] [(+ (first args) 5.0)]))
          trace (p/simulate contra-model [3.0])]
      (is (some? (:choices trace)) "contramap simulate works")
      (is (= [3.0] (:args trace)) "contramap args preserved")
      (let [constraint (cm/choicemap :x 10.0)
            {:keys [trace weight]} (p/update contra-model trace constraint)]
        (is (some? trace) "contramap update works")
        (is (and (number? (mx/item weight)) (js/isFinite (mx/item weight)))
            "contramap update weight finite")
        (is (h/close? 10.0 (to-num (cm/get-choice (:choices trace) [:x])) 0.001)
            "contramap update value")
        (let [expected-score (- (/ (* -1 (- 10.0 8.0) (- 10.0 8.0)) 2.0)
                                (* 0.5 (js/Math.log (* 2 js/Math.PI))))]
          (is (h/close? expected-score (mx/item (:score trace)) 0.01)
              "contramap score correct"))
        (let [{:keys [trace weight]} (p/regenerate contra-model trace sel/all)]
          (is (some? trace) "contramap regenerate works")
          (is (and (number? (mx/item weight)) (js/isFinite (mx/item weight)))
              "contramap regenerate weight finite"))))))

(deftest map-retval-update-regenerate
  (testing "map-retval update and regenerate"
    (let [inner-model (gen [_]
                        (trace :x (dist/gaussian 0.0 1.0)))
          retval-model (comb/map-retval (dyn/auto-key inner-model) #(* % 10))
          trace (p/simulate retval-model [])]
      (is (some? trace) "map-retval simulate works")
      (let [constraint (cm/choicemap :x 2.0)
            {:keys [trace weight]} (p/update retval-model trace constraint)]
        (is (some? trace) "map-retval update works")
        (is (h/close? 2.0 (to-num (cm/get-choice (:choices trace) [:x])) 0.001)
            "map-retval update value")
        (let [{:keys [trace weight]} (p/regenerate retval-model trace sel/all)]
          (is (some? trace) "map-retval regenerate works")
          (is (and (number? (mx/item weight)) (js/isFinite (mx/item weight)))
              "map-retval regenerate weight finite"))))))

(deftest dimap-update
  (testing "dimap update"
    (let [inner-model (gen [x]
                        (trace :z (dist/gaussian x 1.0)))
          dimap-model (comb/dimap (dyn/auto-key inner-model)
                                  (fn [args] [(* (first args) 2.0)])
                                  (fn [retval] (+ retval 100)))
          trace (p/simulate dimap-model [3.0])]
      (is (some? trace) "dimap simulate works")
      (let [{:keys [trace weight]} (p/update dimap-model trace (cm/choicemap :z 7.0))]
        (is (some? trace) "dimap update works")
        (is (h/close? 7.0 (to-num (cm/get-choice (:choices trace) [:z])) 0.001)
            "dimap update value")
        (is (h/close? 107.0 (to-num (:retval trace)) 0.001)
            "dimap retval transformed")))))

;; ---------------------------------------------------------------------------
;; 3.5 Mix Combinator -- update/regenerate
;; ---------------------------------------------------------------------------

(deftest mix-combinator-update-regenerate
  (testing "mix combinator update within same component"
    (let [comp-a (gen [_]
                   (trace :val (dist/gaussian -5.0 1.0)))
          comp-b (gen [_]
                   (trace :val (dist/gaussian 5.0 1.0)))
          log-w (mx/array [0.0 0.0])
          mix-model (comb/mix-combinator [(dyn/auto-key comp-a) (dyn/auto-key comp-b)] log-w)
          {:keys [trace]} (p/generate mix-model [] (cm/choicemap :component-idx (mx/scalar 0 mx/int32)
                                                                 :val -4.0))
          old-idx (to-num (cm/get-choice (:choices trace) [:component-idx]))]
      (is (= 0.0 old-idx) "mix generate sets component")
      (let [{:keys [trace weight]} (p/update mix-model trace (cm/choicemap :val (mx/scalar -3.0)))]
        (is (some? trace) "mix update same component works")
        (is (and (number? (mx/item weight)) (js/isFinite (mx/item weight)))
            "mix update weight finite")
        (is (h/close? -3.0 (to-num (cm/get-choice (:choices trace) [:val])) 0.001)
            "mix update value")
        (is (h/close? 0.0 (to-num (cm/get-choice (:choices trace) [:component-idx])) 0.001)
            "mix component unchanged"))))

  (testing "mix combinator switch component"
    (let [comp-a (gen [_]
                   (trace :val (dist/gaussian -5.0 1.0)))
          comp-b (gen [_]
                   (trace :val (dist/gaussian 5.0 1.0)))
          log-w (mx/array [0.0 0.0])
          mix-model (comb/mix-combinator [(dyn/auto-key comp-a) (dyn/auto-key comp-b)] log-w)
          {:keys [trace]} (p/generate mix-model [] (cm/choicemap :component-idx (mx/scalar 0 mx/int32)
                                                                 :val -4.0))
          {:keys [trace weight]} (p/update mix-model trace
                                           (cm/choicemap :component-idx (mx/scalar 1 mx/int32)
                                                         :val (mx/scalar 4.0)))]
      (is (some? trace) "mix component switch works")
      (is (h/close? 1.0 (to-num (cm/get-choice (:choices trace) [:component-idx])) 0.001)
          "mix new component idx")
      (is (h/close? 4.0 (to-num (cm/get-choice (:choices trace) [:val])) 0.001)
          "mix new value")
      (is (and (number? (mx/item weight)) (js/isFinite (mx/item weight)))
          "mix switch weight finite")
      ;; Regenerate within same component
      (let [{:keys [trace weight]} (p/regenerate mix-model trace (sel/select :val))]
        (is (some? trace) "mix regenerate within component works")
        (is (h/close? 1.0 (to-num (cm/get-choice (:choices trace) [:component-idx])) 0.001)
            "mix regenerate keeps component")
        (is (and (number? (mx/item weight)) (js/isFinite (mx/item weight)))
            "mix regenerate weight finite"))
      ;; Regenerate all
      (let [{:keys [trace weight]} (p/regenerate mix-model trace sel/all)]
        (is (some? trace) "mix regenerate all works")
        (is (and (number? (mx/item weight)) (js/isFinite (mx/item weight)))
            "mix regenerate all weight finite")))))

;; ---------------------------------------------------------------------------
;; 4.3 Programmable VI -- dead code removed
;; ---------------------------------------------------------------------------

(deftest programmable-vi
  (testing "programmable-vi ELBO"
    (let [log-p (fn [z] (let [z-val (mx/index z 0)]
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
      (is (some? (:params result)) "programmable-vi returns params")
      (is (= 100 (count (:loss-history result))) "programmable-vi returns loss history")
      (let [final-mu (mx/item (mx/index (:params result) 0))]
        (is (< 1.0 final-mu 5.0) "programmable-vi mu converges toward target"))))

  (testing "programmable-vi IWELBO"
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
      (is (= 50 (count (:loss-history result))) "IWELBO still works after dead code removal"))))

;; ---------------------------------------------------------------------------
;; 4.4 Wake-Sleep -- auto-discovers guide addresses
;; ---------------------------------------------------------------------------

(deftest wake-sleep-explicit
  (testing "wake-sleep with explicit addresses"
    (let [model (gen [_]
                  (let [x (trace :x (dist/gaussian 0.0 1.0))]
                    (trace :y (dist/gaussian x 0.5))))
          guide (gen [_]
                  (trace :x (dist/gaussian 0.0 1.0)))
          obs (cm/choicemap :y 2.0)
          result-explicit (learn/wake-sleep
                            {:iterations 5 :lr 0.01 :key (rng/fresh-key 42)}
                            model guide [] obs [:x] (mx/array [0.0]))]
      (is (some? (:params result-explicit)) "wake-sleep with explicit addresses works")
      (is (= 5 (count (:wake-losses result-explicit))) "wake-sleep returns wake losses")
      (is (= 5 (count (:sleep-losses result-explicit))) "wake-sleep returns sleep losses"))))

(deftest wake-sleep-auto-discovery
  (testing "wake-sleep auto-discovery"
    (let [model (gen [_]
                  (let [x (trace :x (dist/gaussian 0.0 1.0))]
                    (trace :y (dist/gaussian x 0.5))))
          guide (gen [_]
                  (trace :x (dist/gaussian 0.0 1.0)))
          obs (cm/choicemap :y 2.0)
          result-auto (learn/wake-sleep
                        {:iterations 5 :lr 0.01 :key (rng/fresh-key 42)}
                        model guide [] obs nil nil)]
      (is (some? (:params result-auto)) "wake-sleep auto-discovery works")
      (is (= 5 (count (:wake-losses result-auto))) "auto-discovered wake losses")
      (is (= 5 (count (:sleep-losses result-auto))) "auto-discovered sleep losses"))))

(deftest wake-sleep-multi-addr-auto-discovery
  (testing "multi-addr auto-discovery"
    (let [model (gen [_]
                  (let [a (trace :a (dist/gaussian 0.0 1.0))
                        b (trace :b (dist/gaussian 0.0 1.0))]
                    (trace :obs (dist/gaussian (mx/add a b) 0.5))))
          guide (gen [_]
                  (trace :a (dist/gaussian 0.0 1.0))
                  (trace :b (dist/gaussian 0.0 1.0)))
          obs (cm/choicemap :obs 3.0)
          result (learn/wake-sleep
                   {:iterations 3 :lr 0.01 :key (rng/fresh-key 77)}
                   model guide [] obs nil nil)]
      (is (some? (:params result)) "multi-addr auto-discovery works")
      (is (= 2 (first (mx/shape (:params result))))
          "auto-discovered correct number of params"))))

(cljs.test/run-tests)
