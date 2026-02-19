(ns genmlx.untested-features-test
  "Tests for features identified as untested or weakly tested:
   - Custom proposal MH
   - Involutive MCMC
   - Choice gradients (with direction checks)
   - defdist-transform macro
   - Programmable VI objectives
   - Wake-sleep learning
   - Training loop"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.vi :as vi]
            [genmlx.gradients :as grad]
            [genmlx.learning :as learn])
  (:require-macros [genmlx.gen :refer [gen]]
                   [genmlx.dist.macros :refer [defdist-transform]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Untested Features Tests ===\n")

;; ---------------------------------------------------------------------------
;; 1. Custom Proposal MH
;; ---------------------------------------------------------------------------

(println "-- Custom Proposal MH: symmetric proposal --")
;; Model: mu ~ N(0,10), obs_i ~ N(mu, 1). Observe all obs at 5.0.
;; Proposal: given current choices, propose mu ~ N(current_mu, 0.5)
;; Posterior mu should be near 5.0 (with 5 observations).
(let [model (gen []
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (mx/eval! mu)
                (let [mu-val (mx/item mu)]
                  (doseq [i (range 5)]
                    (dyn/trace (keyword (str "obs" i))
                               (dist/gaussian mu-val 1)))
                  mu-val)))
      observations (reduce (fn [cm i]
                              (cm/set-choice cm [(keyword (str "obs" i))]
                                             (mx/scalar 5.0)))
                            cm/EMPTY (range 5))
      ;; Symmetric proposal: sample from N(current_mu, 0.5)
      proposal (gen [current-choices]
                 (let [current-mu (mx/realize (cm/get-choice current-choices [:mu]))]
                   (dyn/trace :mu (dist/gaussian current-mu 0.5))))
      traces (mcmc/mh-custom
               {:samples 200 :burn 100 :proposal-gf proposal}
               model [] observations)
      mu-vals (mapv (fn [t]
                       (mx/realize (cm/get-choice (:choices t) [:mu])))
                     traces)
      mu-mean (/ (reduce + mu-vals) (count mu-vals))]
  (assert-true "custom MH: 200 samples" (= 200 (count traces)))
  (assert-close "custom MH: posterior mu near 5" 5.0 mu-mean 1.0)
  (let [ar (:acceptance-rate (meta traces))]
    (assert-true "custom MH: has acceptance rate" (some? ar))
    (assert-true "custom MH: acceptance rate > 0" (> ar 0))))

;; ---------------------------------------------------------------------------
;; 2. Involutive MCMC
;; ---------------------------------------------------------------------------

(println "\n-- Involutive MCMC: swap involution --")
;; Model: x ~ N(0,1), y ~ N(x, 0.1). Observe y=2.
;; Proposal: sample aux ~ N(0, 0.5)
;; Involution: swap x with x+aux, swap aux with -aux (its own inverse)
;; This should mix and concentrate x near 2.
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (mx/eval! x)
                (dyn/trace :y (dist/gaussian (mx/item x) 0.1))
                (mx/item x)))
      observations (cm/choicemap :y (mx/scalar 2.0))
      ;; Proposal: sample auxiliary noise
      proposal (gen [current-choices]
                 (dyn/trace :aux (dist/gaussian 0 0.5)))
      ;; Involution: x' = x + aux, aux' = -aux (its own inverse)
      involution (fn [trace-cm aux-cm]
                   (let [x-val (mx/realize (cm/get-choice trace-cm [:x]))
                         aux-val (mx/realize (cm/get-choice aux-cm [:aux]))
                         new-x (+ x-val aux-val)
                         new-aux (- aux-val)]
                     [(cm/set-choice trace-cm [:x] (mx/scalar new-x))
                      (cm/set-choice aux-cm [:aux] (mx/scalar new-aux))]))
      traces (mcmc/involutive-mh
               {:samples 200 :burn 100
                :proposal-gf proposal :involution involution}
               model [] observations)
      x-vals (mapv (fn [t]
                      (mx/realize (cm/get-choice (:choices t) [:x])))
                    traces)
      x-mean (/ (reduce + x-vals) (count x-vals))]
  (assert-true "involutive MH: 200 samples" (= 200 (count traces)))
  (assert-close "involutive MH: posterior x near 2" 2.0 x-mean 0.5)
  (let [ar (:acceptance-rate (meta traces))]
    (assert-true "involutive MH: acceptance rate > 0" (> ar 0))))

;; ---------------------------------------------------------------------------
;; 3. Choice Gradients — direction and magnitude
;; ---------------------------------------------------------------------------

(println "\n-- Choice gradients: gradient direction for Gaussian --")
;; Model: x ~ N(0, 1). Generate with x=3.
;; d(log p(x))/dx = d(-x²/2 - ...)/dx = -x = -3 at x=3
(let [model (gen [mu]
              (dyn/trace :x (dist/gaussian mu 1)))
      constraints (cm/choicemap :x (mx/scalar 3.0))
      {:keys [trace]} (p/generate model [0] constraints)
      result (grad/choice-gradients model trace [:x])]
  (mx/eval! (:x result))
  (let [grad-val (mx/item (:x result))]
    (assert-true "gradient at x=3 is negative" (< grad-val 0))
    (assert-close "gradient at x=3 ≈ -3" -3.0 grad-val 0.5)))

(println "\n-- Score gradient: direction check --")
;; score-gradient: params overwrite observations at the given addresses
;; d/d(obs) log N(obs; mu=0, 1) = -(obs - mu) = -obs at obs=5 → gradient = -5
(let [model (gen [mu]
              (dyn/trace :obs (dist/gaussian mu 1)))
      result (grad/score-gradient model [0]
               (cm/choicemap :obs (mx/scalar 5.0))
               [:obs] (mx/array [5.0]))]
  (mx/eval! (:grad result) (:score result))
  (assert-true "score gradient is negative at obs=5" (< (mx/item (:grad result)) 0))
  (assert-close "score gradient ≈ -5" -5.0 (mx/item (:grad result)) 0.5)
  (assert-true "score is finite" (js/isFinite (mx/item (:score result)))))

;; ---------------------------------------------------------------------------
;; 4. defdist-transform macro
;; ---------------------------------------------------------------------------

(println "\n-- defdist-transform: log-normal via exp(gaussian) --")
(defdist-transform log-normal-t
  "Log-normal via exp transform of gaussian."
  [mu sigma]
  :base (dist/gaussian mu sigma)
  :forward mx/exp
  :inverse mx/log
  :log-det-jac (fn [v] (mx/negative (mx/log v))))

(let [d (log-normal-t 0 1)]
  ;; Sample should be positive (exp of gaussian)
  (let [samples (mapv (fn [_] (let [v (dc/dist-sample d nil)] (mx/eval! v) (mx/item v)))
                      (range 100))]
    (assert-true "log-normal-t: all samples > 0" (every? pos? samples)))
  ;; Log-prob at v=1: log N(log(1); 0, 1) + log|d(log)/dv| at v=1
  ;; = log N(0; 0, 1) + log(1/1) = -0.9189 + 0 = -0.9189
  (let [lp (dc/dist-log-prob d (mx/scalar 1.0))]
    (mx/eval! lp)
    (assert-close "log-normal-t: lp at v=1" -0.9189 (mx/item lp) 0.01))
  ;; Log-prob at v=e (i.e. log(e)=1):
  ;; log N(1; 0, 1) + log(1/e) = (-0.5 - 0.9189) + (-1) = -2.4189
  (let [lp (dc/dist-log-prob d (mx/scalar js/Math.E))]
    (mx/eval! lp)
    (let [expected (+ (- (* -0.5 1.0) 0.9189) -1.0)]
      (assert-close "log-normal-t: lp at v=e" expected (mx/item lp) 0.01))))

;; Verify the transform distribution works through GFI
(println "\n-- defdist-transform: works with generate --")
(let [d (log-normal-t 0 1)
      {:keys [trace weight]} (p/generate d [] (cm/->Value (mx/scalar 2.0)))]
  (mx/eval! weight)
  ;; weight should equal log-prob of v=2 under log-normal
  ;; log N(log(2); 0, 1) - log(2) = (-0.5*log(2)^2 - 0.9189) - log(2)
  (let [log2 (js/Math.log 2)
        expected (+ (* -0.5 log2 log2) -0.9189 (- log2))]
    (assert-close "log-normal-t via generate" expected (mx/item weight) 0.01)))

;; ---------------------------------------------------------------------------
;; 5. Programmable VI — ELBO objective
;; ---------------------------------------------------------------------------

(println "\n-- Programmable VI (ELBO): learns posterior of simple model --")
;; Model: z ~ N(0, 1), obs = z + noise. Observe obs=3.
;; True posterior: z | obs=3 ~ N(1.5, 0.5) (for equal precision)
;; We'll use p(z, obs=3) = N(z; 0, 1) * N(3; z, 1) and optimize a Gaussian guide.
(let [log-p (fn [z]
              ;; log p(z) + log p(obs=3 | z)
              (let [z-scalar (mx/index z 0)]
                (mx/add (dc/dist-log-prob (dist/gaussian 0 1) z-scalar)
                        (dc/dist-log-prob (dist/gaussian z-scalar 1) (mx/scalar 3.0)))))
      ;; Guide: N(mu, exp(log-sigma))
      log-q (fn [z params]
              (let [mu (mx/index params 0)
                    log-sigma (mx/index params 1)
                    sigma (mx/exp log-sigma)]
                (dc/dist-log-prob (dist/gaussian mu sigma) (mx/index z 0))))
      sample-fn (fn [params key n]
                  (let [mu (mx/index params 0)
                        log-sigma (mx/index params 1)
                        sigma (mx/exp log-sigma)
                        eps (rng/normal (rng/ensure-key key) [n 1])]
                    (mx/add mu (mx/multiply sigma eps))))
      init-params (mx/array [0.0 0.0])  ;; mu=0, log-sigma=0 (sigma=1)
      result (vi/programmable-vi
               {:iterations 200 :learning-rate 0.01 :n-samples 20
                :objective :elbo}
               log-p log-q sample-fn init-params)]
  (mx/eval! (:params result))
  (let [final-mu (mx/item (mx/index (:params result) 0))
        final-log-sigma (mx/item (mx/index (:params result) 1))
        final-sigma (js/Math.exp final-log-sigma)]
    ;; Posterior: N(1.5, sqrt(0.5)) — mu should be near 1.5
    (assert-close "prog VI ELBO: mu near 1.5" 1.5 final-mu 0.5)
    (assert-true "prog VI ELBO: sigma reasonable" (< final-sigma 2.0))
    ;; Loss should decrease
    (let [losses (:loss-history result)
          first-loss (first losses)
          last-loss (last losses)]
      (assert-true "prog VI ELBO: loss decreased" (< last-loss first-loss)))))

;; ---------------------------------------------------------------------------
;; 6. IWELBO objective — should be a tighter bound than ELBO
;; ---------------------------------------------------------------------------

(println "\n-- Programmable VI (IWELBO): tighter bound --")
(let [log-p (fn [z]
              (let [z-scalar (mx/index z 0)]
                (mx/add (dc/dist-log-prob (dist/gaussian 0 1) z-scalar)
                        (dc/dist-log-prob (dist/gaussian z-scalar 1) (mx/scalar 3.0)))))
      log-q (fn [z params]
              (let [mu (mx/index params 0)
                    log-sigma (mx/index params 1)
                    sigma (mx/exp log-sigma)]
                (dc/dist-log-prob (dist/gaussian mu sigma) (mx/index z 0))))
      sample-fn (fn [params key n]
                  (let [mu (mx/index params 0)
                        log-sigma (mx/index params 1)
                        sigma (mx/exp log-sigma)
                        eps (rng/normal (rng/ensure-key key) [n 1])]
                    (mx/add mu (mx/multiply sigma eps))))
      init-params (mx/array [0.0 0.0])
      result (vi/programmable-vi
               {:iterations 200 :learning-rate 0.01 :n-samples 20
                :objective :iwelbo}
               log-p log-q sample-fn init-params)]
  (mx/eval! (:params result))
  (let [final-mu (mx/item (mx/index (:params result) 0))
        losses (:loss-history result)
        first-loss (first losses)
        last-loss (last losses)]
    ;; IWELBO has higher gradient variance — just check mu moved toward 1.5
    ;; and loss history is populated
    (assert-true "prog VI IWELBO: mu moved toward 1.5 (> 0)" (> final-mu 0))
    (assert-true "prog VI IWELBO: has loss history" (pos? (count losses)))))

;; ---------------------------------------------------------------------------
;; 7. Training loop
;; ---------------------------------------------------------------------------

(println "\n-- Training loop (SGD): minimize x^2 --")
(let [loss-grad-fn (fn [params _key]
                     (let [loss (mx/sum (mx/square params))
                           grad (mx/multiply (mx/scalar 2.0) params)]
                       {:loss loss :grad grad}))
      result (learn/train
               {:iterations 100 :optimizer :sgd :lr 0.1}
               loss-grad-fn (mx/array [5.0 3.0]))]
  (mx/eval! (:params result))
  (let [final (mx/->clj (:params result))]
    (assert-true "SGD: params near 0" (every? #(< (js/Math.abs %) 0.1) final))
    (assert-true "SGD: loss decreased"
      (< (last (:loss-history result)) (first (:loss-history result))))))

(println "\n-- Training loop (Adam): minimize (x-3)^2 --")
(let [loss-grad-fn (fn [params _key]
                     (let [target (mx/array [3.0 -2.0])
                           diff (mx/subtract params target)
                           loss (mx/sum (mx/square diff))
                           grad (mx/multiply (mx/scalar 2.0) diff)]
                       {:loss loss :grad grad}))
      result (learn/train
               {:iterations 200 :optimizer :adam :lr 0.1}
               loss-grad-fn (mx/array [0.0 0.0]))]
  (mx/eval! (:params result))
  (let [final (mx/->clj (:params result))]
    (assert-close "Adam train: param 0 near 3" 3.0 (first final) 0.5)
    (assert-close "Adam train: param 1 near -2" -2.0 (second final) 0.5)))

;; ---------------------------------------------------------------------------
;; 8. Wake-sleep learning
;; ---------------------------------------------------------------------------

(println "\n-- Wake-sleep: guide learns to match model --")
;; Simple model: z ~ N(3, 0.5). No observations needed.
;; Guide: z ~ N(mu_q, 1) where mu_q is the trainable parameter.
;; Wake phase should push mu_q toward 3.
(let [model (gen []
              (let [z (dyn/trace :z (dist/gaussian 3 0.5))]
                (mx/eval! z) (mx/item z)))
      guide (gen []
              (let [z (dyn/trace :z (dist/gaussian 0 1))]
                (mx/eval! z) (mx/item z)))
      init-params (mx/array [0.0])  ;; guide starts at mu=0
      result (learn/wake-sleep
               {:iterations 30 :wake-steps 1 :sleep-steps 1 :lr 0.05}
               model guide [] cm/EMPTY [:z] init-params)]
  (mx/eval! (:params result))
  (let [final-mu (mx/item (:params result))]
    ;; Guide mu should move toward model's mean (3.0)
    ;; It won't converge perfectly in 30 iters, but should be > 0 (moved toward 3)
    (assert-true "wake-sleep: guide mu moved toward 3" (> final-mu 0.5))
    (assert-true "wake-sleep: has wake losses" (pos? (count (:wake-losses result))))
    (assert-true "wake-sleep: has sleep losses" (pos? (count (:sleep-losses result))))))

;; ---------------------------------------------------------------------------
;; 9. Param store — flatten/unflatten round-trip
;; ---------------------------------------------------------------------------

(println "\n-- Param store: flatten/unflatten round-trip --")
(let [store (learn/make-param-store {:a (mx/scalar 1.0) :b (mx/scalar 2.0) :c (mx/scalar 3.0)})
      names [:a :b :c]
      flat (learn/params->array store names)
      _ (mx/eval! flat)
      unflat (learn/array->params flat names)]
  (assert-close "flatten a" 1.0 (mx/item (mx/index flat 0)) 1e-5)
  (assert-close "flatten b" 2.0 (mx/item (mx/index flat 1)) 1e-5)
  (assert-close "flatten c" 3.0 (mx/item (mx/index flat 2)) 1e-5)
  (mx/eval! (:a unflat))
  (assert-close "unflatten a" 1.0 (mx/item (:a unflat)) 1e-5))

(println "\nAll untested features tests complete.")
