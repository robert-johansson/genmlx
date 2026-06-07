(ns genmlx.synthesis-exact-test
  "Tests for genmlx-w47t: wire LLM model-synthesis to EXACT analytical model
   evidence. Covers the two load-bearing layers, with no LLM required (models
   are assembled from dist-maps exactly as the synthesis pipeline does):

   1. REPRESENTATION — eval-model emits a faithful gen source form, so the
      schema has real keyword trace-sites and conjugacy detection fires.
   2. SCORE PATH — score-model/score-model* route conjugate models to exact
      analytical marginal evidence (single p/generate weight), falling back to
      importance sampling (clearly labeled) for non-conjugate models.

   Run: bun run --bun nbb test/genmlx/synthesis_exact_test.cljs"
  (:require [genmlx.llm.msa :as msa]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; Test harness
;; ---------------------------------------------------------------------------

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn- assert-true [msg pred]
  (if pred
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg))))

(defn- assert-close [msg expected actual tol]
  (let [ok (and (number? actual) (js/isFinite actual)
                (< (js/Math.abs (- expected actual)) tol))]
    (if ok
      (do (swap! pass-count inc)
          (println "  PASS:" msg (str "(" (.toFixed actual 4) " ~= " expected ")")))
      (do (swap! fail-count inc)
          (println "  FAIL:" msg "expected:" expected "got:" actual)))))

;; ---------------------------------------------------------------------------
;; Synthesized models (built from dist-maps, exactly as the pipeline does)
;; ---------------------------------------------------------------------------

(defn- synth
  "Assemble + eval a model from variables and a dist-map, as the synthesis
   pipeline does (assemble-gen-fn -> eval-model)."
  [variables dist-map]
  (msa/eval-model (msa/assemble-gen-fn variables dist-map)))

;; Beta-Bernoulli: p ~ Beta(2,2), obs ~ Bernoulli(p).
;; Marginal P(obs=1) = E[p] = 2/(2+2) = 0.5  ->  log 0.5 = -0.6931
(def ^:private bb-gf
  (synth [:p :obs] {:p "(dist/beta-dist 2 2)" :obs "(dist/bernoulli p)"}))
(def ^:private bb-obs {:obs 1.0})
(def ^:private bb-exact-truth (js/Math.log 0.5))

;; Normal-Normal: mu ~ N(0,3), y ~ N(mu,2).
;; Marginal y ~ N(0, sqrt(3^2 + 2^2) = sqrt 13). At y=1: log N(1;0,sqrt13)
(def ^:private nn-gf
  (synth [:mu :y] {:mu "(dist/gaussian 0 3)" :y "(dist/gaussian mu 2)"}))
(def ^:private nn-obs {:y 1.0})
(def ^:private nn-exact-truth
  (let [var 13.0 y 1.0]
    (- (* -0.5 (js/Math.log (* 2 js/Math.PI var)))
       (/ (* y y) (* 2 var)))))

;; Non-conjugate: a ~ N(0,1), b ~ N(a^2, 1) (nonlinear dependency).
(def ^:private nonconj-gf
  (synth [:a :b] {:a "(dist/gaussian 0 1)" :b "(dist/gaussian (mx/multiply a a) 1)"}))
(def ^:private nonconj-obs {:b 2.0})

;; lgamma via the membrane (test boundary only).
(defn- lgamma [x] (mx/item (mx/lgamma (mx/scalar x))))

;; Gamma-Poisson: lam ~ Gamma(shape 2, rate 1), x ~ Poisson(lam), count x=3.
;; Marginal NegBin: lgamma(a+k)-lgamma(a)-lgamma(k+1) + a*log(b/(b+1)) + k*log(1/(b+1))
(def ^:private gp-gf
  (synth [:lam :x] {:lam "(dist/gamma-dist 2 1)" :x "(dist/poisson lam)"}))
(def ^:private gp-obs {:x 3.0})
(def ^:private gp-exact-truth
  (let [a 2.0 b 1.0 k 3.0]
    (+ (- (lgamma (+ a k)) (lgamma a) (lgamma (+ k 1.0)))
       (* a (js/Math.log (/ b (+ b 1.0))))
       (* k (js/Math.log (/ 1.0 (+ b 1.0)))))))

;; Gamma-Exponential: lam ~ Gamma(2,1), x ~ Exponential(lam), x=1.5.
;; Marginal Lomax: log(a) + a*log(b) - (a+1)*log(b+x)
(def ^:private ge-gf
  (synth [:lam :x] {:lam "(dist/gamma-dist 2 1)" :x "(dist/exponential lam)"}))
(def ^:private ge-obs {:x 1.5})
(def ^:private ge-exact-truth
  (let [a 2.0 b 1.0 x 1.5]
    (- (+ (js/Math.log a) (* a (js/Math.log b)))
       (* (+ a 1.0) (js/Math.log (+ b x))))))

;; MVN-Normal (single obs): mu ~ N([0,0], 2I), y ~ N(mu, I), y=[1,1].
;; Marginal y ~ N([0,0], 2I + I = 3I): log N(y; 0, 3I).
(def ^:private mvn-gf
  (synth [:mu :y] {:mu "(dist/multivariate-normal [0 0] [[2 0] [0 2]])"
                   :y  "(dist/multivariate-normal mu [[1 0] [0 1]])"}))
(def ^:private mvn-obs {:y [1.0 1.0]})
(def ^:private mvn-exact-truth
  (let [d 2.0 log-det (* 2.0 (js/Math.log 3.0)) mahal (/ 2.0 3.0)]
    (* -0.5 (+ (* d (js/Math.log (* 2.0 js/Math.PI))) log-det mahal))))

(defn- strip-analytical
  "Return a copy of gf with the L3 analytical plan removed from its schema, so
   p/generate uses the handler/compiled path (genuine importance sampling)
   instead of the analytical shortcut. Real trace-sites are kept, so method
   selection still sees residual latents and routes to IS."
  [gf]
  (dyn/auto-key
   (update gf :schema dissoc
           :auto-handlers :auto-regenerate-transition :auto-regenerate-handlers
           :conjugate-pairs :has-conjugate? :analytical-plan)))

;; ===========================================================================
;; done-means 1 — schema has non-empty :conjugate-pairs on synthesized models
;; ===========================================================================

(println "\n== done-means 1: conjugacy fires on SCI-evaluated GFs ==")

(assert-true "bb-gf is a DynamicGF" (instance? dyn/DynamicGF bb-gf))
(assert-true "nn-gf is a DynamicGF" (instance? dyn/DynamicGF nn-gf))

(let [pairs (:conjugate-pairs (:schema bb-gf))]
  (assert-true "Beta-Bernoulli: schema has trace-sites"
               (= 2 (count (:trace-sites (:schema bb-gf)))))
  (assert-true "Beta-Bernoulli: :conjugate-pairs non-empty" (boolean (seq pairs)))
  (assert-true "Beta-Bernoulli: family is :beta-bernoulli"
               (boolean (some #(= :beta-bernoulli (:family %)) pairs))))

(let [pairs (:conjugate-pairs (:schema nn-gf))]
  (assert-true "Normal-Normal: schema has trace-sites"
               (= 2 (count (:trace-sites (:schema nn-gf)))))
  (assert-true "Normal-Normal: :conjugate-pairs non-empty" (boolean (seq pairs)))
  (assert-true "Normal-Normal: family is :normal-normal"
               (boolean (some #(= :normal-normal (:family %)) pairs))))

(let [pairs (:conjugate-pairs (:schema gp-gf))]
  (assert-true "Gamma-Poisson: :conjugate-pairs non-empty" (boolean (seq pairs)))
  (assert-true "Gamma-Poisson: family is :gamma-poisson"
               (boolean (some #(= :gamma-poisson (:family %)) pairs))))

(let [pairs (:conjugate-pairs (:schema ge-gf))]
  (assert-true "Gamma-Exponential: :conjugate-pairs non-empty" (boolean (seq pairs)))
  (assert-true "Gamma-Exponential: family is :gamma-exponential"
               (boolean (some #(= :gamma-exponential (:family %)) pairs))))

(let [pairs (:conjugate-pairs (:schema mvn-gf))]
  (assert-true "MVN-Normal: :conjugate-pairs non-empty" (boolean (seq pairs)))
  (assert-true "MVN-Normal: family is :mvn-normal"
               (boolean (some #(= :mvn-normal (:family %)) pairs))))

(let [pairs (:conjugate-pairs (:schema nonconj-gf))]
  (assert-true "Non-conjugate (a^2 dep): no conjugate pairs" (empty? pairs)))

;; ===========================================================================
;; done-means 2 — score-model* routes conjugate -> :exact, else IS (labeled)
;; ===========================================================================

(println "\n== done-means 2: routing (conjugate -> exact, else IS labeled) ==")

(let [{:keys [method log-ml]} (msa/score-model* bb-gf bb-obs)]
  (assert-true "Beta-Bernoulli routed to :exact" (= :exact method))
  (assert-true "Beta-Bernoulli exact log-ml finite" (js/isFinite log-ml)))

(let [{:keys [method log-ml]} (msa/score-model* nn-gf nn-obs)]
  (assert-true "Normal-Normal routed to :exact" (= :exact method))
  (assert-true "Normal-Normal exact log-ml finite" (js/isFinite log-ml)))

(let [{:keys [method log-ml]} (msa/score-model* gp-gf gp-obs)]
  (assert-true "Gamma-Poisson routed to :exact" (= :exact method))
  (assert-true "Gamma-Poisson exact log-ml finite" (js/isFinite log-ml)))

(let [{:keys [method log-ml]} (msa/score-model* ge-gf ge-obs)]
  (assert-true "Gamma-Exponential routed to :exact" (= :exact method))
  (assert-true "Gamma-Exponential exact log-ml finite" (js/isFinite log-ml)))

(let [{:keys [method log-ml]} (msa/score-model* mvn-gf mvn-obs)]
  (assert-true "MVN-Normal routed to :exact" (= :exact method))
  (assert-true "MVN-Normal exact log-ml finite" (js/isFinite log-ml)))

(let [{:keys [method log-ml]} (msa/score-model* nonconj-gf nonconj-obs {:n-particles 100})]
  (assert-true "Non-conjugate NOT routed to exact"
               (not (#{:exact :kalman} method)))
  (assert-true (str "Non-conjugate labeled as IS-family (got " method ")")
               (#{:handler-is :smc :hmc :vi} method))
  (assert-true "Non-conjugate IS log-ml finite" (js/isFinite log-ml)))

;; nil gf
(let [{:keys [method log-ml]} (msa/score-model* nil bb-obs)]
  (assert-true "nil gf -> :method nil" (nil? method))
  (assert-true "nil gf -> log-ml ##-Inf" (= ##-Inf log-ml)))

;; ===========================================================================
;; done-means 3 — exact agrees with IS (and with closed-form truth)
;; ===========================================================================

(println "\n== done-means 3: exact == IS == closed-form on conjugate models ==")

;; Beta-Bernoulli (low-variance IS): exact, closed-form, and IS all agree.
(let [exact (msa/score-model bb-gf bb-obs)
      is    (msa/score-model (strip-analytical bb-gf) bb-obs {:n-particles 3000})]
  (assert-close "Beta-Bernoulli exact == closed-form -0.6931" bb-exact-truth exact 0.05)
  (assert-close "Beta-Bernoulli exact == IS estimate" exact is 0.15))

;; Normal-Normal (looser IS tolerance — higher-variance estimator).
(let [exact (msa/score-model nn-gf nn-obs)
      is    (msa/score-model (strip-analytical nn-gf) nn-obs {:n-particles 6000})]
  (assert-close "Normal-Normal exact == closed-form" nn-exact-truth exact 0.05)
  (assert-close "Normal-Normal exact == IS estimate" exact is 0.5))

;; Gamma-Poisson: exact, closed-form (NegBin), and IS agree.
(let [exact (msa/score-model gp-gf gp-obs)
      is    (msa/score-model (strip-analytical gp-gf) gp-obs {:n-particles 4000})]
  (assert-close "Gamma-Poisson exact == closed-form (NegBin)" gp-exact-truth exact 0.05)
  (assert-close "Gamma-Poisson exact == IS estimate" exact is 0.2))

;; Gamma-Exponential: exact, closed-form (Lomax), and IS agree.
(let [exact (msa/score-model ge-gf ge-obs)
      is    (msa/score-model (strip-analytical ge-gf) ge-obs {:n-particles 4000})]
  (assert-close "Gamma-Exponential exact == closed-form (Lomax)" ge-exact-truth exact 0.05)
  (assert-close "Gamma-Exponential exact == IS estimate" exact is 0.2))

;; MVN-Normal (single obs): exact, closed-form, and IS agree (looser IS tol).
(let [exact (msa/score-model mvn-gf mvn-obs)
      is    (msa/score-model (strip-analytical mvn-gf) mvn-obs {:n-particles 6000})]
  (assert-close "MVN-Normal exact == closed-form" mvn-exact-truth exact 0.05)
  (assert-close "MVN-Normal exact == IS estimate" exact is 0.5))

;; ===========================================================================
;; Regression — synthesized GFs still simulate; score-model number contract
;; ===========================================================================

(println "\n== regression: simulate + score-model number contract ==")

(let [tr (p/simulate bb-gf [])]
  (assert-true "bb-gf simulates with :p" (cm/has-value? (cm/get-submap (:choices tr) :p)))
  (assert-true "bb-gf simulates with :obs" (cm/has-value? (cm/get-submap (:choices tr) :obs))))

(let [w (msa/score-model nn-gf nn-obs)]
  (assert-true "score-model returns a finite number" (and (number? w) (js/isFinite w))))

(assert-true "score-model nil -> ##-Inf" (= ##-Inf (msa/score-model nil nn-obs)))

;; code->source-form drops the (fn [trace] ...) wrapper -> ([] body...)
(let [src (msa/code->source-form
           "(fn [trace] (let [x (trace :x (dist/gaussian 0 1))] {:x x}))")]
  (assert-true "code->source-form returns a list" (seq? src))
  (assert-true "code->source-form: args vector is []" (= [] (first src)))
  (assert-true "code->source-form: non-fn string -> nil"
               (nil? (msa/code->source-form "not a fn"))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(mx/force-gc!)
(let [p @pass-count f @fail-count]
  (println (str "\n=== synthesis-exact: " p "/" (+ p f) " PASS ==="))
  (when (pos? f) (println (str "!!! " f " FAILURES !!!"))))
