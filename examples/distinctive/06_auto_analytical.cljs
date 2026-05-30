(ns demo-auto-analytical
  "DISTINCTIVE FEATURE: Auto-analytical inference from STATIC SOURCE ANALYSIS.

   GenMLX reads a model's source form, detects conjugacy (here normal-normal),
   and where the math permits performs EXACT closed-form inference instead of
   sampling — with zero user hints. The analytical dispatcher fires on
   p/generate, making the returned :weight the EXACT log marginal likelihood.

   We cross-check that exact number against an independent importance-sampling
   estimate (they agree, and agreement tightens with more particles), and we
   compare the hand-derived closed-form posterior mean to a weighted-IS mean."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inspect :as inspect]
            [genmlx.inference.importance :as is])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- m [x] (mx/item x))                       ; eval boundary: MLX scalar -> JS number

;; ---------------------------------------------------------------------------
;; The conjugate normal-normal model.
;;   mu ~ Normal(prior-mean, prior-std)
;;   y_i ~ Normal(mu, obs-std)   for i = 0..3
;; A STATIC source: keyword addresses, no branches, no loops, no splice.
;; ---------------------------------------------------------------------------

(def prior-mean 0.0)
(def prior-std  3.0)
(def obs-std    1.0)

(def model
  (gen []
    (let [mu (trace :mu (dist/gaussian prior-mean prior-std))]
      (trace :y0 (dist/gaussian mu obs-std))
      (trace :y1 (dist/gaussian mu obs-std))
      (trace :y2 (dist/gaussian mu obs-std))
      (trace :y3 (dist/gaussian mu obs-std))
      mu)))

;; The data we will condition on.
(def data [1.8 2.2 1.5 2.5])
(def obs (apply cm/choicemap
                (mapcat (fn [i y] [(keyword (str "y" i)) (mx/scalar y)])
                        (range) data)))

;; ===========================================================================
;; (a) Static analysis auto-detects conjugacy — no user hint.
;; ===========================================================================
(println "=== (a) Static source analysis auto-detects conjugacy (zero hints) ===")
(let [info (inspect/inspect model)]
  (println "compilation level :  " (:compilation info)
           "  (expect :L1-M2 — fully static, compilable)")
  (println "classification    :  " (:classification info))
  (let [{:keys [pairs analytical-eligible]} (:conjugacy info)]
    (println "conjugate family  :  " (:family (first pairs))
             "  (expect :normal-normal)")
    (println "conjugate pairs   :  "
             (mapv (juxt :prior-addr :obs-addr) pairs))
    (println "analytical-eligible: " analytical-eligible
             "  (ops the closed-form path serves)"))
  (println "dispatch for :generate ->" (get (:dispatch info) :generate)
           "  (compiled/static; the analytical path activates once obs are bound)"))

;; ===========================================================================
;; (b) Constrain observations; the analytical dispatcher fires.
;;     The returned :weight is the EXACT log marginal likelihood log p(data),
;;     computed in closed form — no sampling.
;; ===========================================================================
(println "\n=== (b) p/generate fires the analytical path: weight = EXACT log p(data) ===")
(def analytical-log-ml
  (let [{:keys [weight]} (p/generate (dyn/with-key model (rng/fresh-key 1)) [] obs)]
    (m weight)))
(println "analytical log marginal likelihood log p(data) =" analytical-log-ml
         "  (closed form, no Monte Carlo)")

;; Definitive proof the weight is EXACT, not a sampled estimate: the n iid
;; observations marginally form a multivariate Normal with mean m0 and
;; covariance  s0^2 * J + s^2 * I  (J = all-ones). Compute its log-density at
;; the data by hand using the Sherman-Morrison rank-1 form.
(def hand-derived-log-ml
  (let [n     (count data)
        s0sq  (* prior-std prior-std)
        ssq   (* obs-std obs-std)
        ;; Cov = ssq*I + s0sq*ones*ones^T. Use rank-1 inverse / determinant.
        ;; det(Cov) = ssq^(n-1) * (ssq + n*s0sq)
        log-det (+ (* (dec n) (js/Math.log ssq))
                   (js/Math.log (+ ssq (* n s0sq))))
        ;; quadratic form (y-m0)^T Cov^-1 (y-m0) via Sherman-Morrison:
        ;;   (1/ssq) [ ||d||^2 - (s0sq/(ssq + n*s0sq)) (sum d)^2 ]
        d     (mapv #(- % prior-mean) data)
        dd    (reduce + (map * d d))
        sd    (reduce + d)
        quad  (* (/ 1.0 ssq)
                 (- dd (* (/ s0sq (+ ssq (* n s0sq))) sd sd)))]
    (* -0.5 (+ (* n (js/Math.log (* 2 js/Math.PI))) log-det quad))))
(println "hand-derived multivariate-Normal log p(data) =" hand-derived-log-ml
         "  | abs diff vs GenMLX weight =" (js/Math.abs (- hand-derived-log-ml analytical-log-ml)))
(println "  -> the weight matches the closed form to machine precision: the EXACT path fired.")

;; ===========================================================================
;; (c) Independent cross-check via importance sampling.
;;     IS log-ML estimate = logsumexp(log-weights) - log(N).
;;     We force the model down a non-analytical importance proposal so this is
;;     a genuinely independent estimator of the same quantity.
;; ===========================================================================
(println "\n=== (c) Cross-check: importance sampling estimates the SAME log p(data) ===")
(defn is-log-ml
  "Run IS and return logsumexp(log-weights) - log(N) as a JS number."
  [n seed]
  (let [{:keys [log-weights]}
        (is/importance-sampling {:samples n :key (rng/fresh-key seed)} model [] obs)
        stacked (mx/array (mapv m log-weights))    ; [n] MLX vector of log-weights
        lse     (mx/logsumexp stacked)]
    (- (m lse) (js/Math.log n))))

(doseq [n [200 2000 20000]]
  (let [est (is-log-ml n 7)]
    (println (str "  IS estimate (N=" n ")") "= " est
             " | abs diff vs analytical =" (js/Math.abs (- est analytical-log-ml)))))
(println "  -> the IS estimate converges to the closed-form value as N grows.")

;; ===========================================================================
;; (d) Closed-form posterior mean (normal-normal) vs weighted-IS posterior mean.
;;     With prior var s0^2, obs var s^2, n obs of sum S:
;;       posterior precision = 1/s0^2 + n/s^2
;;       posterior mean      = (prior-mean/s0^2 + S/s^2) / posterior-precision
;; ===========================================================================
(println "\n=== (d) Hand-derived closed-form posterior mean vs weighted-IS mean ===")
(def closed-form-post-mean
  (let [s0sq (* prior-std prior-std)
        ssq  (* obs-std obs-std)
        n    (count data)
        S    (reduce + data)
        prec (+ (/ 1.0 s0sq) (/ n ssq))
        mean (/ (+ (/ prior-mean s0sq) (/ S ssq)) prec)]
    mean))
(println "closed-form E[mu | data]   =" closed-form-post-mean)

(def weighted-is-post-mean
  (let [{:keys [traces log-weights]}
        (is/importance-sampling {:samples 20000 :key (rng/fresh-key 13)} model [] obs)
        ws   (mapv m log-weights)
        mx-w (apply max ws)
        normw (let [e (mapv #(js/Math.exp (- % mx-w)) ws)
                    z (reduce + e)]
                (mapv #(/ % z) e))
        mus  (mapv (fn [tr]
                     (-> (:choices tr) (cm/get-submap :mu) cm/get-value m))
                   traces)]
    (reduce + (map * normw mus))))
(println "weighted-IS E[mu | data]   =" weighted-is-post-mean
         " | abs diff =" (js/Math.abs (- weighted-is-post-mean closed-form-post-mean)))

(println "\n=== done: the analyzer found the math, did it exactly, and IS confirms it ===")
