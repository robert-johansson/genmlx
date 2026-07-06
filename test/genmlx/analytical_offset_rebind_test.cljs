;; @tier fast
(ns genmlx.analytical-offset-rebind-test
  "Regression tests for genmlx-rmy7 (Kalman offsets) and genmlx-94qc
   (affine rebinding false-positive conjugacy).

   rmy7: kalman-predict-belief dropped the transition offset (a drift chain
   z1 ~ N(z0 + 5, q) filtered as drift-0: 5-nat silent marginal error), and a
   SYMBOLIC obs offset was silently substituted with 0.0. Now: numeric/array
   offsets are handled exactly (mean' = c*m + b); symbolic coefficient or
   offset — transition or loading — bails to the handler joint path via the
   genmlx-0e0j discipline.

   94qc: analyze-affine matched the prior by NAME inside composite
   expressions, so a local REBINDING (mu = (mx/add mu 1), or (mx/exp mu))
   still classified as the raw draw and the analytical path scored the wrong
   marginal. Now: the name-based target requires live :arg-aliases provenance
   (cleared on rebinding), and :arg-deps lets derived locals under other
   names (q = (mx/exp mu)) classify nonlinear instead of constant.

   Decline correctness is pinned by exact weight equality against the
   analytical-stripped model under the SAME key — the decline path must be
   the handler path, bit for bit."
  (:require [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private fails (atom 0))
(def ^:private passes (atom 0))

(defn- assert-true [desc x]
  (if x (swap! passes inc)
        (do (swap! fails inc) (println "  FAIL" desc))))

(defn- assert-close [desc expected actual tol]
  (assert-true (str desc " (expected " expected " got " actual ")")
               (and (number? actual) (< (js/Math.abs (- expected actual)) tol))))

(defn- gen-weight+type
  "Generate under seed; return [weight score-type]."
  [model args obs seed]
  (let [r (p/generate (dyn/with-key model (rng/fresh-key seed)) args obs)]
    [(mx/item (:weight r)) (tr/score-type (:trace r))]))

(defn- declines-to-handler?
  "The analytical path must DECLINE: :joint score-type and a weight
   bit-identical to the stripped model under the same key."
  [desc model args obs seed]
  (let [[w st] (gen-weight+type model args obs seed)
        [ws _] (gen-weight+type (dyn/strip-analytical-path model) args obs seed)]
    (assert-true (str desc ": score-type :joint (got " st ")") (= :joint st))
    (assert-true (str desc ": weight == stripped-path weight (" w " vs " ws ")")
                 (< (js/Math.abs (- w ws)) 1e-6))))

;; ---------------------------------------------------------------------------
;; rmy7: drift chain — offset now handled EXACTLY (chain stays analytical)
;; ---------------------------------------------------------------------------
(println "\n-- rmy7: drift chain z1 ~ N(z0 + 5, 1) --")
(def drift-model
  (gen []
    (let [z0 (trace :z0 (dist/gaussian 0 1))
          z1 (trace :z1 (dist/gaussian (mx/add z0 5) 1))]
      (trace :y0 (dist/gaussian z0 1))
      (trace :y1 (dist/gaussian z1 1))
      nil)))
;; closed form: [y0 y1] ~ N([0 5], [[2 1] [1 3]]); log p(0, 5) = -2.64260
(let [obs (cm/from-map {:y0 (mx/scalar 0.0) :y1 (mx/scalar 5.0)})
      [w st] (gen-weight+type drift-model [] obs 3)]
  (assert-true "drift chain stays analytical (:marginal)" (= :marginal st))
  (assert-close "drift chain marginal LL exact" -2.6425960226263956 w 1e-4))
;; offset-free control unchanged: z1 ~ N(z0, 1); log p(y0=0, y1=0) with same cov
(def nodrift-model
  (gen []
    (let [z0 (trace :z0 (dist/gaussian 0 1))
          z1 (trace :z1 (dist/gaussian z0 1))]
      (trace :y0 (dist/gaussian z0 1))
      (trace :y1 (dist/gaussian z1 1))
      nil)))
;; [y0 y1] ~ N(0, [[2 1] [1 3]]); log p(0,0) = -0.5*log((2pi)^2 * det) = -2.6426
(let [obs (cm/from-map {:y0 (mx/scalar 0.0) :y1 (mx/scalar 0.0)})
      [w st] (gen-weight+type nodrift-model [] obs 3)]
  (assert-true "offset-free chain stays analytical (:marginal)" (= :marginal st))
  (assert-close "offset-free chain marginal LL exact" -2.6425960226263956 w 1e-4))

;; ---------------------------------------------------------------------------
;; rmy7: symbolic obs offset / symbolic transition coefficient — must DECLINE
;; ---------------------------------------------------------------------------
(println "\n-- rmy7: symbolic forms decline to the handler joint path --")
(def sym-obs-offset-model
  (gen [shift]
    (let [z0 (trace :z0 (dist/gaussian 0 1))
          z1 (trace :z1 (dist/gaussian z0 1))]
      (trace :y0 (dist/gaussian (mx/add (mx/multiply 2 z0) shift) 1))
      (trace :y1 (dist/gaussian z1 1))
      nil)))
(declines-to-handler? "symbolic obs offset"
                      sym-obs-offset-model [3.0]
                      (cm/from-map {:y0 (mx/scalar 3.0) :y1 (mx/scalar 0.0)}) 3)

(def sym-transition-coeff-model
  (gen [a]
    (let [z0 (trace :z0 (dist/gaussian 0 1))
          z1 (trace :z1 (dist/gaussian (mx/multiply a z0) 1))]
      (trace :y0 (dist/gaussian z0 1))
      (trace :y1 (dist/gaussian z1 1))
      nil)))
;; previously a loud NAPI type error (bare list into mx/multiply); must now
;; decline gracefully
(declines-to-handler? "symbolic transition coefficient"
                      sym-transition-coeff-model [0.9]
                      (cm/from-map {:y0 (mx/scalar 0.0) :y1 (mx/scalar 1.0)}) 3)

;; ---------------------------------------------------------------------------
;; 94qc: rebinding — must DECLINE (was scoring the wrong marginal)
;; ---------------------------------------------------------------------------
(println "\n-- 94qc: rebound locals decline --")
(def rebind-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))
          mu (mx/add mu 1)]
      (trace :y (dist/gaussian (mx/add mu 0) 1))
      nil)))
(declines-to-handler? "affine rebinding (mu = mu + 1, composite arg)"
                      rebind-model [] (cm/from-map {:y (mx/scalar 2.0)}) 3)

(def exp-rebind-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))
          mu (mx/exp mu)]
      (trace :y (dist/gaussian (mx/add mu 0) 1))
      nil)))
(declines-to-handler? "nonlinear rebinding (mu = exp mu)"
                      exp-rebind-model [] (cm/from-map {:y (mx/scalar 2.0)}) 3)

(def derived-other-name-model
  ;; the :arg-deps case — a DERIVED local under a different name must not be
  ;; treated as a constant offset (q depends on :mu nonlinearly)
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))
          q  (mx/exp mu)]
      (trace :y (dist/gaussian (mx/add mu q) 1))
      nil)))
(declines-to-handler? "derived local under another name (offset q = exp mu)"
                      derived-other-name-model [] (cm/from-map {:y (mx/scalar 2.0)}) 3)

(def lg-rebind-model
  ;; LG-block shape (two coupled latents per obs) with a rebinding — the
  ;; joint linear-Gaussian elimination path must also decline, not claim the
  ;; raw-draw design (the genmlx-94qc LG-probe concern; pinned as decline).
  (gen []
    (let [slope (trace :slope (dist/gaussian 0 1))
          icpt  (trace :icpt (dist/gaussian 0 1))
          slope (mx/add slope 1)]
      (trace :y0 (dist/gaussian (mx/add (mx/multiply slope 2.0) icpt) 1))
      (trace :y1 (dist/gaussian (mx/add (mx/multiply slope 3.0) icpt) 1))
      nil)))
(declines-to-handler? "LG-block shape with rebound latent"
                      lg-rebind-model []
                      (cm/from-map {:y0 (mx/scalar 2.0) :y1 (mx/scalar 3.0)}) 3)

;; ---------------------------------------------------------------------------
;; controls: genuine conjugacy must STAY analytical and exact
;; ---------------------------------------------------------------------------
(println "\n-- controls: genuine analytical paths intact --")
(def direct-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :y (dist/gaussian mu 1))
      nil)))
(let [[w st] (gen-weight+type direct-model [] (cm/from-map {:y (mx/scalar 2.0)}) 3)]
  (assert-true "direct N-N stays :marginal" (= :marginal st))
  (assert-close "direct N-N marginal exact (logN(2;0,2))" -2.2655121234846454 w 1e-4))

(def literal-affine-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :y (dist/gaussian (mx/add mu 1) 1))
      nil)))
(let [[w st] (gen-weight+type literal-affine-model [] (cm/from-map {:y (mx/scalar 2.0)}) 3)]
  (assert-true "literal affine offset stays :marginal" (= :marginal st))
  (assert-close "literal affine marginal exact (logN(2;1,2))" -1.5155121234846454 w 1e-4))

(println (str "\n== analytical-offset-rebind: " @passes " passed, " @fails " failed =="))
(when (pos? @fails) (set! (.-exitCode js/process) 1))
