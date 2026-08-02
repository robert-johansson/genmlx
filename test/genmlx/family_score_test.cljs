;; @tier fast
(ns family-score-test
  "Vectorized family scoring pins (genmlx-yopl): homogeneous observed site
   families in the tensor-native score are scored as ONE stacked [G]
   log-prob. These tests pin (1) equivalence — tensor score == pure handler
   score, batched tensor score == batched handler score, grads match —
   within float32 summation-reorder tolerance; (2) engagement — the family
   path actually builds on the family-shaped model; (3) declines —
   heterogeneous sites, families under the GFI fallback, and subset
   address sets fall back without changing results."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.compiled-ops :as cops]
            [genmlx.inference.util :as u]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(def passed (atom 0))
(def failed (atom 0))

(defn assert-true [desc ok]
  (if ok
    (do (swap! passed inc) (println "  PASS" desc))
    (do (swap! failed inc) (println "  FAIL" desc))))

(defn assert-rel-close [desc a b tol]
  (let [d (/ (js/Math.abs (- a b)) (max 1.0 (js/Math.abs b)))]
    (assert-true (str desc " (rel " (.toExponential d 2) ")") (< d tol))))

;; ---------------------------------------------------------------------------
;; Family-shaped static model: 2 latents + 6-obs gaussian family with
;; per-site literal means/data (the parity linreg shape).
;; ---------------------------------------------------------------------------

(def xs [0.0 1.0 2.0 3.0 4.0 5.0])
(def ys [-1.1 0.8 2.9 5.7 6.9 7.5])

(def fam-model
  (gen []
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (trace :y0 (dist/gaussian (mx/add (mx/multiply slope 0.0) intercept) 1.0))
      (trace :y1 (dist/gaussian (mx/add (mx/multiply slope 1.0) intercept) 1.0))
      (trace :y2 (dist/gaussian (mx/add (mx/multiply slope 2.0) intercept) 1.0))
      (trace :y3 (dist/gaussian (mx/add (mx/multiply slope 3.0) intercept) 1.0))
      (trace :y4 (dist/gaussian (mx/add (mx/multiply slope 4.0) intercept) 1.0))
      (trace :y5 (dist/gaussian (mx/add (mx/multiply slope 5.0) intercept) 1.0))
      slope)))

(def hmodel (dyn/strip-analytical-path fam-model))

(def obs
  (apply cm/choicemap
         (mapcat (fn [[j y]] [(keyword (str "y" j)) (mx/scalar y)])
                 (map-indexed vector ys))))

(def addresses [:slope :intercept])

(println "\n-- scalar tensor score: family engagement + handler equivalence --")
(let [{:keys [trace]} (p/generate (dyn/with-key hmodel (rng/fresh-key 7)) [] obs)
      prep (u/prepare-mcmc-score hmodel [] obs addresses trace)
      stripped (dyn/strip-alternate-paths fam-model)
      handler-score-fn (u/make-score-fn stripped [] obs addresses)
      q (:init-params prep)]
  (assert-true "tensor-native? engaged" (:tensor-native? prep))
  (mx/eval! q)
  (let [ts (mx/item ((:score-fn prep) q))
        hs (mx/item (handler-score-fn q))]
    (assert-rel-close "tensor score == handler score" ts hs 1e-5))
  ;; val-grad equivalence at the same point
  (let [[tv tg] ((mx/value-and-grad (:score-fn prep)) q)
        [hv hg] ((mx/value-and-grad handler-score-fn) q)]
    (mx/materialize! tv tg hv hg)
    (assert-rel-close "val-grad value" (mx/item tv) (mx/item hv) 1e-5)
    (let [tgs (js->clj (mx/->clj tg))
          hgs (js->clj (mx/->clj hg))]
      (doseq [[i [a b]] (map-indexed vector (map vector tgs hgs))]
        (assert-rel-close (str "grad[" i "]") a b 1e-4)))))

(println "\n-- batched tensor score: shape + GFI equivalence + grads --")
(let [bt (cops/make-batched-tensor-score-with-index
          (:schema hmodel) (:source hmodel) [] obs addresses)
      gfi (u/make-batched-score-fn hmodel [] obs addresses)
      n 5
      params (rng/normal (rng/fresh-key 42) [n 2])]
  (assert-true "batched tensor score builds" (some? bt))
  (mx/eval! params)
  (let [ts ((:score-fn bt) params)
        gs (gfi params)]
    (mx/materialize! ts gs)
    (assert-true "batched score shape [N]" (= [n] (js->clj (mx/shape ts))))
    (let [tv (js->clj (mx/->clj ts))
          gv (js->clj (mx/->clj gs))]
      (doseq [[i [a b]] (map-indexed vector (map vector tv gv))]
        (assert-rel-close (str "batched score[" i "]") a b 1e-5))))
  (let [{:keys [grad-fn tensor-native?]}
        (u/make-compiled-vectorized-score-and-grad hmodel [] obs addresses)
        gfi-grad ((u/make-vectorized-grad-score hmodel [] obs addresses) params)
        tg (grad-fn params)]
    (assert-true "vectorized score-and-grad rides tensor path" tensor-native?)
    (mx/materialize! tg gfi-grad)
    (let [tv (vec (flatten (js->clj (mx/->clj tg))))
          gv (vec (flatten (js->clj (mx/->clj gfi-grad))))]
      (doseq [[i [a b]] (map-indexed vector (map vector tv gv))]
        (assert-rel-close (str "batched grad[" i "]") a b 1e-4)))))

(println "\n-- fused chains agree across score paths (statistical) --")
(let [k (rng/fresh-key 99)
      r (mcmc/fused-mala {:samples 200 :burn 0 :thin 1 :step-size 0.08
                          :addresses addresses :key k :device :gpu}
                         hmodel [] obs)]
  (mx/eval! (:samples r))
  (assert-true "fused-mala runs on family-scored model"
               (some? (:samples r)))
  (assert-true "acceptance sane" (< 0.3 (:acceptance-rate r) 1.0)))

;; ---------------------------------------------------------------------------
;; Rung-2 emission pins (genmlx-1fbs): matmul-form affine families + stacked
;; latent priors + the lazy values-map. Both emissions are PROBE-detected, so
;; they can decline quietly — these pins therefore check ENGAGEMENT via the
;; :emission report as well as equivalence, because a silent decline would
;; otherwise pass every numeric assertion while emitting the old graph.
;; The pair is ON by default with GENMLX_AFFINE_FAMILY=0 as the kill switch,
;; so the assertions below branch on the knob and pin BOTH states.
;; ---------------------------------------------------------------------------
(def rung2-on? (not= "0" (aget (.-env js/process) "GENMLX_AFFINE_FAMILY")))

(println (str "\n-- rung-2 emission (" (if rung2-on? "ON" "off") ") --"))
(let [scalar (cops/make-tensor-score-with-index
              (:schema hmodel) (:source hmodel) [] obs)
      batched (cops/make-batched-tensor-score-with-index
               (:schema hmodel) (:source hmodel) [] obs addresses)
      es (:emission scalar)
      eb (:emission batched)]
  (assert-true "scalar path reports an emission" (map? es))
  (assert-true "batched path reports an emission" (map? eb))
  (if rung2-on?
    (do
      ;; The whole point of the rung: with both emissions live NOTHING reads
      ;; the values-map, so no latent is ever extracted from the tensor —
      ;; no mx/index forward, no scatter-add in the backward.
      (assert-true "scalar: affine family engaged" (= 1 (:affine-families es)))
      (assert-true "scalar: stacked priors engaged" (:stacked-priors? es))
      (assert-true "scalar: no per-site lp survives" (zero? (:per-site-lps es)))
      (assert-true "scalar: ZERO latents extracted" (zero? (:extracted-latents es)))
      (assert-true "batched: affine family engaged" (= 1 (:affine-families eb)))
      (assert-true "batched: stacked priors engaged" (:stacked-priors? eb))
      (assert-true "batched: no per-site lp survives" (zero? (:per-site-lps eb)))
      (assert-true "batched: ZERO latents extracted" (zero? (:extracted-latents eb))))
    (do
      (assert-true "off: no affine family" (zero? (:affine-families es)))
      (assert-true "off: no stacked priors" (not (:stacked-priors? es)))
      (assert-true "off: latents still extracted" (pos? (:extracted-latents es)))
      (assert-true "off: batched latents still extracted"
                   (pos? (:extracted-latents eb))))))

(println "\n-- stacked priors decline on latent-dependent prior args --")
;; A hierarchical prior (:x's args read the latent :mu) is NOT full-cover, so
;; the stacked emission must decline and leave the per-site path — the case
;; where dropping the values-map would silently bake a stale constant.
(let [m (gen []
          (let [mu (trace :mu (dist/gaussian 0 5))
                x  (trace :x (dist/gaussian mu 1.0))]
            (trace :o0 (dist/gaussian (mx/add (mx/multiply x 1.0) mu) 1.0))
            (trace :o1 (dist/gaussian (mx/add (mx/multiply x 2.0) mu) 1.0))
            (trace :o2 (dist/gaussian (mx/add (mx/multiply x 3.0) mu) 1.0))
            mu))
      hm (dyn/strip-analytical-path m)
      ob (cm/choicemap :o0 (mx/scalar 1.0) :o1 (mx/scalar 2.5)
                       :o2 (mx/scalar 3.5))
      built (cops/make-tensor-score-with-index
             (:schema hm) (:source hm) [] ob)
      stripped (dyn/strip-alternate-paths m)
      handler-score-fn (u/make-score-fn stripped [] ob [:mu :x])
      q (mx/array [0.7 -0.4])]
  (assert-true "hierarchical: stacked priors decline"
               (not (:stacked-priors? (:emission built))))
  (assert-true "hierarchical: values-map retained"
               (pos? (:extracted-latents (:emission built))))
  (mx/eval! q)
  (let [ts (mx/item ((:score-fn built) q))
        hs (mx/item (handler-score-fn q))]
    (assert-rel-close "hierarchical: tensor == handler score" ts hs 1e-5))
  (let [[_ tg] ((mx/value-and-grad (:score-fn built)) q)
        [_ hg] ((mx/value-and-grad handler-score-fn) q)]
    (mx/materialize! tg hg)
    (doseq [[i [a b]] (map-indexed vector
                                   (map vector (js->clj (mx/->clj tg))
                                        (js->clj (mx/->clj hg))))]
      (assert-rel-close (str "hierarchical grad[" i "]") a b 1e-4))))

(println "\n-- heterogeneous sites do NOT family-merge (correct decline) --")
;; Same expression shape but different dist types — must stay per-site.
(let [m (gen []
          (let [mu (trace :mu (dist/gaussian 0 5))]
            (trace :a (dist/gaussian mu 1.0))
            (trace :b (dist/laplace mu 1.0))
            (trace :c (dist/gaussian mu 2.0))
            (trace :d (dist/gaussian mu 3.0))
            mu))
      hm (dyn/strip-analytical-path m)
      ob (cm/choicemap :a (mx/scalar 1.0) :b (mx/scalar 2.0)
                       :c (mx/scalar 0.5) :d (mx/scalar 1.5))
      {:keys [trace]} (p/generate (dyn/with-key hm (rng/fresh-key 3)) [] ob)
      prep (u/prepare-mcmc-score hm [] ob [:mu] trace)
      stripped (dyn/strip-alternate-paths m)
      handler-score-fn (u/make-score-fn stripped [] ob [:mu])
      q (:init-params prep)]
  (mx/eval! q)
  (let [ts (mx/item ((:score-fn prep) q))
        hs (mx/item (handler-score-fn q))]
    (assert-rel-close "mixed-dist model: tensor == handler score" ts hs 1e-5)))

(println "\n-- subset addresses decline the batched tensor path --")
(let [bt (cops/make-batched-tensor-score-with-index
          (:schema hmodel) (:source hmodel) [] obs [:slope])]
  (assert-true "subset addresses -> nil (GFI fallback)" (nil? bt)))

(println "\n-- uniform family also vectorizes (allowlist coverage) --")
(let [m (gen []
          (let [w (trace :w (dist/gaussian 0 2))]
            (trace :u0 (dist/uniform (mx/multiply w 0.0) 10.0))
            (trace :u1 (dist/uniform (mx/multiply w 0.0) 11.0))
            (trace :u2 (dist/uniform (mx/multiply w 0.0) 12.0))
            w))
      hm (dyn/strip-analytical-path m)
      ob (cm/choicemap :u0 (mx/scalar 1.0) :u1 (mx/scalar 2.0)
                       :u2 (mx/scalar 3.0))
      {:keys [trace]} (p/generate (dyn/with-key hm (rng/fresh-key 5)) [] ob)
      prep (u/prepare-mcmc-score hm [] ob [:w] trace)
      stripped (dyn/strip-alternate-paths m)
      handler-score-fn (u/make-score-fn stripped [] ob [:w])
      q (:init-params prep)]
  (mx/eval! q)
  (let [ts (mx/item ((:score-fn prep) q))
        hs (mx/item (handler-score-fn q))]
    (assert-rel-close "uniform family: tensor == handler score" ts hs 1e-5)))

;; ---------------------------------------------------------------------------
;; Affine-family SOUNDNESS pins (genmlx-yy8u)
;; ---------------------------------------------------------------------------
;; The affine matmul emission replaces the family's mean with B·q + base. That
;; is only legal if the mean really IS affine in the latents. The original
;; detector decided this NUMERICALLY, from the value at q=0, at each latent
;; basis vector, and at ONE interior point with |q| <= 0.75 — but the residual
;; of alpha*q + beta*(q^2 - q) is structurally ZERO at q=0 and at every basis
;; point, so a quadratic mean was accepted and the compiled score silently
;; became a different distribution. These rows are POSITIVE oracles: they name
;; the value the detector must produce (:affine-families 0) and the score and
;; gradient it must reproduce far outside the probe ball, where MCMC lives.

(defn- assert-abs-close [desc a b tol]
  (assert-true (str desc " (|Δ| " (.toExponential (js/Math.abs (- a b)) 2) ")")
               (< (js/Math.abs (- a b)) tol)))

(defn- handler-assess-score
  "Ground truth: the pure handler's joint log-prob at a latent point."
  [model args obs addr->val]
  (let [c (reduce (fn [m [a v]] (cm/set-choice m [a] (mx/scalar v)))
                  obs addr->val)]
    (mx/item (:weight (p/assess (dyn/auto-key (dyn/strip-alternate-paths model))
                                args c)))))

(println "\n-- mild-quadratic mean: affine emission must DECLINE --")
;; mean_g = 100*slope + 0.005*slope^2 + g, sigma = 1, g in {0,1,2}.
;; The curvature is small enough that the old one-point probe measured a
;; residual of 0.00119 against its 1e-4 * 62.28 = 0.00623 threshold and
;; ACCEPTED. Measured divergence when accepted (sm_120): 945 nats at
;; slope = 10, and a gradient of -4.53 against the handler's -13.54 AT THE
;; MODE.
(def quad-model
  (gen []
    (let [slope (trace :slope (dist/gaussian 0 10))]
      (trace :y0 (dist/gaussian (mx/add (mx/multiply 100 slope)
                                        (mx/add (mx/multiply 0.005 (mx/multiply slope slope)) 0)) 1))
      (trace :y1 (dist/gaussian (mx/add (mx/multiply 100 slope)
                                        (mx/add (mx/multiply 0.005 (mx/multiply slope slope)) 1)) 1))
      (trace :y2 (dist/gaussian (mx/add (mx/multiply 100 slope)
                                        (mx/add (mx/multiply 0.005 (mx/multiply slope slope)) 2)) 1))
      slope)))

(let [ob (cm/choicemap :y0 (mx/scalar 300.0) :y1 (mx/scalar 301.0)
                       :y2 (mx/scalar 302.0))
      built (cops/make-tensor-score-with-index
             (:schema quad-model) (:source quad-model) [] ob)
      es (:emission built)]
  (assert-true "quadratic: tensor score still builds" (some? built))
  (assert-true "quadratic: affine emission DECLINES"
               (zero? (:affine-families es)))
  ;; The exact stacked family must still engage — this pins that the affine
  ;; detector declined, not the whole family machinery. Under the kill switch
  ;; (GENMLX_AFFINE_FAMILY=0) the stacked emission is disabled TOO, so there the
  ;; per-site path is the correct expectation; asserting zero unconditionally
  ;; turned the kill switch red (genmlx-yy8u re-measurement, 2026-08-02).
  (if rung2-on?
    (assert-true "quadratic: exact stacked family still engages"
                 (zero? (:per-site-lps es)))
    (assert-true "quadratic (knob off): per-site lps survive"
                 (pos? (:per-site-lps es))))
  (doseq [q [0.6 3.0 10.0]]
    (let [ts (mx/item ((:score-fn built) (mx/array [q])))
          hs (handler-assess-score quad-model [] ob [[:slope q]])]
      ;; With the defect present these differ by 0.86 / 0.003 / 945.4 nats.
      (assert-abs-close (str "quadratic: tensor == handler score @slope=" q)
                        ts hs (max 1e-2 (* 1e-6 (js/Math.abs hs))))))
  ;; Gradient at the mode — the quantity every MALA/HMC proposal is built from.
  (let [q (mx/array [3.0])
        [_ tg] ((mx/value-and-grad (:score-fn built)) q)
        eps 1e-3
        fd (/ (- (handler-assess-score quad-model [] ob [[:slope (+ 3.0 eps)]])
                 (handler-assess-score quad-model [] ob [[:slope (- 3.0 eps)]]))
              (* 2 eps))]
    (mx/materialize! tg)
    ;; Defect present: -4.53 vs -13.54.
    (assert-abs-close "quadratic: grad @mode == handler finite difference"
                      (mx/item (mx/index tg 0)) fd 0.05)))

(println "\n-- sigma-blind quadratic (|mu| ~ 1e4, sigma = 0.01) --")
;; The second unsoundness: the acceptance tolerance was scaled by the
;; magnitude of the MEAN and sigma appeared nowhere in it, so at |mu| ~ 1e4
;; the emission was allowed 0.1 of absolute mean error — 10 sigma at
;; sigma = 0.01. This one diverges 90,007 nats AT THE MODE, so a tail-only
;; row would miss it entirely.
(def sigma-blind-model
  (gen []
    (let [slope (trace :slope (dist/gaussian 0 10))]
      (trace :y0 (dist/gaussian (mx/add (mx/add 10000 (mx/multiply 100 slope))
                                        (mx/add (mx/multiply 0.5 (mx/multiply slope slope)) 0)) 0.01))
      (trace :y1 (dist/gaussian (mx/add (mx/add 10000 (mx/multiply 100 slope))
                                        (mx/add (mx/multiply 0.5 (mx/multiply slope slope)) 0.01)) 0.01))
      (trace :y2 (dist/gaussian (mx/add (mx/add 10000 (mx/multiply 100 slope))
                                        (mx/add (mx/multiply 0.5 (mx/multiply slope slope)) 2)) 0.01))
      slope)))

(let [ob (cm/choicemap :y0 (mx/scalar 10304.5) :y1 (mx/scalar 10305.5)
                       :y2 (mx/scalar 10306.5))
      built (cops/make-tensor-score-with-index
             (:schema sigma-blind-model) (:source sigma-blind-model) [] ob)
      es (:emission built)]
  (assert-true "sigma-blind: affine emission DECLINES"
               (zero? (:affine-families es)))
  (doseq [q [3.0 2.99 3.01]]
    (let [ts (mx/item ((:score-fn built) (mx/array [q])))
          hs (handler-assess-score sigma-blind-model [] ob [[:slope q]])]
      ;; Defect present: -90004.77 vs +2.697 at q = 3.
      (assert-abs-close (str "sigma-blind: tensor == handler score @slope=" q)
                        ts hs (max 1e-2 (* 1e-6 (js/Math.abs hs))))))
  (let [[_ tg] ((mx/value-and-grad (:score-fn built)) (mx/array [3.0]))
        eps 1e-3
        fd (/ (- (handler-assess-score sigma-blind-model [] ob [[:slope (+ 3.0 eps)]])
                 (handler-assess-score sigma-blind-model [] ob [[:slope (- 3.0 eps)]]))
              (* 2 eps))]
    (mx/materialize! tg)
    ;; Defect present: 6.03e6 vs 101.5.
    (assert-rel-close "sigma-blind: grad @mode == handler finite difference"
                      (mx/item (mx/index tg 0)) fd 1e-2)))

(println "\n-- affine emission ENGAGES and stays handler-equal at large |q| --")
;; The complement of the two declines: on a genuinely affine family the
;; matmul emission must still engage, and must agree with the handler far
;; outside the detector's own probe ball.
(let [built (cops/make-tensor-score-with-index
             (:schema hmodel) (:source hmodel) [] obs)
      stripped (dyn/strip-alternate-paths fam-model)
      handler-score-fn (u/make-score-fn stripped [] obs addresses)]
  (when rung2-on?
    (assert-true "linreg: affine emission engaged"
                 (= 1 (:affine-families (:emission built)))))
  (doseq [q [[10.0 -10.0] [-30.0 25.0] [0.7 -0.4]]]
    (let [qt (mx/array q)]
      (mx/eval! qt)
      (let [ts (mx/item ((:score-fn built) qt))
            hs (mx/item (handler-score-fn qt))]
        (assert-rel-close (str "linreg @" q ": tensor == handler score") ts hs 1e-5))
      (let [[_ tg] ((mx/value-and-grad (:score-fn built)) qt)
            [_ hg] ((mx/value-and-grad handler-score-fn) qt)]
        (mx/materialize! tg hg)
        (doseq [[i [a b]] (map-indexed vector
                                       (map vector (js->clj (mx/->clj tg))
                                            (js->clj (mx/->clj hg))))]
          (assert-rel-close (str "linreg @" q " grad[" i "]") a b 1e-4))))))

;; ---------------------------------------------------------------------------
;; Family-merge build safety (genmlx-spid, genmlx-1ol8)
;; ---------------------------------------------------------------------------

(println "\n-- static (mx/index xs i) regression: no crash, handler-equal --")
;; abstract-dist-args used to replace EVERY numeric literal with a family
;; placeholder regardless of argument position, so the index column of
;; (mx/index xs 0), (mx/index xs 1), ... materialized float32 — and mx/index
;; is the one gather-family op without ensure-int-indices. The family's arg
;; graph then threw "[gather] Got indices with invalid dtype" on first
;; evaluation, inside the MCMC step, killing compiled-mh / fused-MALA /
;; fused-HMC for the model instead of declining to the handler.
(def idx-model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (trace :y0 (dist/gaussian (mx/add (mx/multiply slope (mx/index xs 0)) intercept) 1.0))
      (trace :y1 (dist/gaussian (mx/add (mx/multiply slope (mx/index xs 1)) intercept) 1.0))
      (trace :y2 (dist/gaussian (mx/add (mx/multiply slope (mx/index xs 2)) intercept) 1.0))
      slope)))

(let [xarr (mx/array [0.0 1.0 2.0])
      ob (cm/choicemap :y0 (mx/scalar 0.1) :y1 (mx/scalar 2.1)
                       :y2 (mx/scalar 3.9))
      hm (dyn/strip-analytical-path idx-model)
      built (cops/make-tensor-score-with-index
             (:schema hm) (:source hm) [xarr] ob)
      q (mx/array [1.0 0.0])]
  (mx/eval! xarr q)
  (assert-true "index model is static (family detection does run)"
               (:static? (:schema idx-model)))
  (assert-true "index model: tensor score builds" (some? built))
  ;; BIT-identical, not merely close: the site graphs are the same ops in the
  ;; same order once the family declines.
  (let [ts (mx/item ((:score-fn built) q))
        hs (handler-assess-score idx-model [xarr] ob
                                 [[:slope 1.0] [:intercept 0.0]])]
    (assert-true (str "index model: tensor score BIT-identical to handler ("
                      ts " vs " hs ")")
                 (= ts hs)))
  (let [{:keys [trace]} (p/generate (dyn/with-key hm (rng/fresh-key 7)) [xarr] ob)
        prep (u/prepare-mcmc-score hm [xarr] ob addresses trace)]
    (assert-true "index model: prepare-mcmc-score returns a usable score"
                 (some? (mx/item ((:score-fn prep) (:init-params prep))))))
  (let [r (mcmc/compiled-mh {:samples 5 :addresses addresses
                             :key (rng/fresh-key 3)} idx-model [xarr] ob)]
    (assert-true "index model: compiled-mh runs to completion" (= 5 (count r)))))

(println "\n-- raw-number choicemap on a :static? true model --")
;; mx/stack is the one op with an array-ONLY NAPI signature; every other
;; observation consumer takes Either<&MxArray, f64>. So (cm/choicemap :a 1.0
;; :b 2.0) — legal everywhere else, and blessed by the choicemap docstring —
;; threw "Failed to recover MxArray type from napi value" out of the family
;; stack. The in-tree raw-number choicemaps all sit on doseq models whose
;; :static? is false, so they decline before family detection and missed it.
(def raw-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :a (dist/gaussian mu 1.0))
      (trace :b (dist/gaussian mu 1.0))
      mu)))

(let [raw-ob (cm/choicemap :a 1.0 :b 2.0)
      arr-ob (cm/choicemap :a (mx/scalar 1.0) :b (mx/scalar 2.0))
      built (cops/make-tensor-score-with-index
             (:schema raw-model) (:source raw-model) [] raw-ob)
      built-arr (cops/make-tensor-score-with-index
                 (:schema raw-model) (:source raw-model) [] arr-ob)
      q (mx/array [0.5])]
  (mx/eval! q)
  (assert-true "raw-number obs: model is :static? true (family detection runs)"
               (:static? (:schema raw-model)))
  (assert-true "raw-number obs: tensor score builds" (some? built))
  (let [ts (mx/item ((:score-fn built) q))
        as (mx/item ((:score-fn built-arr) q))
        hs (mx/item (:weight (p/assess (dyn/auto-key (dyn/strip-alternate-paths raw-model))
                                       [] (cm/set-choice raw-ob [:mu] (mx/scalar 0.5)))))]
    (assert-true (str "raw-number obs: BIT-identical to handler (" ts " vs " hs ")")
                 (= ts hs))
    (assert-true "raw-number obs: BIT-identical to the mx/scalar choicemap"
                 (= ts as)))
  ;; Stronger than "did not crash": raw numbers must not change WHICH path
  ;; runs. Without the coercion the family stack throws and the build-time
  ;; guard degrades it to per-site — correct, but a silent de-optimization
  ;; the score assertions above cannot see.
  (assert-true (str "raw-number obs: same emission as the mx/scalar choicemap "
                    (pr-str (:emission built)))
               (= (:emission built) (:emission built-arr)))
  (when rung2-on?
    (assert-true "raw-number obs: affine family engaged (not silently declined)"
                 (= 1 (:affine-families (:emission built)))))
  (let [r (mcmc/compiled-mh {:samples 5 :addresses [:mu] :key (rng/fresh-key 3)}
                            raw-model [] raw-ob)]
    (assert-true "raw-number obs: compiled-mh runs to completion" (= 5 (count r)))))

(println (str "\nfamily_score_test: " @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
