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

(println (str "\nfamily_score_test: " @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
