;; @tier medium
;; genmlx-y3ls: batched-splice coverage — Map fast path, Mask, contramap /
;; map-retval delegation, Recurse's inherently-scalar verdict, the fallback
;; notice hook, and inspect eligibility.
(ns genmlx.batched-splice-test
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.handler :as h]
            [genmlx.combinators :as comb]
            [genmlx.inspect :as inspect])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn- assert-true [desc ok?]
  (if ok?
    (do (swap! pass inc) (println "  PASS:" desc))
    (do (swap! fail inc) (println "  FAIL:" desc))))
(defn- assert-close [desc expected actual tol]
  (assert-true (str desc " (" actual " ≈ " expected ")")
               (and (js/isFinite actual) (< (js/Math.abs (- expected actual)) tol))))

;; NOTE: kernels must not call mx/item in the body — batched execution contract.
(def kernel (dyn/auto-key (gen [x] (trace :y (dist/gaussian x 1)))))
(def plate (comb/map-combinator kernel))
(def xs [1.0 2.0 3.0])

(def plate-model (dyn/auto-key (gen [xs] (splice :plate plate xs))))

(def plate-obs
  (reduce (fn [acc [i v]]
            (cm/set-choice acc [:plate i] (cm/choicemap :y (mx/scalar v))))
          cm/EMPTY
          (map-indexed vector [1.3 2.4 2.6])))

(println "\n-- Map fast path: constrained vgenerate == scalar generate (deterministic) --")
(let [scalar-w (mx/item (:weight (p/generate (dyn/with-key plate-model (rng/fresh-key 1))
                                             [xs] plate-obs)))
      vt (dyn/vgenerate plate-model [xs] plate-obs 4 (rng/fresh-key 2))
      ws (mx/->clj (:weight vt))]
  (assert-true "vgenerate weight is [4]" (= 4 (count ws)))
  (doseq [[i w] (map-indexed vector ws)]
    (assert-close (str "particle " i " weight == scalar") scalar-w w 1e-4)))

(println "\n-- Map fast path: vsimulate shapes --")
(let [vt (dyn/vsimulate plate-model [xs] 8 (rng/fresh-key 3))]
  (assert-true "score shape [8]" (= [8] (mx/shape (:score vt))))
  (assert-true "leaf [:plate 0 :y] shape [8]"
               (= [8] (mx/shape (cm/get-choice (:choices vt) [:plate 0 :y]))))
  (let [rv (:retval vt)]
    (assert-true "retval is a 3-vector (one per element)" (= 3 (count rv)))
    (assert-true "each element retval is [8]-shaped"
                 (every? #(= [8] (mx/shape %)) rv))))

(println "\n-- contramap delegation (Map inner) --")
(let [cplate (comb/contramap-gf plate (fn [[xs]] [(vec (reverse xs))]))
      cmodel (dyn/auto-key (gen [xs] (splice :plate cplate xs)))
      scalar-w (mx/item (:weight (p/generate (dyn/with-key cmodel (rng/fresh-key 4))
                                             [xs] plate-obs)))
      vt (dyn/vgenerate cmodel [xs] plate-obs 4 (rng/fresh-key 5))
      ws (mx/->clj (:weight vt))]
  (assert-true "contramap(Map) eligibility :native"
               (= :native (inspect/batched-splice-eligibility cplate)))
  (doseq [[i w] (map-indexed vector ws)]
    (assert-close (str "particle " i " weight == scalar") scalar-w w 1e-4)))

(println "\n-- contramap delegation (DynamicGF inner) --")
(let [inner (dyn/auto-key (gen [x] (trace :z (dist/gaussian x 2))))
      cgf (comb/contramap-gf inner (fn [[x]] [(* 2 x)]))
      cmodel (dyn/auto-key (gen [x] (splice :c cgf x)))
      obs (cm/set-choice cm/EMPTY [:c] (cm/choicemap :z (mx/scalar 3.0)))
      scalar-w (mx/item (:weight (p/generate (dyn/with-key cmodel (rng/fresh-key 6))
                                             [1.5] obs)))
      vt (dyn/vgenerate cmodel [1.5] obs 4 (rng/fresh-key 7))
      ws (mx/->clj (:weight vt))]
  (doseq [[i w] (map-indexed vector ws)]
    (assert-close (str "particle " i " weight == scalar") scalar-w w 1e-4)))

(println "\n-- map-retval delegation: g applied to the batched retval --")
(let [inner (dyn/auto-key (gen [x] (trace :z (dist/gaussian x 1))))
      mgf (comb/map-retval inner (fn [r] (mx/add r 100)))
      mmodel (dyn/auto-key (gen [x] (splice :m mgf x)))
      obs (cm/set-choice cm/EMPTY [:m] (cm/choicemap :z (mx/scalar 0.25)))
      scalar-w (mx/item (:weight (p/generate (dyn/with-key mmodel (rng/fresh-key 8))
                                             [0.0] obs)))
      vt (dyn/vgenerate mmodel [0.0] obs 4 (rng/fresh-key 9))
      ws (mx/->clj (:weight vt))
      ;; With a scalar constraint the batched retval is legitimately []-shaped
      ;; (broadcast-deferred) — normalize before asserting the value.
      rv (let [r (mx/->clj (:retval vt))] (if (number? r) [r] r))]
  (doseq [[i w] (map-indexed vector ws)]
    (assert-close (str "particle " i " weight == scalar") scalar-w w 1e-4))
  (assert-true "retval = z + 100 (g applied on the batched path)"
               (every? #(< (js/Math.abs (- % 100.25)) 1e-4) rv)))

(println "\n-- Mask: active delegates, inactive contributes zero --")
(let [inner (dyn/auto-key (gen [x] (trace :z (dist/gaussian x 1))))
      masked (comb/mask-combinator inner)
      mmodel (dyn/auto-key (gen [flag x]
                             (let [r (splice :m masked flag x)]
                               (trace :top (dist/gaussian 0 1))
                               r)))
      obs-on (-> cm/EMPTY
                 (cm/set-choice [:m] (cm/choicemap :z (mx/scalar 0.7)))
                 (cm/set-choice [:top] (mx/scalar 0.1)))
      obs-off (cm/set-choice cm/EMPTY [:top] (mx/scalar 0.1))
      scalar-on (mx/item (:weight (p/generate (dyn/with-key mmodel (rng/fresh-key 10))
                                              [true 0.5] obs-on)))
      vt-on (dyn/vgenerate mmodel [true 0.5] obs-on 4 (rng/fresh-key 11))
      scalar-off (mx/item (:weight (p/generate (dyn/with-key mmodel (rng/fresh-key 12))
                                               [false 0.5] obs-off)))
      vt-off (dyn/vgenerate mmodel [false 0.5] obs-off 4 (rng/fresh-key 13))]
  (assert-true "mask eligibility :native"
               (= :native (inspect/batched-splice-eligibility masked)))
  (doseq [[i w] (map-indexed vector (mx/->clj (:weight vt-on)))]
    (assert-close (str "active particle " i " weight == scalar") scalar-on w 1e-4))
  (doseq [[i w] (map-indexed vector (mx/->clj (:weight vt-off)))]
    (assert-close (str "inactive particle " i " weight == scalar (top only)")
                  scalar-off w 1e-4)))

(println "\n-- fallback notice hook: fires for a non-batchable splice, names the type --")
(let [nested (comb/map-combinator plate)  ;; kernel is a MapCombinator (no :body-fn) -> falls back
      nmodel (dyn/auto-key (gen [xss] (splice :n nested xss)))
      hits (atom [])
      orig @h/fallback-notice-fn]
  (reset! h/fallback-notice-fn (fn [gf addr n] (swap! hits conj [(pr-str (type gf)) addr n])))
  (try
    (dyn/vsimulate nmodel [[[0.5 1.0] [2.0 3.0]]] 2 (rng/fresh-key 14))
    (finally (reset! h/fallback-notice-fn orig)))
  (assert-true "notice fired" (pos? (count @hits)))
  (let [[t addr n] (first @hits)]
    (assert-true "notice names MapCombinator" (boolean (re-find #"MapCombinator" t)))
    (assert-true "notice carries addr + batch size" (and (= :n addr) (= 2 n)))))

(println "\n-- inspect eligibility census --")
(assert-true "Map :native" (= :native (inspect/batched-splice-eligibility plate)))
(assert-true "DynamicGF :dynamic" (= :dynamic (inspect/batched-splice-eligibility kernel)))
(assert-true "Recurse :scalar-fallback (inherently scalar — data-dependent structure)"
             (= :scalar-fallback
                (inspect/batched-splice-eligibility
                 (comb/recurse (fn [_self] kernel)))))
(assert-true "non-GF nil" (nil? (inspect/batched-splice-eligibility 42)))
(let [r (inspect/inspect plate)]
  (assert-true "inspect on a combinator reports :batched-splice"
               (= :native (:batched-splice r))))
(let [r (inspect/inspect plate-model)]
  (assert-true "inspect on a DynamicGF includes :batched-splice"
               (= :dynamic (:batched-splice r))))

(println (str "\n== batched_splice_test: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (js/process.exit 1))
