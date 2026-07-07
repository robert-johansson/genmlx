;; @tier fast
(ns genmlx.vcone-regen-test
  "Batched cone-restricted regenerate + vmh (genmlx-js93).

   The [N]-lane generalization of the scalar cone (genmlx-ltx2): for a
   single-site selection on a flat static model, the cone is
   address-determined — per-MODEL, not per-particle — so the static? gate IS
   the lane-uniformity gate and all N chains recompute the same
   {s} ∪ direct-children(s) as one broadcast.

   Contract under test:
     B1 vregenerate via the vcone path is BIT-IDENTICAL to the full-body
        batched handler path under the same key: sampling goes through the
        same dc/dist-sample-n on the same Distribution with the handler's
        exact key discipline (retained sites never split; one split at s)
     B2 declines (multi-site, empty selection, missing constructor) fall
        through to the batched handler unchanged
     B3 vmh sweeps N independent chains to the correct stationary
        distribution (unconstrained model → the prior), and the incremental
        per-lane scores stay consistent with a fresh full re-score"
  (:require [genmlx.dist :as dist]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.protocols :as p])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn- assert-true [label v]
  (if v (swap! pass inc)
        (do (swap! fail inc) (println "  FAIL" label))))

(defn- assert-close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (< d tol) (swap! pass inc)
        (do (swap! fail inc)
            (println "  FAIL" label "expected" expected "actual" actual "d" d)))))

(defn- arrays-equal? [a b]
  (pos? (mx/item (mx/all (mx/equal a b)))))

(defn- max-abs-diff [a b]
  (mx/item (mx/amax (mx/abs (mx/subtract a b)))))

;; =========================================================================
;; Models (same shapes as cone_regen_test)
;; =========================================================================

(def chain3 (gen [] (let [a (trace :a (dist/gaussian 0 1))
                          b (trace :b (dist/gaussian a 1))
                          c (trace :c (dist/gaussian b 1))]
                      c)))
(def skip (gen [] (let [a (trace :a (dist/gaussian 0 1))
                        b (trace :b (dist/gaussian a 1))
                        c (trace :c (dist/gaussian (mx/add a b) 1))]
                    c)))
(def diamond (gen [] (let [a (trace :a (dist/gaussian 0 1))
                           b (trace :b (dist/gaussian a 1))
                           c (trace :c (dist/gaussian a 2))
                           d (trace :d (dist/gaussian (mx/add b c) 1))]
                       d)))
(def hub (gen [] (let [a (trace :a (dist/gaussian 0 1))
                       y1 (trace :y1 (dist/gaussian a 1))
                       y2 (trace :y2 (dist/gaussian a 1))
                       y3 (trace :y3 (dist/gaussian a 1))]
                   a)))

;; =========================================================================
;; 1. Attachment
;; =========================================================================
(println "\n-- 1. :vcone-regenerate attachment --")

(doseq [[nm m] [["chain3" chain3] ["skip" skip] ["diamond" diamond] ["hub" hub]]]
  (assert-true (str nm " has :vcone-regenerate")
               (fn? (:vcone-regenerate (:schema m)))))

;; =========================================================================
;; 2. B1: bitwise ground truth vs the full-body batched handler
;; =========================================================================
(println "\n-- 2. vregenerate: vcone == batched handler (bitwise, same key) --")

(def ^:private N 64)

(defn- vleaf [vt addr] (cm/get-value (cm/get-submap (:choices vt) addr)))

(doseq [[nm m addrs] [["chain3" chain3 [:a :b :c]]
                      ["skip" skip [:a :b :c]]
                      ["diamond" diamond [:a :b :c :d]]
                      ["hub" hub [:a :y1 :y2 :y3]]]
        addr addrs
        seed [7 8]]
  (let [vt (dyn/vsimulate m [] N (rng/fresh-key (* seed 13)))
        k #(rng/fresh-key seed)
        rc (dyn/vregenerate m vt (sel/select addr) (k))
        rh (dyn/vregenerate (update m :schema dissoc :vcone-regenerate)
                            vt (sel/select addr) (k))]
    (assert-true (str nm " " addr " s" seed ": resampled [N] values bit-identical")
                 (arrays-equal? (vleaf (:vtrace rc) addr) (vleaf (:vtrace rh) addr)))
    (assert-true (str nm " " addr " s" seed ": weight [N] within 1e-4")
                 (< (max-abs-diff (:weight rc) (:weight rh)) 1e-4))
    (assert-true (str nm " " addr " s" seed ": score [N] within 1e-4")
                 (< (max-abs-diff (:score (:vtrace rc)) (:score (:vtrace rh))) 1e-4))
    (doseq [other (remove #{addr} addrs)]
      (assert-true (str nm " " addr " s" seed ": retained " other " unchanged")
                   (arrays-equal? (vleaf vt other) (vleaf (:vtrace rc) other))))))

;; =========================================================================
;; 3. B2: declines fall through to the batched handler unchanged
;; =========================================================================
(println "\n-- 3. declines --")

(let [vt (dyn/vsimulate chain3 [] N (rng/fresh-key 91))
      k #(rng/fresh-key 92)
      rc (dyn/vregenerate chain3 vt (sel/select :a :c) (k))
      rh (dyn/vregenerate (update chain3 :schema dissoc :vcone-regenerate)
                          vt (sel/select :a :c) (k))]
  (assert-true "multi-site: values match handler"
               (and (arrays-equal? (vleaf (:vtrace rc) :a) (vleaf (:vtrace rh) :a))
                    (arrays-equal? (vleaf (:vtrace rc) :c) (vleaf (:vtrace rh) :c))))
  (assert-true "multi-site: weight matches handler"
               (< (max-abs-diff (:weight rc) (:weight rh)) 1e-4)))

(let [vt (dyn/vsimulate chain3 [] N (rng/fresh-key 93))
      r (dyn/vregenerate chain3 vt sel/none (rng/fresh-key 94))]
  (assert-true "empty selection: [N] weight is ~0 (full-pass decline)"
               (< (mx/item (mx/amax (mx/abs (:weight r)))) 1e-4)))

;; =========================================================================
;; 4. B3: vmh — stationary distribution + incremental-score drift
;; =========================================================================
(println "\n-- 4. vmh: prior stationarity + score drift (N=256, 40 sweeps) --")

(let [n 256
      vt0 (dyn/vsimulate chain3 [] n (rng/fresh-key 400))
      vt (mcmc/vmh chain3 vt0 {:iters 40 :key (rng/fresh-key 401)})
      a (vleaf vt :a)
      a-mean (mx/item (mx/mean a))
      a-sd (js/Math.sqrt (mx/item (mx/variance a)))]
  (assert-close "vmh: lane-mean of :a ~ prior mean 0" 0.0 a-mean 0.25)
  (assert-close "vmh: lane-sd of :a ~ prior sd 1" 1.0 a-sd 0.35)
  ;; drift: a no-op vregenerate (empty selection) does a FULL batched handler
  ;; re-score of the retained choices — compare against the incremental score
  (let [fresh (dyn/vregenerate chain3 vt sel/none (rng/fresh-key 402))]
    (assert-true "vmh: incremental [N] scores == fresh full re-score (1e-2)"
                 (< (max-abs-diff (:score vt) (:score (:vtrace fresh))) 1e-2))))

(println (str "\n== vcone-regen: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
