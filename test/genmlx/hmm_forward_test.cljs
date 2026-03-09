(ns genmlx.hmm-forward-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.inference.hmm-forward :as hmm]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println "  PASS:" msg)
      (println "  FAIL:" msg "- expected" expected "got" actual "diff" diff))))

(println "\n=== HMM Forward Middleware Tests ===\n")

;; ---------------------------------------------------------------------------
;; Test setup: 2-state HMM
;; State 0: "low" emission ~ N(0, 1)
;; State 1: "high" emission ~ N(5, 1)
;; ---------------------------------------------------------------------------

(def log-trans
  ;; Transition matrix (log-space):
  ;; P(stay) = 0.9, P(switch) = 0.1
  (mx/array [[(js/Math.log 0.9) (js/Math.log 0.1)]
             [(js/Math.log 0.1) (js/Math.log 0.9)]]))

(def K 2) ;; number of states

;; Emission log-probs for a given observation
(defn emission-log-probs [obs]
  "Compute [K]-shaped log p(obs | state=k) for Gaussian emissions."
  (let [lp0 (dc/dist-log-prob (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) obs)
        lp1 (dc/dist-log-prob (dist/gaussian (mx/scalar 5.0) (mx/scalar 1.0)) obs)]
    (mx/stack [lp0 lp1])))

;; -- 1. Pure hmm-predict --
(println "-- 1. Pure hmm-predict --")
(let [;; Start with certainty in state 0
      log-alpha (mx/array [(js/Math.log 1.0) (js/Math.log 1e-30)])
      predicted (hmm/hmm-predict log-alpha log-trans)]
  (mx/eval! predicted)
  (assert-true "predicted is [2]-shaped" (= [2] (mx/shape predicted)))
  ;; After transition from state 0: P(0)≈0.9, P(1)≈0.1
  (let [probs (mx/exp predicted)
        p0 (mx/item (mx/index probs 0))
        p1 (mx/item (mx/index probs 1))]
    (assert-close "P(state 0) ≈ 0.9" 0.9 p0 0.02)
    (assert-close "P(state 1) ≈ 0.1" 0.1 p1 0.02)
    (println "  P(state 0):" (.toFixed p0 4) "P(state 1):" (.toFixed p1 4))))

;; -- 2. Pure hmm-update --
(println "\n-- 2. Pure hmm-update --")
(let [;; Uniform prior
      log-alpha (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)])
      ;; Observe value near state 1 (x=4.5)
      obs (mx/scalar 4.5)
      log-ep (emission-log-probs obs)
      mask (mx/scalar 1.0)
      {:keys [log-alpha ll]} (hmm/hmm-update log-alpha log-ep mask)]
  (mx/eval! log-alpha)
  (mx/eval! ll)
  (assert-true "updated belief is [2]-shaped" (= [2] (mx/shape log-alpha)))
  (assert-true "ll is scalar" (= [] (mx/shape ll)))
  ;; After seeing x=4.5, state 1 should be much more likely
  (let [probs (mx/exp log-alpha)
        p1 (mx/item (mx/index probs 1))]
    (assert-true "state 1 posterior > 0.95" (> p1 0.95))
    (println "  P(state 1 | x=4.5):" (.toFixed p1 4)
             "marginal LL:" (.toFixed (mx/item ll) 4))))

;; -- 3. hmm-update with missing data --
(println "\n-- 3. Missing data (mask=0) --")
(let [log-alpha (mx/array [(js/Math.log 0.7) (js/Math.log 0.3)])
      obs (mx/scalar 4.5)
      log-ep (emission-log-probs obs)
      mask (mx/scalar 0.0)
      {:keys [log-alpha ll]} (hmm/hmm-update log-alpha log-ep mask)]
  (mx/eval! log-alpha)
  (mx/eval! ll)
  ;; With mask=0, belief should not change
  (let [probs (mx/exp log-alpha)
        p0 (mx/item (mx/index probs 0))]
    (assert-close "belief unchanged with mask=0" 0.7 p0 0.01)
    (assert-close "LL = 0 with mask=0" 0.0 (mx/item ll) 0.001)
    (println "  P(state 0):" (.toFixed p0 4) "LL:" (.toFixed (mx/item ll) 4))))

;; -- 4. hmm-step: predict + update --
(println "\n-- 4. hmm-step (predict + update) --")
(let [log-alpha (mx/array [(js/Math.log 1.0) (js/Math.log 1e-30)])
      obs (mx/scalar 4.8)
      log-ep (emission-log-probs obs)
      mask (mx/scalar 1.0)
      {:keys [log-alpha ll]} (hmm/hmm-step log-alpha log-trans log-ep mask)]
  (mx/eval! log-alpha)
  (mx/eval! ll)
  ;; Starting in state 0 with P(switch)=0.1, then observing near state 1
  ;; Posterior should favor state 1 despite low transition prob
  (let [probs (mx/exp log-alpha)
        p1 (mx/item (mx/index probs 1))]
    (assert-true "state 1 posterior > 0.5 after observation near 5" (> p1 0.5))
    (println "  P(state 1):" (.toFixed p1 4) "LL:" (.toFixed (mx/item ll) 4))))

;; -- 5. Full sequence: brute force vs forward algorithm --
(println "\n-- 5. Sequence marginal LL: exact vs brute force --")
(let [;; 3-step sequence of observations
      obs-seq [(mx/scalar 0.2) (mx/scalar 0.1) (mx/scalar 4.8)]
      ;; Forward algorithm
      forward-ll
      (loop [t 0
             log-alpha (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)])
             acc-ll (mx/scalar 0.0)]
        (if (>= t (count obs-seq))
          acc-ll
          (let [predicted (if (zero? t)
                            log-alpha
                            (hmm/hmm-predict log-alpha log-trans))
                log-ep (emission-log-probs (nth obs-seq t))
                {:keys [log-alpha ll]} (hmm/hmm-update predicted log-ep (mx/scalar 1.0))]
            (recur (inc t) log-alpha (mx/add acc-ll ll)))))
      _ (mx/eval! forward-ll)
      ;; Brute force: enumerate all 2^3 = 8 state sequences
      mu [0.0 5.0]
      log-init [(js/Math.log 0.5) (js/Math.log 0.5)]
      trans-p [[0.9 0.1] [0.1 0.9]]
      ;; Compute log-prob for each of 8 state sequences
      log-probs
      (for [s0 [0 1] s1 [0 1] s2 [0 1]]
        (+ (nth log-init s0)
           (mx/item (dc/dist-log-prob
             (dist/gaussian (mx/scalar (nth mu s0)) (mx/scalar 1.0))
             (nth obs-seq 0)))
           (js/Math.log (get-in trans-p [s0 s1]))
           (mx/item (dc/dist-log-prob
             (dist/gaussian (mx/scalar (nth mu s1)) (mx/scalar 1.0))
             (nth obs-seq 1)))
           (js/Math.log (get-in trans-p [s1 s2]))
           (mx/item (dc/dist-log-prob
             (dist/gaussian (mx/scalar (nth mu s2)) (mx/scalar 1.0))
             (nth obs-seq 2)))))
      ;; logsumexp over all 8 sequences
      max-lp (apply max log-probs)
      brute-ll (+ max-lp (js/Math.log
                           (reduce + (map #(js/Math.exp (- % max-lp)) log-probs))))]
  (assert-close "forward LL matches brute force"
                brute-ll (mx/item forward-ll) 0.01)
  (println "  forward LL:" (.toFixed (mx/item forward-ll) 4)
           "brute force LL:" (.toFixed brute-ll 4)))

;; -- 6. Batched (multi-element) forward --
(println "\n-- 6. Batched [P,K] forward --")
(let [P 50
      ;; P elements, each starting uniform
      log-alpha (mx/multiply (mx/scalar (- (js/Math.log 2)))
                             (mx/ones [P K]))
      ;; Each element observes a different value
      obs-vals (mx/add (mx/multiply (rng/uniform (rng/fresh-key) [P])
                                    (mx/scalar 6.0))
                       (mx/scalar -0.5))
      ;; Compute [P,K] emission log-probs
      log-ep (let [lp0 (dc/dist-log-prob (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) obs-vals)
                   lp1 (dc/dist-log-prob (dist/gaussian (mx/scalar 5.0) (mx/scalar 1.0)) obs-vals)]
               (mx/stack [lp0 lp1] -1))  ;; stack along last axis -> [P,2]
      mask (mx/ones [P])
      {:keys [log-alpha ll]} (hmm/hmm-update log-alpha log-ep mask)]
  (mx/eval! log-alpha)
  (mx/eval! ll)
  (assert-true "batched belief is [P,K]-shaped" (= [P K] (mx/shape log-alpha)))
  (assert-true "batched LL is [P]-shaped" (= [P] (mx/shape ll)))
  (println "  belief shape:" (mx/shape log-alpha) "LL shape:" (mx/shape ll)
           "mean LL:" (.toFixed (mx/item (mx/mean ll)) 4)))

;; -- 7. Handler middleware: gen function under HMM handler --
(println "\n-- 7. Handler middleware --")

(def hmm-step-fn
  (gen [obs log-ep]
    (let [z-prev (mx/scalar 0 mx/int32)  ;; dummy, handler ignores it
          z (trace :z (hmm/hmm-latent (mx/index log-trans 0) z-prev))
          _ (trace :obs (hmm/hmm-obs log-ep (mx/scalar 1.0)))]
      z)))

(let [obs (mx/scalar 4.5)
      log-ep (emission-log-probs obs)
      constraints (cm/set-value cm/EMPTY :obs obs)
      result (hmm/hmm-generate hmm-step-fn [obs log-ep] constraints
                               :z log-trans 0 K (rng/fresh-key))]
  (assert-true "hmm-generate returns result" (some? result))
  (mx/eval! (or (:hmm-ll result) (mx/scalar 0.0)))
  (let [ll (mx/item (or (:hmm-ll result) (mx/scalar 0.0)))]
    (assert-true "LL is finite" (js/isFinite ll))
    (println "  hmm-ll:" (.toFixed ll 4))
    (let [belief (:hmm-belief result)]
      (when (mx/array? belief)
        (mx/eval! belief)
        (println "  hmm-belief shape:" (mx/shape belief)))))
  (assert-true "hmm-belief exists" (some? (:hmm-belief result))))

;; -- 8. hmm-fold over sequence --
(println "\n-- 8. hmm-fold over sequence --")
(let [;; Generate observations from a known sequence: state 0, 0, 1, 1, 1
      true-states [0 0 1 1 1]
      obs-seq (mapv (fn [s]
                      (let [mu (if (zero? s) 0.0 5.0)]
                        (mx/scalar (+ mu (* 0.3 (- (rand) 0.5))))))
                    true-states)
      T (count obs-seq)
      context-fn (fn [t]
                   (let [obs (nth obs-seq t)
                         log-ep (emission-log-probs obs)]
                     {:args [obs log-ep]
                      :constraints (cm/set-value cm/EMPTY :obs obs)}))
      {:keys [ll belief]} (hmm/hmm-fold hmm-step-fn :z log-trans 0 K T context-fn)]
  (mx/eval! ll)
  (assert-true "fold LL is scalar" (= [] (mx/shape ll)))
  (assert-true "fold LL is finite" (js/isFinite (mx/item ll)))
  ;; After seeing 3 observations near 5, should believe in state 1
  (let [final-probs (mx/exp belief)
        p1 (mx/item (mx/index final-probs 1))]
    (assert-true "final belief favors state 1" (> p1 0.8))
    (println "  total LL:" (.toFixed (mx/item ll) 4)
             "final P(state 1):" (.toFixed p1 4))))

;; -- 9. Batched hmm-fold --
(println "\n-- 9. Batched hmm-fold (P elements) --")
(def hmm-step-batched
  (gen [log-ep-batched]
    (let [z-prev (mx/scalar 0 mx/int32)
          z (trace :z (hmm/hmm-latent (mx/index log-trans 0) z-prev))
          _ (trace :obs (hmm/hmm-obs log-ep-batched (mx/scalar 1.0)))]
      z)))

(let [P 30
      T 5
      ;; Generate P independent sequences, each T steps
      ;; All from state 0 (emissions near 0)
      context-fn (fn [t]
                   (let [obs (mx/multiply (rng/uniform (rng/fresh-key (* 1000 t)) [P])
                                          (mx/scalar 2.0))
                         lp0 (dc/dist-log-prob (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) obs)
                         lp1 (dc/dist-log-prob (dist/gaussian (mx/scalar 5.0) (mx/scalar 1.0)) obs)
                         log-ep (mx/stack [lp0 lp1] -1)]
                     {:args [log-ep]
                      :constraints (cm/set-value cm/EMPTY :obs obs)}))
      {:keys [ll belief]} (hmm/hmm-fold hmm-step-batched :z log-trans P K T context-fn)]
  (mx/eval! ll)
  (assert-true "batched fold LL is [P]-shaped" (= [P] (mx/shape ll)))
  (assert-true "batched fold belief is [P,K]-shaped" (= [P K] (mx/shape belief)))
  (println "  LL shape:" (mx/shape ll) "belief shape:" (mx/shape belief)
           "mean LL:" (.toFixed (mx/item (mx/mean ll)) 4)))

;; -- 10. Compose with Kalman via analytical.cljs --
(println "\n-- 10. Composable middleware (wrap-analytical) --")
(let [dispatch (hmm/make-hmm-dispatch :z log-trans)
      transition (ana/wrap-analytical h/generate-transition dispatch)]
  (assert-true "wrap-analytical returns function" (fn? transition))
  (println "  wrap-analytical creates composable transition"))

(println "\n=== Done ===")
