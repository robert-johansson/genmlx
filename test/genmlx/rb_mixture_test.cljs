;; @tier fast
(ns genmlx.rb-mixture-test
  "Acceptance for genmlx.inference.rb-mixture — the Rao-Blackwellized (collapsed)
   Gaussian-mixture marginal-likelihood scorer (R4.5-minimal, genmlx-9mos): an
   EXACT oracle primitive that analytically integrates out the component means and
   enumerates only the discrete assignments. Native-free + deterministic, so every
   assertion is reproducible and non-circular.

   Run: bun run --bun nbb test/genmlx/rb_mixture_test.cljs

   PROVES:
     (a) K=1 identity: collapsed evidence == an INDEPENDENT closed-form single-mean
         Gaussian-Gaussian marginal (the synth_test form, derived outside this ns).
     (b) MONOTONE PARTIAL CREDIT + OCCAM on a true 2-cluster mixture: K=2 >> K=1
         (the right structure is rewarded), and K=3 does NOT exceed K=2 beyond a
         small epsilon (the unwarranted component is penalized).
     (c) OCCAM on single-Gaussian data: K=1 is best (within epsilon of K=2) — the
         scorer does not prefer a spurious mixture.
     (d) DETERMINISM / variance: the collapsed estimator is bit-identical across
         runs (zero Monte Carlo variance), unlike a naive IS estimate which spreads."
  (:require [genmlx.inference.rb-mixture :as rb]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(defn assert-close [label expected actual tol]
  (let [ok (and (number? actual) (js/isFinite actual)
                (<= (js/Math.abs (- expected actual)) tol))]
    (if ok
      (do (swap! pass inc) (println "  PASS" label (str "(" actual " ~ " expected ")")))
      (do (swap! fail inc) (println "  FAIL" label (str "(" actual " vs " expected " tol " tol ")"))))))

;; ===========================================================================
;; INDEPENDENT closed-form Gaussian-Gaussian marginal — DERIVED OUTSIDE GenMLX
;; (the same Sherman-Morrison form used in synth_test). mu ~ N(m0, s0^2),
;; y_i ~ N(mu, sn^2). The K=1 identity check below compares the scorer's
;; enumeration against THIS, so the cross-check is non-circular.
;; ===========================================================================
(defn gaussian-gaussian-marginal [ys m0 s0 sn]
  (let [n (count ys) s0² (* s0 s0) sn² (* sn sn)
        ds (map #(- % m0) ys) sd (reduce + ds) sd² (reduce + (map #(* % %) ds))
        denom (+ sn² (* n s0²))
        logdet (+ (js/Math.log denom) (* (dec n) (js/Math.log sn²)))
        quad (* (/ 1.0 sn²) (- sd² (* (/ s0² denom) (* sd sd))))]
    (- (* -0.5 n (js/Math.log (* 2 js/Math.PI))) (* 0.5 logdet) (* 0.5 quad))))

(defn- fmt [x] (.toFixed (js/Number x) 4))

;; ===========================================================================
;; (a) K=1 identity — non-circular cross-check
;; ===========================================================================

(defn part-a []
  (println "\n== (a) K=1 identity vs independent closed-form Gaussian-Gaussian ==")
  ;; several configs so the agreement is not a single-point coincidence
  (doseq [[ys m0 s0 sigma]
          [[[2.0 2.3 1.7 2.1]              2.0 1.0 0.5]
           [[-3.1 -2.9 3.0 3.2 2.8]        0.0 5.0 0.5]
           [[10.0 11.0 9.5 10.5 10.2 9.8]  0.0 20.0 1.0]
           [[0.0]                          0.0 1.0 1.0]]]
    (let [c1    (rb/collapsed-gmm-log-evidence ys {:k 1 :m0 m0 :s0 s0 :sigma sigma})
          sg    (rb/single-gaussian-log-evidence ys {:m0 m0 :s0 s0 :sigma sigma})
          truth (gaussian-gaussian-marginal ys m0 s0 sigma)]
      (assert-close (str "collapsed K=1 == single-gaussian helper  (m0=" m0 " s0=" s0 " σ=" sigma ")")
                    sg c1 1e-9)
      (assert-close (str "collapsed K=1 == INDEPENDENT closed form  (m0=" m0 " s0=" s0 " σ=" sigma ")")
                    truth c1 1e-9))))

;; ===========================================================================
;; (b) monotone partial credit + Occam — true 2-cluster mixture
;; ===========================================================================

(defn part-b []
  (println "\n== (b) monotone partial-credit + Occam: true 2-cluster mixture ==")
  ;; data drawn around -3 and +3 (clearly two components), with a wide prior
  ;; so neither component mean is fought by the prior.
  (let [ys    [-3.1 -2.9 -3.2 -2.8 3.0 3.2 2.9 3.1]
        prior {:m0 0.0 :s0 5.0 :sigma 0.5}
        ev    (fn [k] (rb/collapsed-gmm-log-evidence ys (assoc prior :k k)))
        e1 (ev 1) e2 (ev 2) e3 (ev 3)]
    (println (str "    uniform prior:  K=1=" (fmt e1) "  K=2=" (fmt e2) "  K=3=" (fmt e3)))
    (println (str "    delta K1->K2 = +" (fmt (- e2 e1)) " nats   |   delta K2->K3 = " (fmt (- e3 e2)) " nats"))
    (assert-true "K=2 beats K=1 by a CLEAR margin (>10 nats — right structure rewarded)"
                 (> (- e2 e1) 10.0))
    (assert-true "K=3 does NOT exceed K=2 beyond a small epsilon (Occam: no reward for the spurious component)"
                 (<= e3 (+ e2 0.5)))
    (assert-true "the partial-credit path is monotone up to K=2 then non-improving (peak at the true K)"
                 (and (> e2 e1) (<= e3 e2)))

    ;; same qualitative shape under the Dirichlet-marginalized assignment prior
    (let [d (fn [k] (rb/collapsed-gmm-log-evidence ys (assoc prior :k k :weights :dirichlet :alpha 1.0)))
          d1 (d 1) d2 (d 2) d3 (d 3)]
      (println (str "    dirichlet prior: K=1=" (fmt d1) "  K=2=" (fmt d2) "  K=3=" (fmt d3)))
      (assert-true "dirichlet prior: K=2 >> K=1" (> (- d2 d1) 10.0))
      (assert-true "dirichlet prior: K=3 within epsilon of K=2 (no reward for spurious component)"
                   (<= d3 (+ d2 0.5))))))

;; ===========================================================================
;; (c) Occam on single-Gaussian data — no spurious mixture preferred
;; ===========================================================================

(defn part-c []
  (println "\n== (c) Occam on single-Gaussian data: K=1 best, no spurious mixture ==")
  ;; one Gaussian blob near 0 — there is no real cluster structure.
  (let [ys    [0.1 -0.2 0.05 0.3 -0.1 0.2 -0.05 0.15]
        prior {:m0 0.0 :s0 5.0 :sigma 0.5}
        ev    (fn [k] (rb/collapsed-gmm-log-evidence ys (assoc prior :k k)))
        e1 (ev 1) e2 (ev 2) e3 (ev 3)]
    (println (str "    uniform prior:  K=1=" (fmt e1) "  K=2=" (fmt e2) "  K=3=" (fmt e3)))
    (assert-true "K=1 is the best (or within epsilon of K=2) — no spurious-mixture preference"
                 (>= e1 (- e2 0.5)))
    (assert-true "evidence is non-increasing in K on structureless data (K1 >= K2 >= K3 - eps)"
                 (and (>= e1 (- e2 1e-6)) (>= e2 (- e3 1e-6))))

    (let [d (fn [k] (rb/collapsed-gmm-log-evidence ys (assoc prior :k k :weights :dirichlet :alpha 1.0)))
          d1 (d 1) d2 (d 2) d3 (d 3)]
      (println (str "    dirichlet prior: K=1=" (fmt d1) "  K=2=" (fmt d2) "  K=3=" (fmt d3)))
      (assert-true "dirichlet prior: K=1 best (within epsilon of K=2)" (>= d1 (- d2 0.5))))))

;; ===========================================================================
;; (d) determinism (zero variance) vs a naive Monte-Carlo estimate's spread
;; ===========================================================================

(defn- mc-gmm-log-evidence
  "Naive IS estimate of the SAME marginal: sample assignments z ~ uniform and
   component means mu_k ~ prior, score the joint likelihood, log-mean-exp. This
   is the high-variance fallback the collapsed estimator replaces — it samples
   BOTH the discrete K^N sum AND the continuous means. Used only to exhibit MC
   spread; not part of the scorer."
  [ys {:keys [k m0 s0 sigma]} n-samples]
  (let [ysv (vec ys) n (count ysv)
        ln2pi (js/Math.log (* 2 js/Math.PI))
        snsq (* sigma sigma)
        gauss (fn [] ; Box-Muller standard normal
                (let [u1 (js/Math.random) u2 (js/Math.random)]
                  (* (js/Math.sqrt (* -2 (js/Math.log u1)))
                     (js/Math.cos (* 2 js/Math.PI u2)))))
        logp-y (fn [y mu] (- (* -0.5 ln2pi) (js/Math.log sigma)
                             (/ (* (- y mu) (- y mu)) (* 2 snsq))))
        sample (fn []
                 (let [mus (vec (repeatedly k #(+ m0 (* s0 (gauss)))))
                       zs  (vec (repeatedly n #(rand-int k)))]
                   ;; p(z)=(1/k)^n is constant over uniform z draws -> the IS
                   ;; weight is the data likelihood (prior-proposal for mu and z).
                   (reduce + (map (fn [y z] (logp-y y (nth mus z))) ysv zs))))
        ws (vec (repeatedly n-samples sample))
        m  (apply max ws)]
    (+ m (js/Math.log (/ (reduce + (map #(js/Math.exp (- % m)) ws)) n-samples)))))

(defn part-d []
  (println "\n== (d) determinism (exact, zero variance) vs naive Monte-Carlo spread ==")
  (let [ys   [-3.1 -2.9 -3.2 -2.8 3.0 3.2 2.9 3.1]
        opts {:k 2 :m0 0.0 :s0 5.0 :sigma 0.5}
        a (rb/collapsed-gmm-log-evidence ys opts)
        b (rb/collapsed-gmm-log-evidence ys opts)
        c (rb/collapsed-gmm-log-evidence ys opts)]
    (assert-true "collapsed estimator is BIT-IDENTICAL across runs (deterministic, zero variance)"
                 (and (= a b) (= b c)))
    ;; contrast: the naive IS estimate of the SAME marginal spreads run-to-run
    ;; and is biased LOW (it almost never hits the right assignment by chance).
    (let [mc (vec (repeatedly 5 #(mc-gmm-log-evidence ys opts 4000)))
          lo (apply min mc) hi (apply max mc) spread (- hi lo)]
      (println (str "    exact (collapsed) = " (fmt a)
                    "   |   naive IS over 5 runs: [" (fmt lo) " , " (fmt hi)
                    "]  spread=" (fmt spread)))
      (assert-true "naive Monte-Carlo estimate has NON-ZERO run-to-run spread (variance the RB scorer removes)"
                   (> spread 1e-6))
      (assert-true "naive Monte-Carlo estimate is biased LOW vs the exact value (misses the right assignment)"
                   (< hi a)))))

;; ===========================================================================

(defn- summary []
  (println (str "\n== rb-mixture (genmlx-9mos): " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(part-a)
(part-b)
(part-c)
(part-d)
(summary)
