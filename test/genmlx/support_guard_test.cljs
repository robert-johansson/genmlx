;; @tier fast
(ns genmlx.support-guard-test
  "genmlx-7oen: out-of-support log-probs must be -Inf, never finite garbage
   or NaN.

   Audit genmlx-ansg (VERIFIED) found nine families missing the support-
   membership guard that uniform/exponential already carry: geometric scored
   an impossible value HIGHER than a legal one;
   bernoulli/poisson/binomial/neg-binomial returned finite garbage on
   non-integer/negative v; gamma/beta/log-normal/inv-gamma returned NaN
   (log of a negative) — silently poisoning IS/SMC weights and MH accept
   comparisons. This suite pins: out-of-support -> exactly -Inf (finite
   check + NaN check), in-support values unchanged/finite, and the specific
   audit repros (geometric ordering, bernoulli lp(0.5)).

   CORRECTION (genmlx-4x5w): the sentence above used to name
   discrete-uniform and truncated-normal among the families that 'already
   carry' the guard. That claim was copied from genmlx-7oen's scope note and
   was FALSE for both, and being written down here is what kept anyone from
   testing them. Measured on sm_120 before the fix:
     - discrete-uniform(0,10): lp(3.5) = lp(3) = -2.3979. No integer factor at
       all, on a family whose own sampler only emits integers.
     - dirichlet: no validate clause and no support mask — NaN at alpha=1 with
       a zero component, +Inf at alpha<1, a finite +1.18 for the off-simplex
       v=[0.3,0.3,0.3], and the zero component was reachable from our own
       batch sampler (min component over 2000 draws was exactly 0.0).
     - beta: the mask is INCLUSIVE of 0 and 1, so unlike gamma's strict v>0 the
       0*(-Inf) boundary product escaped: beta(1,3) lp(0) and beta(2,1) lp(1)
       were NaN.
   All three are fixed and pinned below.

   truncated-normal (genmlx-g2iu) is now fixed too. Its normalizer collapsed
   beyond ~5-6 sigma on BOTH tails, turning a subtracted log-normalizer into a
   +61 nat bonus. This bean was long recorded as blocked on a native `mx/erfc`;
   it was not — log(erfc) is computable in log space from `erf`/`expm1`/`log1p`,
   all of which the membrane already exports, so the fix is pure ClojureScript
   (`dist/log-normal-mass`). What remains genuinely native is the far-tail
   SAMPLER, which saturates `erfinv` and returns a bound: genmlx-smrk.

   Run: bunx --bun nbb@1.4.208 test/genmlx/support_guard_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def NEG-INF js/Number.NEGATIVE_INFINITY)

(defn- lp [d v] (mx/realize (dc/dist-log-prob d (mx/scalar v))))
(defn- lpv [d v] (mx/realize (dc/dist-log-prob d (mx/array v))))

(defn- check-dist [label d bad-vs good-vs]
  (doseq [v bad-vs]
    (let [x (lp d v)]
      (assert-true (str label ": lp(" v ") = -Inf (got " x ")")
                   (= x NEG-INF))))
  (doseq [v good-vs]
    (let [x (lp d v)]
      (assert-true (str label ": lp(" v ") finite (got " x ")")
                   (js/isFinite x)))))

;; ===========================================================================
(println "\n-- discrete families: integer + range masks --")

(check-dist "geometric(0.3)" (dist/geometric 0.3)
            [-1 -0.5 0.5 1.7] [0 1 2 10])

;; the audit's headline: an impossible value must NOT outscore a legal one
(let [d (dist/geometric 0.3)]
  (assert-true "geometric: lp(-1) no longer beats lp(0)"
               (< (lp d -1) (lp d 0))))

(check-dist "bernoulli(0.4)" (dist/bernoulli 0.4)
            [0.5 -1 2 0.999] [0 1])

(let [d (dist/bernoulli 0.4)]
  (assert-true "bernoulli: lp(1) = log(p) exactly"
               (< (js/Math.abs (- (lp d 1) (js/Math.log 0.4))) 1e-6))
  (assert-true "bernoulli: lp(0) = log(1-p) exactly"
               (< (js/Math.abs (- (lp d 0) (js/Math.log 0.6))) 1e-6)))

(check-dist "poisson(3)" (dist/poisson 3.0)
            [-1 -0.5 1.5 2.0000305] [0 1 5 20])

(check-dist "binomial(10, 0.4)" (dist/binomial 10 0.4)
            [-1 0.5 3.5 11 15] [0 3 10])

(check-dist "neg-binomial(5, 0.4)" (dist/neg-binomial 5 0.4)
            [-1 -0.5 2.5] [0 2 12])

;; genmlx-4x5w: discrete-uniform had NO integer factor. lp(3.5) was exactly
;; lp(3) — a value the sampler can never produce scored as well as a legal one.
(check-dist "discrete-uniform(0, 10)" (dist/discrete-uniform 0 10)
            [-1 -0.5 0.5 3.5 10.5 11] [0 3 10])

(let [d (dist/discrete-uniform 0 10)]
  (assert-true "discrete-uniform: lp(3.5) is NOT lp(3) (the genmlx-4x5w repro)"
               (not= (lp d 3.5) (lp d 3)))
  (assert-true "discrete-uniform: lp(3) = -log(11) exactly"
               (< (js/Math.abs (- (lp d 3) (- (js/Math.log 11)))) 1e-6))
  (assert-true "discrete-uniform: every legal integer scores identically"
               (apply = (map #(lp d %) (range 0 11)))))

;; ===========================================================================
(println "\n-- continuous families: positivity / interval masks (NaN class) --")

(check-dist "gamma(2, 1.5)" (dist/gamma-dist 2.0 1.5)
            [-1 -0.001 0] [0.1 1.0 7.0])

(check-dist "inv-gamma(2, 1.5)" (dist/inv-gamma 2.0 1.5)
            [-1 -0.001 0] [0.1 1.0 7.0])

(check-dist "log-normal(0, 1)" (dist/log-normal 0.0 1.0)
            [-1 -0.001 0] [0.1 1.0 7.0])

(check-dist "beta(2, 3)" (dist/beta-dist 2.0 3.0)
            [-0.1 1.1 2 -1] [0.2 0.5 0.8])

;; ===========================================================================
;; genmlx-4x5w: beta's mask is inclusive of the endpoints, so unlike gamma
;; (strict v>0) the 0*(-Inf) boundary product is NOT discarded by the where.
;; At alpha=1 the density at v=0 is finite and equal to 1/B(1,b) = b.
(println "\n-- beta endpoints: xlogy, not NaN --")

(let [d13 (dist/beta-dist 1.0 3.0)
      d21 (dist/beta-dist 2.0 1.0)]
  (assert-true (str "beta(1,3): lp(0) is not NaN (got " (lp d13 0) ")")
               (not (js/isNaN (lp d13 0))))
  (assert-true (str "beta(1,3): lp(0) = log(3) = " (js/Math.log 3))
               (< (js/Math.abs (- (lp d13 0) (js/Math.log 3))) 1e-5))
  (assert-true (str "beta(2,1): lp(1) is not NaN (got " (lp d21 1) ")")
               (not (js/isNaN (lp d21 1))))
  (assert-true (str "beta(2,1): lp(1) = log(2) = " (js/Math.log 2))
               (< (js/Math.abs (- (lp d21 1) (js/Math.log 2))) 1e-5))
  ;; the interior must be untouched by the xlogy gate
  (assert-true "beta(1,3): lp(0.5) = log(3*0.25) (interior unchanged)"
               (< (js/Math.abs (- (lp d13 0.5) (js/Math.log (* 3 0.25)))) 1e-5))
  ;; alpha>1 still vanishes at the boundary: -Inf, not a finite bonus
  (assert-true "beta(2,3): lp(0) = -Inf (density genuinely 0 there)"
               (= (lp (dist/beta-dist 2.0 3.0) 0) NEG-INF)))

;; ===========================================================================
;; genmlx-4x5w: dirichlet had no validate clause and no support mask.
(println "\n-- dirichlet: open-simplex mask + xlogy --")

(let [uniform  (dist/dirichlet [1.0 1.0 1.0])   ;; alpha = 1 -> the NaN repro
      sparse   (dist/dirichlet [0.5 0.5 0.5])   ;; alpha < 1 -> the +Inf repro
      peaked   (dist/dirichlet [2.0 2.0 2.0])]
  ;; (1) zero component, alpha = 1: was NaN (0 * log 0)
  (assert-true (str "dirichlet(1,1,1): lp([0,0.5,0.5]) not NaN (got "
                    (lpv uniform [0.0 0.5 0.5]) ")")
               (not (js/isNaN (lpv uniform [0.0 0.5 0.5]))))
  (assert-true "dirichlet(1,1,1): lp([0,0.5,0.5]) = -Inf (off the open simplex)"
               (= (lpv uniform [0.0 0.5 0.5]) NEG-INF))
  ;; (2) zero component, alpha < 1: was +Inf — an infinite REWARD
  (assert-true (str "dirichlet(.5,.5,.5): lp([0,0.5,0.5]) not +Inf (got "
                    (lpv sparse [0.0 0.5 0.5]) ")")
               (not= (lpv sparse [0.0 0.5 0.5]) js/Number.POSITIVE_INFINITY))
  (assert-true "dirichlet(.5,.5,.5): lp([0,0.5,0.5]) = -Inf"
               (= (lpv sparse [0.0 0.5 0.5]) NEG-INF))
  ;; (3) negative component: was NaN
  (assert-true "dirichlet(2,2,2): lp([-0.1,0.6,0.5]) = -Inf (was NaN)"
               (= (lpv peaked [-0.1 0.6 0.5]) NEG-INF))
  ;; (4) off the simplex entirely: was a finite +1.18
  (assert-true (str "dirichlet(2,2,2): lp([0.3,0.3,0.3]) = -Inf (sum 0.9, was "
                    "finite +1.18)")
               (= (lpv peaked [0.3 0.3 0.3]) NEG-INF))
  ;; (5) interior points are bit-for-bit what they always were
  (assert-true "dirichlet(2,2,2): lp([0.2,0.3,0.5]) unchanged (1.28093338)"
               (< (js/Math.abs (- (lpv peaked [0.2 0.3 0.5]) 1.2809333801269531))
                  1e-6))
  (assert-true "dirichlet(1,1,1): lp([0.2,0.3,0.5]) unchanged (log 2)"
               (< (js/Math.abs (- (lpv uniform [0.2 0.3 0.5]) (js/Math.log 2)))
                  1e-5))
  (assert-true "dirichlet(.5,.5): lp([0.2,0.8]) unchanged (-0.22843921)"
               (< (js/Math.abs (- (lpv (dist/dirichlet [0.5 0.5]) [0.2 0.8])
                                  -0.22843921184539795))
                  1e-6))
  ;; (6) the FD probe step must stay inside the support: gradient_fd_test
  ;;     finite-differences d/dv at h=1e-3, which walks one component off the
  ;;     simplex by design. The mask tolerance has to clear that.
  (assert-true "dirichlet: v perturbed by the h=1e-3 FD step stays in support"
               (every? #(js/isFinite (lpv peaked %))
                       [[0.501 0.3 0.2] [0.499 0.3 0.2]
                        [0.5 0.301 0.2] [0.5 0.3 0.201]])))

;; the constructor guard
(assert-true "dirichlet: a non-positive alpha component throws at construction"
             (try (dist/dirichlet [1.0 -1.0]) false
                  (catch :default _ true)))
(assert-true "dirichlet: alpha = 0 throws too"
             (try (dist/dirichlet [1.0 0.0]) false
                  (catch :default _ true)))
(assert-true "dirichlet: a legal alpha still constructs"
             (some? (dist/dirichlet [1.0 2.0])))

;; The reachability leg: our OWN batch sampler used to emit exact 0.0
;; components for small alpha, feeding the NaN/+Inf above straight into a
;; vectorized weight. Measured before the floor: min component = 0.0.
(println "\n-- dirichlet batch sampler stays in the open simplex --")
(let [alphas [[0.05 0.05 0.05] [1.0 1.0 1.0] [0.5 2.0 0.1]]]
  (doseq [a alphas]
    (let [d  (dist/dirichlet a)
          s  (dc/dist-sample-n* d (rng/fresh-key 7) 2000)
          mn (mx/realize (mx/amin s))
          ;; sum over the K axis must still be 1 for every row
          worst-sum (mx/realize
                     (mx/amax (mx/abs (mx/subtract (mx/sum s [-1])
                                                   (mx/scalar 1.0)))))
          lps (mx/->clj (dc/dist-log-prob d s))]
      (assert-true (str "dirichlet" (pr-str a) ": no zero component in 2000x"
                        (count a) " draws (min " mn ")")
                   (> mn 0.0))
      (assert-true (str "dirichlet" (pr-str a) ": every row still sums to 1"
                        " (worst |sum-1| " worst-sum ")")
                   (< worst-sum 1e-4))
      (assert-true (str "dirichlet" (pr-str a) ": no NaN log-prob over its own"
                        " 2000 draws")
                   (not-any? js/isNaN lps))
      (assert-true (str "dirichlet" (pr-str a) ": every own draw is in support")
                   (every? js/isFinite lps)))))

;; ===========================================================================
;; genmlx-4x5w / genmlx-g2iu — truncated-normal: bounds mask AND far tail.
(println "\n-- truncated-normal: bounds mask + far-tail normalizer (genmlx-g2iu) --")

;; what IS guarded: values outside [lo,hi] score -Inf.
(check-dist "truncated-normal(0,1,-2,2)" (dist/truncated-normal 0 1 -2 2)
            [-2.5 2.5 -10 10] [-1.9 0 1.9])

;; The defect: the normalizer WAS 0.5(1+erf(b/sqrt2)) - 0.5(1+erf(a/sqrt2)).
;; Beyond ~5-6 sigma both terms round to 1.0 in float32, the difference
;; collapsed onto a 1e-38 clamp, and because log-norm is SUBTRACTED the result
;; was a large POSITIVE bonus — a spurious attractor for any chain moving mu
;; into the tail. `dist/log-normal-mass` now reflects onto one tail and works in
;; log space, so nothing ever rounds to 1.0. Reference values below are from an
;; independent float64 implementation (Mills-ratio continued fraction for the
;; tail, erf Taylor series near the origin) cross-checked against the +/-
;; mirror symmetry.
;;
;; Reinstating the old formula reproduces the exact defect these rows kill:
;; [7,8] -> log-norm -87.498 instead of -27.385, i.e. lp(7.1) = +61.374.
(let [tol 2e-3]
  (assert-true (str "truncated-normal(0,1,7,8) lp(7.1) = +1.2609, not the old +61.374 "
                    "(got " (lp (dist/truncated-normal 0 1 7 8) 7.1) ")")
               (< (js/Math.abs (- (lp (dist/truncated-normal 0 1 7 8) 7.1) 1.2608555)) tol))
  (assert-true (str "truncated-normal(0,0.1,1,2) lp(1.0) = +4.6149, not the old +38.882 "
                    "(got " (lp (dist/truncated-normal 0 0.1 1 2) 1.0) ")")
               (< (js/Math.abs (- (lp (dist/truncated-normal 0 0.1 1 2) 1.0) 4.6149316)) tol))
  ;; SIBLING the original report missed: the NEGATIVE tail was equally broken,
  ;; and by symmetry must now give the identical value.
  (assert-true (str "truncated-normal(0,1,-8,-7) lp(-7.1) mirrors the +tail exactly "
                    "(got " (lp (dist/truncated-normal 0 1 -8 -7) -7.1) ")")
               (< (js/Math.abs (- (lp (dist/truncated-normal 0 1 -8 -7) -7.1) 1.2608555)) tol))
  (assert-true "truncated-normal: +/- tails agree to float32 (mirror symmetry)"
               (< (js/Math.abs (- (lp (dist/truncated-normal 0 1 7 8) 7.1)
                                  (lp (dist/truncated-normal 0 1 -8 -7) -7.1)))
                  1e-5))
  ;; a narrow interval with its lower bound exactly at mu — the erf branch must
  ;; stay cancellation-free there. log(Phi(1e-6)-Phi(0)) = log(3.9894e-7).
  (assert-true (str "truncated-normal(0,1,0,1e-6) lp(0) = +13.8155 = -0.9189 + 14.7344 "
                    "(got " (lp (dist/truncated-normal 0 1 0 1e-6) 0) ")")
               (< (js/Math.abs (- (lp (dist/truncated-normal 0 1 0 1e-6) 0)
                                  (- 14.734449 0.9189385)))
                  1e-2))
  ;; the near-tail region float32 always resolved must be unchanged:
  ;; log N(0;0,1) - log(Phi(2)-Phi(-2)) = -0.9189385 - log(0.9544997) = -0.8723565
  (assert-true "truncated-normal(0,1,-2,2): lp(0) is still the correct -0.8723565"
               (< (js/Math.abs (- (lp (dist/truncated-normal 0 1 -2 2) 0)
                                  -0.8723565))
                  1e-4)))

;; POSITIVE ORACLE (CLAUDE.md rule 2): pinned constants alone would survive a
;; plausible-constant regression in the normalizer. This does not — it integrates
;; the density over [lo,hi] and demands 1. A wrong normalizer is a multiplicative
;; error on the whole integral, so any drift shows up directly.
(println "\n-- truncated-normal: the density integrates to 1 on every branch --")
(doseq [[label mu sigma lo hi n]
        [["central (0,1,-2,2)"      0 1.0   -2   2   20001]
         ["far + tail (0,1,7,8)"    0 1.0    7   8   20001]
         ["far - tail (0,1,-8,-7)"  0 1.0   -8  -7   20001]
         ["narrow far (0,0.1,1,2)"  0 0.1    1   2  100001]
         ["one-sided (0,1,0,4)"     0 1.0    0   4   20001]
         ["offset (5,2,0,10)"       5 2.0    0  10   20001]]]
  (let [dv   (/ (- hi lo) (dec n))
        grid (mapv #(+ lo (* % dv)) (range n))
        lps  (mx/->clj (dc/dist-log-prob (dist/truncated-normal mu sigma lo hi)
                                         (mx/array (clj->js grid))))
        ;; trapezoid rule, summed in float64 on the host
        ps   (mapv js/Math.exp lps)
        area (* dv (+ (* 0.5 (+ (first ps) (last ps)))
                      (reduce + (subvec ps 1 (dec (count ps))))))]
    (assert-true (str "truncated-normal " label ": integral = 1 (got " (.toFixed area 6) ")")
                 (< (js/Math.abs (- area 1.0)) 2e-3))))

;; the NaN class specifically: no bounded-support dist may return NaN
(println "\n-- no NaN from any bounded-support family, in or out of domain --")
(let [cases [["gamma" (dist/gamma-dist 2.0 1.5) [-3 -1 0 0.5 2]]
             ["inv-gamma" (dist/inv-gamma 2.0 1.5) [-3 -1 0 0.5 2]]
             ["log-normal" (dist/log-normal 0.0 1.0) [-3 -1 0 0.5 2]]
             ["beta" (dist/beta-dist 2.0 3.0) [-1 -0.1 0.5 1.1 2]]
             ["geometric" (dist/geometric 0.3) [-2 -0.5 0.5 3]]
             ["poisson" (dist/poisson 3.0) [-2 -0.5 1.5 4]]
             ["binomial" (dist/binomial 10 0.4) [-2 0.5 11 4]]
             ["neg-binomial" (dist/neg-binomial 5 0.4) [-2 0.5 3]]
             ["bernoulli" (dist/bernoulli 0.4) [-1 0.5 2 1]]
             ["discrete-uniform" (dist/discrete-uniform 0 10) [-2 -0.5 3.5 5 11]]
             ["beta(1,3) endpoints" (dist/beta-dist 1.0 3.0) [0 1 0.5]]
             ["beta(2,1) endpoints" (dist/beta-dist 2.0 1.0) [0 1 0.5]]
             ["truncated-normal" (dist/truncated-normal 0 1 -2 2) [-3 -2 0 2 3]]]]
  (doseq [[nm d vs] cases]
    (assert-true (str nm ": no NaN over " (pr-str vs))
                 (not-any? #(js/isNaN (lp d %)) vs))))

(let [vs [[0.0 0.5 0.5] [-0.1 0.6 0.5] [0.3 0.3 0.3] [0.2 0.3 0.5] [1.0 0.0 0.0]]]
  (doseq [a [[1.0 1.0 1.0] [0.5 0.5 0.5] [2.0 2.0 2.0]]]
    (let [d (dist/dirichlet a)]
      (assert-true (str "dirichlet" (pr-str a) ": no NaN over " (pr-str vs))
                   (not-any? #(js/isNaN (lpv d %)) vs)))))

;; ===========================================================================
;; categorical (genmlx-bey6) — found by the genmlx-4x5w sibling sweep, and the
;; most severe instance of this class: before the guard, an OUT-OF-RANGE index
;; scored 0, the maximum possible log-probability, so it outscored every legal
;; index. categorical is on the LLM-token, HMM-transition and
;; vectorized-mixture hot paths.
;; ===========================================================================

(println "\n-- categorical --")

(let [d (dist/categorical (mx/array [-1.0 -2.0 -0.5]))]
  ;; the three defects, each measured before the fix
  (assert-true (str "categorical: lp(7) = -Inf, was 0 — an out-of-range index "
                    "must not outscore the legal support (got " (lp d 7) ")")
               (= ##-Inf (lp d 7)))
  (assert-true (str "categorical: lp(-1) = -Inf, was -0.604 (silent wrap to the "
                    "last index) (got " (lp d -1) ")")
               (= ##-Inf (lp d -1)))
  (assert-true (str "categorical: lp(1.7) = -Inf, was lp(1) (silent int32 "
                    "truncation) (got " (lp d 1.7) ")")
               (= ##-Inf (lp d 1.7)))
  ;; the discriminator the truncation bug would pass: lp(1.7) must not merely
  ;; be finite-and-different, it must not EQUAL lp(1)
  (assert-true "categorical: lp(1.7) is NOT lp(1)" (not= (lp d 1.7) (lp d 1)))
  ;; false-negatives-only: every legal index is untouched
  (assert-true "categorical: lp(0) finite" (js/isFinite (lp d 0)))
  (assert-true "categorical: lp(1) finite" (js/isFinite (lp d 1)))
  (assert-true "categorical: lp(2) finite" (js/isFinite (lp d 2)))
  (assert-true "categorical: legal lps still sum to 1"
               (< (js/Math.abs (- 1.0 (reduce + (map #(js/Math.exp (lp d %)) [0 1 2])))) 1e-5)))

;; The per-particle [B,K] branch is the busiest one (HMM transitions,
;; genmlx-ql6a). Guarding only the 1-D branch would have left it open.
(let [d (dist/categorical (mx/array [[-1.0 -2.0 -0.5] [-0.2 -3.0 -1.0]]))
      legal   (mx/->clj (dc/dist-log-prob d (mx/array [0 2] mx/int32)))
      illegal (mx/->clj (dc/dist-log-prob d (mx/array [5 2] mx/int32)))]
  (assert-true (str "categorical [B,K]: legal [0 2] both finite (got " legal ")")
               (every? js/isFinite legal))
  (assert-true (str "categorical [B,K]: illegal lane -> -Inf, legal lane untouched "
                    "(got " illegal ")")
               (and (= ##-Inf (nth illegal 0))
                    (js/isFinite (nth illegal 1))
                    (< (js/Math.abs (- (nth illegal 1) (nth legal 1))) 1e-6))))

;; ===========================================================================
(println (str "\n== support-guard: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
