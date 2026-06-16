(ns bench.agents-bounds-recovery
  "Continuous resource-bound recovery + calibration over a bounded-rational agent
   (bean genmlx-6l31 / paper-side topml-wvku). The agents-axis 'diamond': the
   bounded-rationality knobs of an MDP agent are CONTINUOUS traced latents, and
   inverting the agent GF over them recovers them only up to an IDENTIFIABILITY
   RIDGE — yet the Bayesian inference is calibrated, which SBC confirms.

   The agent is a single-goal corridor planner. Two continuous bounds:
     alpha  = soft-max rationality (inverse temperature)
     u      = the goal's utility magnitude
   The policy logits are alpha * Q, and Q scales with u, so the decisiveness of a
   trajectory is governed by the PRODUCT alpha*u. Many (alpha, u) pairs along the
   hyperbola alpha*u = const produce the same policy and hence the same trajectory
   likelihood. That curved band is the ridge (differentiable.cljs names it: 'a
   multiplicative interaction with alpha, alpha*Q enters the policy').

   THREE artifacts, one per done-mean:

   1. RECOVERY to likelihood-equivalence (not point-identity). Adam through the
      differentiable planner (genmlx.agents.differentiable/recover-params) on a
      planted instance: loss(recovered) <= loss(plant) + eps. The individual
      (alpha, u) drift along the ridge; the conserved product alpha*u is recovered.

   2. The (alpha, u) POSTERIOR HEATMAP exposing the ridge. The gridworld geometry
      is fixed, so we precompute the per-cell log-policy table over an (alpha, u)
      grid ONCE; the exact log-posterior of a planted instance is then a table
      lookup. The high-posterior band is the ridge.

   3. SBC ranks over the bounds. With the same precomputed table, simulation-based
      calibration is exact-discrete and table-driven (no value iteration in the
      loop): draw (alpha*, u*) ~ uniform over the grid, roll the agent out, read
      the exact grid posterior, sample it, rank the truth. Uniform ranks (chi^2
      GOF) certify the inference is calibrated. This is the calibration analog of
      'the handler is ground truth': even where point-identification fails along
      the ridge, the posterior is honest.

   Deterministic: a local mulberry32 PRNG seeded by integer, MLX RNG unused for the
   host rollout, so the artifact is bit-reproducible.

   Output: results/agents-bounds-recovery/data.json
   Usage:  bun run --bun nbb bench/agents_bounds_recovery.cljs
           GENMLX_BENCH_QUICK=1 bun run --bun nbb bench/agents_bounds_recovery.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.agents.differentiable :as diff]))

;; ---------------------------------------------------------------------------
;; Infrastructure (matches sibling bench/agents_pluggable_inference.cljs)
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/agents-bounds-recovery")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

(def QUICK? (= "1" (aget (.-env js/process) "GENMLX_BENCH_QUICK")))

;; ---------------------------------------------------------------------------
;; Deterministic host PRNG (mulberry32) — bit-reproducible, no MLX RNG for the
;; host rollout. Script-local atom (benches are scripts; src/ purity unaffected).
;; ---------------------------------------------------------------------------

(defn make-rng [seed]
  (let [s (atom (bit-or (int seed) 0))]
    (fn []
      (swap! s (fn [x] (bit-or (+ x 0x6D2B79F5) 0)))
      (let [t0 @s
            t1 (js/Math.imul (bit-xor t0 (unsigned-bit-shift-right t0 15)) (bit-or t0 1))
            t2 (bit-xor t1 (+ t1 (js/Math.imul (bit-xor t1 (unsigned-bit-shift-right t1 7))
                                               (bit-or t1 61))))]
        (/ (unsigned-bit-shift-right (bit-xor t2 (unsigned-bit-shift-right t2 14)) 0)
           4294967296.0)))))

;; ---------------------------------------------------------------------------
;; The agent: a single-goal corridor. Bounds = (alpha, u).
;; ---------------------------------------------------------------------------

(def CORRIDOR [[:empty :empty :empty :empty :empty :empty :empty :empty :empty :G]])
(def TIME-COST -0.05)
;; gamma < 1 is essential: with gamma = 1 and a reachable goal, the utility u is a
;; constant offset to every action's Q at a state and CANCELS in the softmax, so
;; the policy depends only on path-length (time cost). Discounting makes staying /
;; detouring cost (1-gamma)*u, so alpha*u genuinely governs decisiveness -> ridge.
(def GAMMA 0.9)
(def N-ITERS 25)

(def dmdp (diff/build-diff-mdp {:grid CORRIDOR :goals [:G] :start [0 0]
                                :gamma GAMMA :noise 0.0}))

(defn log-policy-at
  "[S,A] host nested vector of log pi(a|s) at bounds (alpha, u). Soft value
   iteration (alpha enters the backup AND the policy), then log_softmax(alpha*Q)."
  [alpha u]
  (let [theta-u (mx/array #js [u] mx/float32)
        Q       (diff/diff-q dmdp theta-u TIME-COST alpha N-ITERS)   ; [S,A] lazy
        logp    (mx/log-softmax (mx/multiply alpha Q) 1)]            ; [S,A]
    (mapv vec (mx/->clj logp))))

;; ---------------------------------------------------------------------------
;; Grids (log-spaced over the positive bounds; uniform = log-uniform prior).
;; ---------------------------------------------------------------------------

(defn logspace [lo hi n]
  (let [a (js/Math.log lo) b (js/Math.log hi)]
    (mapv (fn [i] (js/Math.exp (+ a (* (/ i (dec n)) (- b a))))) (range n))))

(def NA (if QUICK? 14 28))
(def NU (if QUICK? 14 28))
(def ALPHA-GRID (logspace 0.3 12.0 NA))
(def U-GRID     (logspace 0.3 8.0  NU))

(println (str "  precomputing " NA "x" NU " log-policy table ..."))
(def t-pre (js/performance.now))
;; LP[i][j] = [S][A] log-policy at (ALPHA-GRID[i], U-GRID[j]).
(def LP
  (mapv (fn [alpha]
          (mapv (fn [u] (log-policy-at alpha u)) U-GRID))
        ALPHA-GRID))
(println (str "  ... table ready in "
              (.toFixed (/ (- (js/performance.now) t-pre) 1000) 2) "s"))

(def S (:S dmdp))
(def A (:A dmdp))
(def ns-fn (:ns-fn dmdp))
(def start-idx (:start-idx dmdp))
(def terminals (:terminals dmdp))
(defn terminal? [s] (contains? terminals s))

;; ---------------------------------------------------------------------------
;; Host helpers: categorical sampling, rollout, grid posterior.
;; ---------------------------------------------------------------------------

(defn logps->probs [logps]
  (let [m (apply max logps)
        es (mapv #(js/Math.exp (- % m)) logps)
        z  (reduce + es)]
    (mapv #(/ % z) es)))

(defn sample-cat [probs u]
  (loop [i 0 acc 0.0]
    (let [acc' (+ acc (nth probs i))]
      (if (or (>= acc' u) (= i (dec (count probs))))
        i
        (recur (inc i) acc')))))

(defn rollout
  "One trajectory of (s,a) pairs under the policy at grid cell (i,j)."
  [i j rng max-steps]
  (let [cell (get-in LP [i j])]
    (loop [s start-idx steps 0 obs []]
      (if (or (>= steps max-steps) (terminal? s))
        obs
        (let [a  (sample-cat (logps->probs (nth cell s)) (rng))
              s' (ns-fn s a)]
          (recur s' (inc steps) (conj obs [s a])))))))

(defn obs-loglik-grid
  "Flat NA*NU vector: Sigma_{(s,a) in obs} LP[i][j][s][a]. Pure table lookup."
  [obs]
  (vec (for [i (range NA) j (range NU)]
         (let [cell (get-in LP [i j])]
           (reduce (fn [acc [s a]] (+ acc (get-in cell [s a]))) 0.0 obs)))))

(defn loglik->posterior
  "Flat posterior over grid cells from flat log-likelihood (uniform prior)."
  [lls]
  (let [m (apply max lls)
        es (mapv #(js/Math.exp (- % m)) lls)
        z  (reduce + es)]
    (mapv #(/ % z) es)))

(defn cell->ij [c] [(quot c NU) (rem c NU)])

(defn sample-cells
  "Sample L flat cell indices from a flat posterior by inverse CDF."
  [posterior L rng]
  (let [cum (vec (reductions + posterior))]
    (vec (repeatedly L
                     (fn []
                       (let [u (rng)]
                         (loop [k 0]
                           (if (or (>= (nth cum k) u) (= k (dec (count cum))))
                             k
                             (recur (inc k))))))))))

;; ---------------------------------------------------------------------------
;; Done-mean 2 — the (alpha, u) posterior heatmap exposing the ridge.
;; ---------------------------------------------------------------------------

(defn build-heatmap [true-alpha true-u n-traj max-steps seed]
  (let [rng    (make-rng seed)
        ;; nearest grid cell to the planted bounds
        i*     (apply min-key #(js/Math.abs (- (nth ALPHA-GRID %) true-alpha)) (range NA))
        j*     (apply min-key #(js/Math.abs (- (nth U-GRID %) true-u)) (range NU))
        obs    (vec (mapcat (fn [_] (rollout i* j* rng max-steps)) (range n-traj)))
        lls    (obs-loglik-grid obs)
        post   (loglik->posterior lls)
        ;; reshape flat -> NA x NU
        post2d (mapv (fn [i] (mapv (fn [j] (nth post (+ (* i NU) j))) (range NU))) (range NA))
        ll2d   (mapv (fn [i] (mapv (fn [j] (nth lls  (+ (* i NU) j))) (range NU))) (range NA))
        marg-a (mapv (fn [i] (reduce + (nth post2d i))) (range NA))
        marg-u (mapv (fn [j] (reduce + (mapv #(nth % j) post2d))) (range NU))
        map-c  (apply max-key #(nth post %) (range (count post)))
        [mi mj] (cell->ij map-c)
        ;; ridge diagnostic: posterior-weighted correlation of log(alpha) & log(u)
        cells  (for [i (range NA) j (range NU)] [i j (nth post (+ (* i NU) j))])
        la     (fn [i] (js/Math.log (nth ALPHA-GRID i)))
        lu     (fn [j] (js/Math.log (nth U-GRID j)))
        ma     (reduce + (map (fn [[i _ w]] (* w (la i))) cells))
        mu     (reduce + (map (fn [[_ j w]] (* w (lu j))) cells))
        cov    (reduce + (map (fn [[i j w]] (* w (- (la i) ma) (- (lu j) mu))) cells))
        va     (reduce + (map (fn [[i _ w]] (* w (let [d (- (la i) ma)] (* d d)))) cells))
        vu     (reduce + (map (fn [[_ j w]] (* w (let [d (- (lu j) mu)] (* d d)))) cells))
        corr   (/ cov (max 1e-12 (js/Math.sqrt (* va vu))))]
    {:true_alpha true-alpha :true_u true-u :true_product (* true-alpha true-u)
     :n_obs (count obs)
     :alpha_grid ALPHA-GRID :u_grid U-GRID
     :log_posterior ll2d :posterior post2d
     :marginal_alpha marg-a :marginal_u marg-u
     :map {:alpha (nth ALPHA-GRID mi) :u (nth U-GRID mj) :product (* (nth ALPHA-GRID mi) (nth U-GRID mj))}
     :ridge {:logalpha_logu_corr corr
             :note "posterior-weighted Corr(log alpha, log u); strongly negative = the alpha*u ridge"}}))

;; ---------------------------------------------------------------------------
;; Done-mean 3 — SBC ranks over the bounds (exact-discrete, table-driven).
;; ---------------------------------------------------------------------------

(defn randomized-rank
  "Rank of the truth among L posterior samples on a discrete coordinate, with an
   independent U(0,1) tie-break per sample and per truth -> uniform under H0."
  [true-coord sample-coords rng]
  (let [v-true (rng)]
    (reduce (fn [r ci]
              (let [v (rng)]
                (cond (< ci true-coord)                  (inc r)
                      (and (= ci true-coord) (< v v-true)) (inc r)
                      :else r)))
            0 sample-coords)))

(defn histogram [ranks n-bins L]
  (let [span (inc L)]                                  ; ranks in 0..L
    (reduce (fn [cs r] (let [b (min (dec n-bins) (int (* n-bins (/ r span))))]
                         (update cs b inc)))
            (vec (repeat n-bins 0)) ranks)))

;; chi^2 GOF p-value via the regularized upper incomplete gamma Q(a,x).
(defn- gammln [x]
  (let [cof [76.18009172947146 -86.50532032941677 24.01409824083091
             -1.231739572450155 0.1208650973866179e-2 -0.5395239384953e-5]
        ser (atom 1.000000000190015)
        y (atom x)
        tmp (+ x 5.5)
        tmp2 (- tmp (* (+ x 0.5) (js/Math.log tmp)))]
    (doseq [c cof] (swap! y inc) (swap! ser + (/ c @y)))
    (+ (- tmp2) (js/Math.log (* 2.5066282746310005 (/ @ser x))))))

(defn- gammq [a x]
  (cond
    (<= x 0.0) 1.0
    (< x (+ a 1.0))                                     ; series for P, Q = 1-P
    (let [gln (gammln a)]
      (loop [ap a sum (/ 1.0 a) del (/ 1.0 a) n 0]
        (if (or (> n 200) (< (js/Math.abs del) (* (js/Math.abs sum) 1e-12)))
          (- 1.0 (* sum (js/Math.exp (+ (- x) (* a (js/Math.log x)) (- gln)))))
          (let [ap' (inc ap) del' (* del (/ x ap'))]
            (recur ap' (+ sum del') del' (inc n))))))
    :else                                              ; continued fraction for Q
    (let [gln (gammln a) tiny 1e-30]
      (loop [i 1 b (+ x 1.0 (- a)) c (/ 1.0 tiny) d (/ 1.0 (+ x 1.0 (- a))) h (/ 1.0 (+ x 1.0 (- a)))]
        (if (> i 200)
          (* (js/Math.exp (+ (- x) (* a (js/Math.log x)) (- gln))) h)
          (let [an (* (- i) (- i a))
                b' (+ b 2.0)
                d0 (+ (* an d) b')
                d1 (if (< (js/Math.abs d0) tiny) tiny d0)
                c0 (+ b' (/ an c))
                c1 (if (< (js/Math.abs c0) tiny) tiny c0)
                d2 (/ 1.0 d1)
                del (* d2 c1)]
            (recur (inc i) b' c1 d2 (* h del))))))))

(defn chi2-sf [stat df] (gammq (/ df 2.0) (/ stat 2.0)))

(defn chi2-uniformity [counts n-sims]
  (let [n-bins (count counts)
        exp    (/ n-sims n-bins)
        stat   (reduce + (map (fn [o] (/ (* (- o exp) (- o exp)) exp)) counts))
        df     (dec n-bins)]
    {:counts counts :expected exp :chi2 stat :df df
     :p_value (chi2-sf stat df)}))

(defn run-sbc [n-sims L n-bins n-traj max-steps base-seed]
  (println (str "  SBC: " n-sims " sims, L=" L " posterior draws, " n-traj " traj/sim ..."))
  (let [rng (make-rng base-seed)]
    (loop [k 0 alpha-ranks [] u-ranks [] n-obs-acc 0]
      (if (>= k n-sims)
        (let [a-hist (histogram alpha-ranks n-bins L)
              u-hist (histogram u-ranks n-bins L)]
          {:n_sims n-sims :n_posterior_draws L :n_bins n-bins
           :traj_per_sim n-traj :max_steps max-steps
           :mean_obs_per_sim (/ n-obs-acc n-sims)
           :alpha (assoc (chi2-uniformity a-hist n-sims) :ranks alpha-ranks)
           :u     (assoc (chi2-uniformity u-hist n-sims) :ranks u-ranks)})
        (let [;; draw truth uniformly over the grid (log-uniform prior)
              i* (int (* (rng) NA))
              j* (int (* (rng) NU))
              i* (min i* (dec NA)) j* (min j* (dec NU))
              obs (vec (mapcat (fn [_] (rollout i* j* rng max-steps)) (range n-traj)))
              post (loglik->posterior (obs-loglik-grid obs))
              cells (sample-cells post L rng)
              a-idx (mapv #(first (cell->ij %)) cells)
              u-idx (mapv #(second (cell->ij %)) cells)
              ra (randomized-rank i* a-idx rng)
              ru (randomized-rank j* u-idx rng)]
          (recur (inc k) (conj alpha-ranks ra) (conj u-ranks ru)
                 (+ n-obs-acc (count obs))))))))

;; ---------------------------------------------------------------------------
;; Done-mean 1 — recovery to likelihood-equivalence (Adam through the planner).
;; ---------------------------------------------------------------------------

(defn run-recovery [plant-alpha plant-u n-traj max-steps seed iterations]
  (let [rng  (make-rng seed)
        i*   (apply min-key #(js/Math.abs (- (nth ALPHA-GRID %) plant-alpha)) (range NA))
        j*   (apply min-key #(js/Math.abs (- (nth U-GRID %) plant-u)) (range NU))
        ;; observations sampled from the EXACT planted continuous policy (not the grid)
        cell-lp (log-policy-at plant-alpha plant-u)
        roll (fn []
               (loop [s start-idx steps 0 obs []]
                 (if (or (>= steps max-steps) (terminal? s))
                   obs
                   (let [a  (sample-cat (logps->probs (nth cell-lp s)) (rng))
                         s' (ns-fn s a)]
                     (recur s' (inc steps) (conj obs [s a]))))))
        obs  (vec (mapcat (fn [_] (roll)) (range n-traj)))
        rec  (diff/recover-params dmdp TIME-COST N-ITERS obs
                                  {:iterations iterations :lr 0.05 :key (rng/fresh-key 7)})
        rec-u     (first (mapv identity (mx/->clj (:theta-u rec))))
        rec-alpha (:alpha rec)
        plant-loss (diff/loss-at dmdp TIME-COST [plant-u] (js/Math.log plant-alpha) N-ITERS obs)
        rec-loss   (diff/loss-at dmdp TIME-COST [rec-u] (js/Math.log rec-alpha) N-ITERS obs)
        n          (count obs)]
    {:n_obs n
     :plant {:alpha plant-alpha :u plant-u :product (* plant-alpha plant-u) :loss plant-loss}
     :recovered {:alpha rec-alpha :u rec-u :product (* rec-alpha rec-u) :loss rec-loss}
     :gap_total (- rec-loss plant-loss)
     :gap_per_obs (/ (- rec-loss plant-loss) n)
     :product_rel_err (js/Math.abs (/ (- (* rec-alpha rec-u) (* plant-alpha plant-u))
                                      (* plant-alpha plant-u)))
     :loss_history (:loss-history rec)}))

;; ---------------------------------------------------------------------------
;; Orchestration
;; ---------------------------------------------------------------------------

(defn -main []
  (let [t0 (js/performance.now)
        _  (println "\n=== Continuous (alpha, u) bound recovery + SBC (genmlx-6l31) ===")

        ;; Done-mean 1: recovery to likelihood-equivalence.
        _  (println "\n-- 1. recovery to likelihood-equivalence --")
        rec (run-recovery 3.0 4.0 (if QUICK? 10 24) 12 4242 (if QUICK? 250 600))
        eps 0.02                                        ; nats / observation
        rec-pass (<= (:gap_per_obs rec) eps)
        _  (println (str "     plant (a*u)=" (.toFixed (-> rec :plant :product) 3)
                         "  recovered (a*u)=" (.toFixed (-> rec :recovered :product) 3)
                         "  product rel-err=" (.toFixed (* 100 (:product_rel_err rec)) 1) "%"))
        _  (println (str "     loss/obs plant=" (.toFixed (/ (-> rec :plant :loss) (:n_obs rec)) 4)
                         " recovered=" (.toFixed (/ (-> rec :recovered :loss) (:n_obs rec)) 4)
                         "  gap/obs=" (.toFixed (:gap_per_obs rec) 5)
                         (if rec-pass "  PASS" "  FAIL")))

        ;; Done-mean 2: posterior heatmap exposing the ridge.
        _  (println "\n-- 2. (alpha, u) posterior heatmap --")
        heat (build-heatmap 2.5 4.0 (if QUICK? 12 30) 12 1234)
        ridge-corr (-> heat :ridge :logalpha_logu_corr)
        ridge-pass (< ridge-corr -0.5)
        _  (println (str "     n-obs=" (:n_obs heat)
                         "  MAP alpha=" (.toFixed (-> heat :map :alpha) 2)
                         " u=" (.toFixed (-> heat :map :u) 2)
                         " (a*u=" (.toFixed (-> heat :map :product) 2) ")"
                         "  ridge corr(log a, log u)=" (.toFixed ridge-corr 3)
                         (if ridge-pass "  PASS" "  FAIL")))

        ;; Done-mean 3: SBC ranks over the bounds.
        _  (println "\n-- 3. SBC over the bounds --")
        sbc (run-sbc (if QUICK? 40 150) (if QUICK? 49 99) 10
                     (if QUICK? 3 5) 12 9000)
        a-p (-> sbc :alpha :p_value)
        u-p (-> sbc :u :p_value)
        sbc-pass (and (> a-p 0.05) (> u-p 0.05))
        _  (println (str "     alpha ranks chi2=" (.toFixed (-> sbc :alpha :chi2) 2)
                         " (df " (-> sbc :alpha :df) ") p=" (.toFixed a-p 3)))
        _  (println (str "     u     ranks chi2=" (.toFixed (-> sbc :u :chi2) 2)
                         " (df " (-> sbc :u :df) ") p=" (.toFixed u-p 3)
                         (if sbc-pass "  PASS" "  FAIL")))

        elapsed (- (js/performance.now) t0)
        all-pass (and rec-pass ridge-pass sbc-pass)]

    (write-json "data.json"
      {:experiment "agents-bounds-recovery"
       :bean "genmlx-6l31"
       :description "Continuous (alpha, u) bound recovery to likelihood-equivalence, the (alpha,u) posterior ridge, and SBC calibration over the bounds for a bounded-rational corridor agent."
       :claim "The continuous resource bounds of an MDP agent (rationality alpha and utility scale u) are identifiable only up to the alpha*u ridge: gradient recovery reaches plant likelihood while (alpha, u) drift along the ridge, and the exact (alpha,u) posterior is a curved band. Yet inverting the agent GF over the bounds is CALIBRATED: SBC ranks are uniform (chi^2 GOF p > 0.05)."
       :model {:grid "1x8 corridor, single goal :G at the far end"
               :bounds {:alpha "soft-max rationality (inverse temperature), log-uniform [0.3,12]"
                        :u "goal utility magnitude, log-uniform [0.3,8]"}
               :time_cost TIME-COST :gamma GAMMA :n_iters N-ITERS
               :grid_resolution {:n_alpha NA :n_u NU}}
       :recovery rec
       :heatmap heat
       :sbc sbc
       :verdict {:recovery_pass rec-pass :recovery_eps_per_obs eps
                 :ridge_pass ridge-pass :ridge_threshold -0.5
                 :sbc_pass sbc-pass :sbc_alpha_p a-p :sbc_u_p u-p
                 :all_pass all-pass}
       :config {:seed_scheme "local mulberry32 PRNG, fixed integer seeds (recovery 4242 / 7, heatmap 1234, SBC 9000); host rollout uses no MLX RNG, so bit-reproducible"
                :quick QUICK?}
       :duration_ms elapsed})

    (println (str "\n  Total: " (.toFixed (/ elapsed 1000) 1) "s   "
                  (if all-pass "ALL PASS" "FAIL")))
    (when-not all-pass
      (println "  FAIL: one or more done-means did not hold")
      (js/process.exit 1))))

;; Run only when invoked as a script (a test/require reuses the fns above without
;; triggering the full run).
(defn- run-as-script? []
  (boolean (some #(re-find #"agents_bounds_recovery\.cljs" (str %)) (vec (.-argv js/process)))))

(when (run-as-script?) (-main))
