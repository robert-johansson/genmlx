;; Headless tests for the BATCHED bandit inference path (bean genmlx-tl6p):
;; pomdp/simulate-bandit-batched runs N independent Thompson episodes at once as
;; [N,K]-shaped tensor steps. The batched aggregate (final posterior means, modal
;; best-arm fraction, mean cumulative regret) must match N independent host
;; simulate-bandit calls to DISTRIBUTIONAL tolerance (means over N — the RNG path
;; differs, so per-episode bit-exactness is not expected), and be materially faster
;; on a large operand.
;;
;; Run: bunx nbb@1.4.206 test/genmlx/bandit_batched_test.cljs

(ns genmlx.bandit-batched-test
  (:require [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as env]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol)
    (do (vswap! passed inc) (println " PASS" msg "  =" actual))
    (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))

(defn modal-arm [row k] (apply max-key #(get (frequencies row) % 0) (range k)))

;; ENV CONSTRAINT (genmlx-5ucd): under `bunx nbb` there is no exposed JS GC, so the
;; rejection-sampler scratch buffers from sequential HOST rollouts are never
;; reclaimed and the Metal buffer-COUNT limit (~499000) is hit at trivial memory
;; (~4MB) — an uncatchable crash after ~13 rollouts. That is precisely the pathology
;; the batched [N,K] path removes (one rollout, per-step materialize). So the host
;; cross-check here is run at a small feasible N; the PRIMARY validation is the
;; batched path against analytical ground truth (moments, mass conservation,
;; convergence to the true theta, sublinear regret) at full N.
;; Each host rollout leaves ~75k unreclaimable Metal buffers (60 steps × the
;; Marsaglia-Tsang gamma loop), so the whole process can do only ~6 rollouts before
;; the ~499k buffer-count limit (genmlx-5ucd). HOST-N=4 leaves room for the one
;; batched(256) rollout. The host cross-check is therefore a loose feasible-N sanity
;; tie; batched→analytical (Section 3b) is the real validation.
(def HOST-N 4)
(def BATCH-N 256)

(def THETAS [0.25 0.50 0.80])
(def BEST 2)
(def H 60)
(def bandit (env/bandit-pomdp {:thetas THETAS :horizon H}))
(def ag (pomdp/make-bandit-agent {:strategy :thompson}))

(println "\n== Section 1: [N,K] Beta sampler moments (distinct per-row shapes) ==")
(let [N 20000
      av (mx/array (clj->js (vec (repeat N [2.0 5.0 8.0]))) mx/float32)
      bv (mx/array (clj->js (vec (repeat N [8.0 5.0 2.0]))) mx/float32)
      means (vec (mx/->clj (mx/mean (dist/beta-sample-vec av bv (rng/fresh-key 7)) [0])))]
  (println "  column means:" (mapv #(.toFixed % 3) means))
  (doseq [[m e] (map vector means [0.2 0.5 0.8])]
    (assert-close "[N,K] Beta column mean == alpha/(alpha+beta)" e m 0.02)))

;; --- Run the HOST reference (N=8, timed) and the BATCHED path (N=256, timed) ONCE ---
;; (Both reused across Sections 2-5; total stays within the buffer-count budget.)
(def host-t0 (js/Date.now))
(def host-stats
  (mapv (fn [k]
          (let [r (pomdp/simulate-bandit ag bandit k)]
            {:best (nth ((:arm-values ag) (last (:beliefs r))) BEST)
             :modal (modal-arm (:arms r) 3)
             :reg (last (:regret r))
             :npulls (count (:arms r))}))
        (rng/split-n (rng/fresh-key 42) HOST-N)))
(def host-ms (- (js/Date.now) host-t0))
(def host-best-mean (/ (reduce + (map :best host-stats)) (double HOST-N)))
(def host-modal-frac (/ (count (filter #(= BEST (:modal %)) host-stats)) (double HOST-N)))
(def host-reg-mean (/ (reduce + (map :reg host-stats)) (double HOST-N)))

(def bat-t0 (js/Date.now))
(def bat (pomdp/simulate-bandit-batched bandit BATCH-N (rng/fresh-key 42)))
(mx/materialize! (:final-means bat) (:regret bat) (:arms bat) (:cum-reward bat))
(def bat-ms (- (js/Date.now) bat-t0))
(def bat-arms (mx/->clj (:arms bat)))
(def bat-best-mean (mx/item (mx/mean (mx/idx (:final-means bat) BEST 1))))      ; mean over N
(def bat-modal-frac (/ (count (filter (fn [row] (= BEST (modal-arm row 3))) bat-arms)) (double BATCH-N)))
(def bat-step-mean (vec (mx/->clj (mx/mean (:regret bat) [0]))))                 ; [H] mean cum regret
(def bat-reg-mean (last bat-step-mean))

(println "\n== Section 2: conjugate-increment exactness (mass conservation) ==")
;; The one-hot-masked increment must add exactly one pull per step (no mass lost or
;; duplicated) and the reward channel must stay in {0,1}.
(assert-true "every batched episode ran exactly H pulls" (every? #(= H (count %)) bat-arms))
(assert-true "every host episode ran exactly H pulls" (every? #(= H (:npulls %)) host-stats))
(assert-true "cumulative reward per episode in [0,H]"
             (every? #(<= 0 % H) (mx/->clj (:cum-reward bat))))

(println "\n== Section 3: batched aggregate matches host + converges to analytical ==")
(println (str "  best-arm posterior mean  host(" HOST-N "): " (.toFixed host-best-mean 3) "  batched(" BATCH-N "): " (.toFixed bat-best-mean 3)))
(println (str "  modal-best fraction      host(" HOST-N "): " (.toFixed host-modal-frac 3) "  batched(" BATCH-N "): " (.toFixed bat-modal-frac 3)))
(println (str "  mean cumulative regret   host(" HOST-N "): " (.toFixed host-reg-mean 2) "  batched(" BATCH-N "): " (.toFixed bat-reg-mean 2)))
;; (a) host(feasible N) vs batched(256) — loose direction-agreement tie (the host
;;     aggregate is noisy at the small feasible N; best-arm mean is the most
;;     concentrated statistic, regret mean looser, modal-frac too noisy to cross-check).
(assert-close "best-arm posterior mean agrees with host (feasible N, 0.1)" host-best-mean bat-best-mean 0.1)
(assert-close "mean cumulative regret agrees with host (feasible N, 0.2*H)" host-reg-mean bat-reg-mean (* 0.2 H))
;; (b) batched(256) converges to the analytical target (true theta of the best arm)
;;     and concentrates pulls on it — the primary, full-N validation.
(assert-close "batched best-arm posterior mean -> true theta 0.8" 0.80 bat-best-mean 0.05)
(assert-true  "batched modal pull is the best arm in a clear majority of episodes" (> bat-modal-frac 0.7))
(assert-true  "host modal pull is the best arm in a majority of (feasible-N) episodes" (>= host-modal-frac 0.5))

(println "\n== Section 4: sublinear regret in batched form (mean over N) ==")
(let [early (/ (nth bat-step-mean 9) 10.0)
      late  (/ (- (nth bat-step-mean (dec H)) (nth bat-step-mean 49)) 10.0)]
  (println "  mean-over-N regret: early-slope" (.toFixed early 3) " late-slope" (.toFixed late 3)
           " total" (.toFixed bat-reg-mean 2))
  (assert-true "regret accrues early"              (> early 0.0))
  (assert-true "regret slope flattens (sublinear)" (< late early))
  (assert-true "total regret below worst-arm bound" (< bat-reg-mean (* H (- 0.80 0.25)))))

(println "\n== Section 5: speed — per-episode wall-clock (batched vs host) ==")
;; Per-episode cost: host runs HOST-N episodes sequentially; batched runs BATCH-N at
;; once. (A 512-host-vs-batched comparison is infeasible in one process under the
;; buffer-count limit genmlx-5ucd; per-episode cost is the honest, feasible metric.)
(let [host-per-ep (/ host-ms (double HOST-N))
      bat-per-ep  (/ bat-ms  (double BATCH-N))]
  (println "  host:" host-ms "ms /" HOST-N "ep =" (.toFixed host-per-ep 2) "ms/ep   "
           "batched:" bat-ms "ms /" BATCH-N "ep =" (.toFixed bat-per-ep 3) "ms/ep   "
           "per-episode speedup:" (.toFixed (/ host-per-ep (max 0.001 bat-per-ep)) 1) "x")
  (assert-true "batched per-episode cost is >= 3x lower than the host path"
               (< bat-per-ep (/ host-per-ep 3.0))))

(println "\n== Section 6: scalar host path unchanged ==")
(assert-true "host simulate-bandit still returns H pulls per episode" (every? #(= H (:npulls %)) host-stats))

(println (str "\n== Results: " @passed " passed, " @failed " failed =="))
(when (pos? @failed) (js/process.exit 1))
