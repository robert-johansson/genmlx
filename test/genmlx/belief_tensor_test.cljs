;; @tier fast
;; Headless tests for the TENSOR belief-update kernel (bean genmlx-kpuo):
;; genmlx.agents.belief. The pure-MLX observation-Bayes filter b'=normalize(b⊙L)
;; must agree to float32 tolerance (1e-6) with the host Clojure-map filters
;; (pomdp.cljs normalize-logs and biased_planners.cljs bayes-update) on every demo,
;; both as a one-step kernel and threaded through simulate-pomdp /
;; simulate-biased-pomdp in :belief-mode :tensor. The host filters stay default
;; (:host) and remain ground truth.
;;
;; Run: bunx nbb@1.4.206 test/genmlx/belief_tensor_test.cljs

(ns genmlx.belief-tensor-test
  (:require [genmlx.agents.belief :as belief]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as env]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.mlx :as mx]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol)
    (do (vswap! passed inc) (println " PASS" msg "  =" actual))
    (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))

(defn maps-close?
  "Every world's prob in two {world->prob} beliefs agrees within tol."
  [m1 m2 tol]
  (and (= (set (keys m1)) (set (keys m2)))
       (every? (fn [k] (<= (Math/abs (- (double (m1 k)) (double (m2 k)))) tol)) (keys m1))))

(defn vecs-close? [v1 v2 tol]
  (and (= (count v1) (count v2))
       (every? (fn [[a b]] (<= (Math/abs (- (double a) (double b))) tol)) (map vector v1 v2))))

;; ===========================================================================
;; Fixture 1 — hidden-goal restaurant gridworld (worlds = [:A :B], signpost reveal)
;; ===========================================================================
(def grid [[:A    :empty :B]
           [:wall :empty :wall]
           [:wall :empty :wall]
           [:wall :empty :wall]
           [:wall :empty :wall]])
(def signpost 7)
(def goals [:A :B])
(def gw-env (env/restaurant-gridworld {:grid grid :goals goals :signpost signpost
                                       :true-world :A :start [1 4]}))
(def gw-agent (pomdp/make-pomdp-agent (assoc gw-env :alpha ##Inf :gamma 1.0 :n-iters 40)))
(def gw-worlds (:worlds gw-agent))             ; [:A :B]
(def gw-observe (:observe gw-env))
(def gw-prior (:prior gw-agent))               ; {:A 0.5 :B 0.5}

(println "\n== Section 1: obs-likelihood-vec (host geometry -> [W] indicator) ==")
(assert-equal "L at (signpost,:A) == [1 0]" [1.0 0.0]
              (vec (mx/->clj (belief/obs-likelihood-vec gw-observe gw-worlds signpost :A))))
(assert-equal "L at (signpost,:B) == [0 1]" [0.0 1.0]
              (vec (mx/->clj (belief/obs-likelihood-vec gw-observe gw-worlds signpost :B))))

(println "\n== Section 2: tensor kernel == host filter (restaurant gridworld) ==")
(let [ub-host (:update-belief gw-agent)]
  (doseq [[loc o] [[10 nil] [signpost :A] [signpost :B]]]
    (let [host   (ub-host gw-prior loc o)
          tens   (belief/update-belief-map gw-observe gw-worlds gw-prior loc o)]
      (assert-true (str "tensor == host at (loc " loc ", o " o ")") (maps-close? host tens 1e-6))
      (when o
        (assert-close (str "normalized at (loc " loc ", o " o ")")
                      1.0 (reduce + (vals tens)) 1e-6)))))

(println "\n== Section 3: nil identity + z=0 defensive (kernel) ==")
(let [b (belief/belief->vec gw-worlds gw-prior)]
  ;; nil obs -> bit-identical input belief (fast-path), no L built
  (assert-equal "nil obs -> belief unchanged (bit-equal)"
                (vec (mx/->clj b))
                (vec (mx/->clj (belief/tensor-update-belief gw-observe gw-worlds b signpost nil))))
  ;; impossible obs (:C, which no world produces at the signpost) -> keep b (mirrors
  ;; biased_planners.cljs (pos? z) guard; the host pomdp filter guards the same
  ;; way since genmlx-xpbm).
  (assert-true "impossible obs -> belief unchanged (z=0 defensive)"
               (vecs-close? (vec (mx/->clj b))
                            (vec (mx/->clj (belief/tensor-update-belief gw-observe gw-worlds b signpost :C)))
                            1e-6)))

(println "\n== Section 3b: genmlx-xpbm regressions (host guard + safe-where grad) ==")
;; HOST pomdp filter: impossible obs keeps belief unchanged, never NaN
;; (pre-fix: all log-weights -Inf -> normalize-logs exp(-Inf - -Inf) = NaN)
(let [ub-host (:update-belief gw-agent)
      out     (ub-host gw-prior signpost :C)]
  (assert-true "host filter: impossible obs -> belief unchanged (no NaN)"
               (maps-close? gw-prior out 1e-12))
  (assert-true "host filter: no NaN in output"
               (every? #(js/isFinite %) (vals out))))
;; host == tensor on the impossible-obs case (the equivalence claim is now true)
(let [ub-host (:update-belief gw-agent)
      b       (belief/belief->vec gw-worlds gw-prior)
      host    (ub-host gw-prior signpost :C)
      tens    (belief/vec->belief gw-worlds (belief/tensor-update-belief gw-observe gw-worlds b signpost :C))]
  (assert-true "host == tensor on impossible obs" (maps-close? host tens 1e-6)))
;; safe-where: gradient through a z=0 filter-step is finite (pre-fix: raw/0 in
;; the untaken where branch poisoned the gradient with NaN)
(let [L0   (mx/array #js [0.0 0.0] mx/float32)            ; impossible obs likelihood
      loss (fn [bv] (mx/sum (belief/filter-step bv L0)))
      g    ((mx/grad loss) (mx/array #js [0.5 0.5] mx/float32))]
  (assert-true "filter-step gradient finite at z=0 (safe-where)"
               (every? #(js/isFinite %) (vec (mx/->clj g)))))

(println "\n== Section 4: belief<->vec round-trip ==")
(doseq [[label m] [["uniform" gw-prior] ["skewed" {:A 0.8 :B 0.2}]]]
  (assert-true (str "round-trip preserves " label " belief")
               (maps-close? m (belief/vec->belief gw-worlds (belief/belief->vec gw-worlds m)) 1e-6)))

;; ===========================================================================
;; Fixture 2 — adjacency-reveal restaurant POMDP (2^2 worlds, partial reveal)
;; ===========================================================================
(def adj-grid [[:a     :empty :empty :b]
               [:empty :empty :empty :empty]
               [:empty :empty :empty :empty]])
(def adj-env (env/restaurant-pomdp {:grid adj-grid :utilities {:a 5.0 :b 3.0}
                                    :open-prob {:a 0.6 :b 0.9} :true-world {:a true :b true} :start [1 2]}))
(def adj-agent (pomdp/make-pomdp-agent (assoc adj-env :alpha ##Inf :gamma 1.0 :noise 0.0 :n-iters 40)))
(def adj-worlds (:worlds adj-agent))
(def adj-observe (:observe adj-env))
(def adj-prior (:prior adj-agent))
(defn P-b-open [m] (reduce + (map (fn [[w p]] (if (:b w) p 0.0)) m)))

(println "\n== Section 5: tensor kernel == host filter (adjacency POMDP, 4 worlds) ==")
(let [ub-host (:update-belief adj-agent)]
  (doseq [[loc o] [[5 nil] [1 [[:a true]]] [1 [[:a false]]] [2 [[:b true]]]]]
    (let [host (ub-host adj-prior loc o)
          tens (belief/update-belief-map adj-observe adj-worlds adj-prior loc o)]
      (assert-true (str "tensor == host at (loc " loc ", o " o ")") (maps-close? host tens 1e-6)))))
;; the discriminating partial-reveal: observing A pins A but leaves B at its prior 0.9
(let [tens (belief/update-belief-map adj-observe adj-worlds adj-prior 1 [[:a true]])]
  (assert-close "partial reveal: P(B-open) stays prior 0.9 after observing A" 0.9 (P-b-open tens) 1e-6))

;; ===========================================================================
;; Section 6 — full-rollout agreement: :belief-mode :host vs :tensor
;; ===========================================================================
(println "\n== Section 6: simulate-pomdp host-mode == tensor-mode ==")
(doseq [tw [:A :B]]
  (let [e     (env/restaurant-gridworld {:grid grid :goals goals :signpost signpost
                                         :true-world tw :start [1 4]})
        pa    (pomdp/make-pomdp-agent (assoc e :alpha ##Inf :gamma 1.0 :n-iters 40))
        host  (pomdp/simulate-pomdp pa e (:start-idx e) 12)
        tens  (pomdp/simulate-pomdp pa e (:start-idx e) 12 {:belief-mode :tensor})]
    (assert-equal (str "true " tw ": same :states")  (:states host) (:states tens))
    (assert-equal (str "true " tw ": same :actions") (:actions host) (:actions tens))
    (assert-true  (str "true " tw ": #beliefs == #states") (= (count (:beliefs tens)) (count (:states tens))))
    (assert-true  (str "true " tw ": per-step beliefs agree to 1e-6")
                  (every? (fn [[h t]] (maps-close? h t 1e-6)) (map vector (:beliefs host) (:beliefs tens))))))

(println "\n== Section 7: adjacency rollout host-mode == tensor-mode ==")
(doseq [tw [{:a true :b true} {:a false :b true}]]
  (let [e    (env/restaurant-pomdp {:grid adj-grid :utilities {:a 5.0 :b 3.0}
                                    :open-prob {:a 0.6 :b 0.9} :true-world tw :start [1 2]})
        pa   (pomdp/make-pomdp-agent (assoc e :alpha ##Inf :gamma 1.0 :noise 0.0 :n-iters 40))
        host (pomdp/simulate-pomdp pa e (:start-idx e) 14)
        tens (pomdp/simulate-pomdp pa e (:start-idx e) 14 {:belief-mode :tensor})]
    (assert-equal (str tw ": same :states") (:states host) (:states tens))
    (assert-equal (str tw ": same :observations") (:observations host) (:observations tens))
    (assert-true  (str tw ": per-step beliefs agree to 1e-6")
                  (every? (fn [[h t]] (maps-close? h t 1e-6)) (map vector (:beliefs host) (:beliefs tens))))))

;; ===========================================================================
;; Section 8 — biased POMDP rollout (voi-world, prob-vector belief)
;; ===========================================================================
(println "\n== Section 8: biased filter (voi-world) — tensor == host bayes-update ==")
;; The voi-world agent (alpha=Inf) has argmax action TIES, so two independent
;; rollouts can take different paths run-to-run — that is action stochasticity, not
;; a belief disagreement. We isolate the FILTER: take ONE host rollout's
;; (state', observation) sequence and replay the TENSOR filter along that same
;; trajectory, asserting the resulting belief vectors equal the host :beliefs
;; (done-means [4]: tensor :update-belief == host bayes-update vectors to 1e-6).
(let [voi  (bp/voi-world {})
      opt  (bp/make-biased-pomdp-agent (assoc voi :alpha ##Inf :n-iters 8)
                                       {:discount 0.0 :bias :sophisticated :update-myopic-bound ##Inf})
      s0   (:start-idx opt)
      pv   (:prior-vec opt)
      host (bp/simulate-biased-pomdp opt :B s0 8 pv)        ; host-filter beliefs (ground truth)
      ub-t (:update-belief-tensor opt)
      ;; replay the tensor filter along the host trajectory's (s', o) pairs
      tens-beliefs (reductions (fn [b [s' o]] (ub-t b s' o))
                               pv
                               (map vector (rest (:states host)) (:observations host)))]
  (assert-true  "voi: tensor filter produces one belief per state (replay aligns)"
                (= (count tens-beliefs) (count (:beliefs host))))
  (assert-true  "voi: tensor filter == host bayes-update at every step (1e-6)"
                (every? (fn [[h t]] (vecs-close? h t 1e-6))
                        (map vector (:beliefs host) tens-beliefs))))

(println (str "\n== Results: " @passed " passed, " @failed " failed =="))
(when (pos? @failed) (js/process.exit 1))
