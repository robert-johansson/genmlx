;; @tier medium
;; Fused POMDP rollout + the tensor observation model (beans genmlx-7kaf / qbb0 / r35c).
;; Section 1 pins the in-graph observation likelihood against the host obs-likelihood-vec
;; (the qbb0 tensor-observe primitive). Section 2 pins a fully-fused single-trajectory
;; rollout against the host simulate-pomdp at alpha=Inf/noise=0 (the r35c requirement:
;; s' threads as a tensor, no per-step mx/item).
;;
;; Run: bun run --bun nbb test/genmlx/agents_fused_pomdp_test.cljs

(ns genmlx.agents-fused-pomdp-test
  (:require [genmlx.agents.belief :as belief]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as penv]
            [genmlx.mlx :as mx]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))

(defn- onehot [i n]
  (.astype (mx/equal (mx/arange n) (mx/scalar (double i))) mx/float32))
(defn- idx-of [v x] (first (keep-indexed (fn [i e] (when (= e x) i)) v)))

;; corridor fixture (same as agentmodels_pomdp_test): goals A/B at the top, signpost
;; at idx 7, the only route from the start passes it.
(def pgrid [[:A    :empty :B]
            [:wall :empty :wall]
            [:wall :empty :wall]
            [:wall :empty :wall]
            [:wall :empty :wall]])
(def signpost 7)
(def env (penv/restaurant-gridworld {:grid pgrid :goals [:A :B] :signpost signpost
                                     :true-world :A :start [1 4]}))
(def observe (:observe env))
(def worlds  (vec (:worlds env)))
(def S       (* (count (first pgrid)) (count pgrid)))   ; 3×5 = 15

;; ---------------------------------------------------------------------------
;; Section 1 — tensor observation likelihood == host obs-likelihood-vec
;; ---------------------------------------------------------------------------
(println "\n== 1. tensor obs-likelihood == host obs-likelihood-vec (all s', all true worlds) ==")
(def Obs (belief/obs-id-tensor observe worlds S))
(assert-true "obs-id-tensor has shape [S,W]" (= [S (count worlds)] (mx/shape Obs)))

(let [mism (atom 0)]
  (doseq [tw worlds]
    (let [tw-oh (onehot (idx-of worlds tw) (count worlds))]
      (doseq [s (range S)]
        (let [o      (observe tw s)
              host-L (vec (mx/->clj (belief/obs-likelihood-vec observe worlds s o)))
              tens-L (vec (mx/->clj (belief/obs-likelihood-tensor Obs (onehot s S) tw-oh)))]
          (when (not= host-L tens-L) (swap! mism inc))))))
  (assert-true (str "tensor L matches host L at every (s', true-world) — " (* (count worlds) S) " cases")
               (zero? @mism)))

;; spot-check the signpost semantics directly
(let [tw-oh (onehot (idx-of worlds :A) (count worlds))]
  (assert-true "at the signpost, true=:A -> L = onehot(:A) = [1 0]"
               (= [1.0 0.0] (vec (mx/->clj (belief/obs-likelihood-tensor Obs (onehot signpost S) tw-oh)))))
  (assert-true "off the signpost (nil obs) -> L = ones (uninformative)"
               (= [1.0 1.0] (vec (mx/->clj (belief/obs-likelihood-tensor Obs (onehot 13 S) tw-oh))))))

;; ---------------------------------------------------------------------------
;; Section 2 — fused-simulate-pomdp == host simulate-pomdp (alpha=Inf, noise=0)
;; ---------------------------------------------------------------------------
(println "\n== 2. fused-simulate-pomdp == host simulate-pomdp (alpha=Inf, noise=0) ==")
(doseq [tw [:A :B]]
  (let [e     (penv/restaurant-gridworld {:grid pgrid :goals [:A :B] :signpost signpost
                                          :true-world tw :start [1 4]})
        pa    (pomdp/make-pomdp-agent (assoc e :alpha ##Inf :gamma 1.0 :n-iters 40))
        host  (pomdp/simulate-pomdp pa e (:start-idx e) 12)
        fused (pomdp/fused-simulate-pomdp pa e (:start-idx e) 12)]
    (println (str "  true=" (name tw) "  host states " (:states host)
                  "  fused states " (:states fused)))
    (assert-true (str "true=" (name tw) ": fused states == host states")
                 (= (:states host) (:states fused)))
    (assert-true (str "true=" (name tw) ": #beliefs aligns with #states")
                 (= (count (:states fused)) (count (:beliefs fused)) (count (:beliefs host))))
    (let [maxerr (apply max 0.0 (for [[hb fb] (map vector (:beliefs host) (:beliefs fused))
                                      w worlds]
                                  (Math/abs (- (double (get hb w 0.0)) (double (get fb w 0.0))))))]
      (assert-true (str "true=" (name tw) ": fused beliefs == host beliefs (max err " (.toFixed maxerr 8) ")")
                   (< maxerr 1e-5)))))

;; ---------------------------------------------------------------------------
;; Section 3 — WORLD-DEPENDENT nil observe model (genmlx-2sgt)
;; ---------------------------------------------------------------------------
;; The shipped observe models gate nil by geometry (world-INDEPENDENT), so the
;; old emergent all-ones happened to hold. Here nil is world-DEPENDENT: at s'=0,
;; world :w0 yields nil but :w1 yields :x. The fused obs-likelihood-tensor must
;; still reproduce the host filter's unconditional nil-skip (effective L = ones),
;; so the fused belief == host filter belief. Pre-fix it gave [1 0] -> divergence.
(println "\n== 3. world-dependent nil observe model (genmlx-2sgt) ==")
(let [wd-worlds [:w0 :w1]
      wd-S 2
      wd-observe (fn [w s] (cond (and (= w :w0) (= s 0)) nil      ; world-dependent nil
                                 (and (= w :w1) (= s 0)) :x
                                 :else :y))                       ; s=1: world-independent non-nil
      WObs (belief/obs-id-tensor wd-observe wd-worlds wd-S)
      W    (count wd-worlds)
      b0   (belief/belief->vec wd-worlds {:w0 0.5 :w1 0.5})]
  ;; the true world (:w0) yields nil at s'=0 -> L must be forced all-ones
  (let [tens-L (vec (mx/->clj (belief/obs-likelihood-tensor WObs (onehot 0 wd-S) (onehot 0 W))))]
    (assert-true (str "true=:w0 (yields nil at s'=0): tensor L = all-ones, not [1 0] — got " tens-L)
                 (= [1.0 1.0] tens-L)))
  ;; fused belief update == host filter belief, for BOTH true worlds at s'=0
  (doseq [[twi o] [[0 nil] [1 :x]]]
    (let [tw-oh    (onehot twi W)
          host-b   (belief/tensor-update-belief wd-observe wd-worlds b0 0 o)   ; nil -> skip
          fused-L  (belief/obs-likelihood-tensor WObs (onehot 0 wd-S) tw-oh)
          fused-b  (belief/filter-step b0 fused-L)
          maxerr   (apply max 0.0 (map (fn [a c] (Math/abs (- a c)))
                                       (mx/->clj host-b) (mx/->clj fused-b)))]
      (assert-true (str "true=" (name (nth wd-worlds twi)) " (o=" o "): fused belief == host filter belief (err "
                        (.toFixed maxerr 8) ")")
                   (< maxerr 1e-5)))))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
