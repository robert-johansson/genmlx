;; @tier medium
(ns genmlx.qmm-determinism-test
  "genmlx-mdet regression guard: mx/gather-qmm must be bit-deterministic at
   the MoE expert DOWN-projection shape (x [BT k 1 inter] against packed
   [E out in] experts) — the exact shape whose qmm_sm80 smem-reuse race
   (missing cp_async_wait<0> before the epilogue's C staging over the
   mainloop A/B union) produced the quantized-MoE run-to-run jitter on CUDA
   (0.2–2.6 nats at the logits, near-tie greedy flips; fixed in the mlx fork
   2026-07-12). Synthetic weights, no model, no LLM — the kernel race was
   value-independent. Also guards the gate/up shape (historically clean) and
   the full expert-block composition.

   Bit-exactness is asserted via two fixed-key random projections + sum +
   absmax per rep: a single flipped bit anywhere in the output moves the
   projections with probability ~1."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(def ^:private fails (atom 0))
(defn assert-true [msg p] (if p (println (str "  [PASS] " msg))
                              (do (swap! fails inc) (println (str "  [FAIL] " msg)))))

;; Real Ornith-1.0-35B dims (the shape that raced); REPS tier-capped.
(def BT 512) (def K 8) (def E 256) (def HID 2048) (def INTER 512)
(def BITS 8) (def GS 64) (def REPS 10)
(def PER-U32 (quot 32 BITS))

(defn- u32-arr [key shape]
  (mx/astype (mx/multiply (rng/uniform key shape) 4.0e9) mx/uint32))
(defn- bf16-arr [key shape scale]
  (mx/astype (mx/multiply (mx/subtract (mx/multiply (rng/uniform key shape) 2.0) 1.0)
                          scale)
             mx/bfloat16))

(def ks (rng/split-n (rng/fresh-key 20260712) 12))

(def wd  (u32-arr (nth ks 0) [E HID (quot INTER PER-U32)]))
(def sd  (bf16-arr (nth ks 1) [E HID (quot INTER GS)] 0.01))
(def bd  (bf16-arr (nth ks 2) [E HID (quot INTER GS)] 0.01))
(def wg  (u32-arr (nth ks 3) [E INTER (quot HID PER-U32)]))
(def sg  (bf16-arr (nth ks 4) [E INTER (quot HID GS)] 0.01))
(def bg  (bf16-arr (nth ks 5) [E INTER (quot HID GS)] 0.01))
(def wu  (u32-arr (nth ks 6) [E INTER (quot HID PER-U32)]))
(def su  (bf16-arr (nth ks 7) [E INTER (quot HID GS)] 0.01))
(def bu  (bf16-arr (nth ks 8) [E INTER (quot HID GS)] 0.01))
(def idx (mx/astype (mx/multiply (rng/uniform (nth ks 9) [BT K]) E) mx/uint32))
(def x1  (bf16-arr (nth ks 10) [BT 1 1 HID] 1.0))
(def xd  (bf16-arr (nth ks 11) [BT K 1 INTER] 1.0))
(def topw (mx/astype (mx/full [BT K 1] 0.125) mx/bfloat16))
(apply mx/materialize! [wd sd bd wg sg bg wu su bu idx x1 xd topw])

(def r-down [(rng/uniform (rng/fresh-key 111) [BT K 1 HID])
             (rng/uniform (rng/fresh-key 222) [BT K 1 HID])])
(def r-gate [(rng/uniform (rng/fresh-key 555) [BT K 1 INTER])
             (rng/uniform (rng/fresh-key 666) [BT K 1 INTER])])
(def r-blk  [(rng/uniform (rng/fresh-key 333) [BT HID])
             (rng/uniform (rng/fresh-key 444) [BT HID])])
(apply mx/materialize! (concat r-down r-gate r-blk))

(defn- fp [x [ra rb]]
  (let [xf (mx/astype x mx/float32)
        v [(mx/sum (mx/multiply xf ra)) (mx/sum (mx/multiply xf rb))
           (mx/sum xf) (mx/amax (mx/abs xf))]]
    (apply mx/materialize! v)
    (mapv mx/item v)))

(def qopts {:rhs-indices idx :bits BITS :group-size GS :transpose true :sorted? false})

(defn- distinct-over-reps [f rs]
  (count (distinct (mapv (fn [_] (let [v (fp (f) rs)]
                                   (mx/sweep-dead-arrays!)
                                   v))
                         (range REPS)))))

(println "\n== gather-qmm bit-determinism at the MoE expert shapes ==")

(assert-true (str "down-projection shape bit-identical over " REPS " reps "
                  "(the genmlx-mdet raced shape)")
             (= 1 (distinct-over-reps #(mx/gather-qmm xd wd sd bd qopts) r-down)))

(assert-true (str "gate/up-projection shape bit-identical over " REPS " reps")
             (= 1 (distinct-over-reps #(mx/gather-qmm x1 wg sg bg qopts) r-gate)))

(assert-true (str "full expert block (gate+up+silu·mul+down+weighted k-sum) "
                  "bit-identical over " REPS " reps")
             (= 1 (distinct-over-reps
                   #(let [g (mx/gather-qmm x1 wg sg bg qopts)
                          u (mx/gather-qmm x1 wu su bu qopts)
                          h (mx/multiply (mx/silu g) u)
                          y (mx/gather-qmm h wd sd bd qopts)]
                      (mx/sum (mx/multiply (mx/reshape y [BT K HID])
                                           (mx/astype topw (mx/dtype y)))
                              [1] false))
                   r-blk)))

(println (str "\n== qmm_determinism_test: "
              (if (zero? @fails) "ALL PASS" (str @fails " FAIL")) " =="))
(when (pos? @fails) (js/process.exit 1))
