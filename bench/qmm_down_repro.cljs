(ns qmm-down-repro
  "genmlx-mdet Phase-1.5b: standalone nondeterminism probe for the expert-path
   gather-qmm at the EXACT real Ornith-35B down_proj shape (the op the in-situ
   bisect caught diverging with bit-identical inputs at T=512 prefill).

   Real dims (Ornith-1.0-35B-8bit): E=256 experts, k=8, hidden=2048,
   moe-inter=512, bits=8, group-size=64, BT=512, x/scales bf16.

   Variant A — bare loop: 30x the down gather-qmm alone on fixed inputs
   (the shape the old standalone exoneration never tested).
   Variant B — block-context: 30x the FULL expert-branch graph per eval
   (gate qmm + up qmm + silu*mul + down qmm + weighted k-sum), mimicking the
   in-situ evaluation where the divergence was observed.

   Variant C — control at the gate/up shape (in situ those never diverged).
   Variant D — sorted?=true probe (Phase-2 lever candidate).

   Bit-exact fingerprints (2 fixed projections + sum + amax) per rep.

   MEASURED (Thor CUDA, 2026-07-12): A NONDET (3-4 distinct/30) · B NONDET ·
   C CLEAN (30/30) · D NONDET — the down-projection SHAPE takes a
   nondeterministic kernel path in isolation; sorted? is not the fix.
   Usage (repo root, box otherwise idle):
     bunx --bun nbb@1.4.208 bench/qmm_down_repro.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(def BT 512) (def K 8) (def E 256) (def HID 2048) (def INTER 512)
(def BITS 8) (def GS 64) (def REPS 30)
(def PER-U32 (quot 32 BITS))

(defn- u32-arr [key shape]
  (mx/astype (mx/multiply (rng/uniform key shape) 4.0e9) mx/uint32))
(defn- bf16-arr [key shape scale]
  (mx/astype (mx/multiply (mx/subtract (mx/multiply (rng/uniform key shape) 2.0) 1.0)
                          scale)
             mx/bfloat16))

(def ks (rng/split-n (rng/fresh-key 20260712) 12))

;; packed expert tensors at real shapes
(def wd  (u32-arr (nth ks 0) [E HID (quot INTER PER-U32)]))     ; down: [256 2048 128]
(def sd  (bf16-arr (nth ks 1) [E HID (quot INTER GS)] 0.01))
(def bd  (bf16-arr (nth ks 2) [E HID (quot INTER GS)] 0.01))
(def wg  (u32-arr (nth ks 3) [E INTER (quot HID PER-U32)]))     ; gate: [256 512 512]
(def sg  (bf16-arr (nth ks 4) [E INTER (quot HID GS)] 0.01))
(def bg  (bf16-arr (nth ks 5) [E INTER (quot HID GS)] 0.01))
(def wu  (u32-arr (nth ks 6) [E INTER (quot HID PER-U32)]))     ; up
(def su  (bf16-arr (nth ks 7) [E INTER (quot HID GS)] 0.01))
(def bu  (bf16-arr (nth ks 8) [E INTER (quot HID GS)] 0.01))

(def idx (mx/astype (mx/multiply (rng/uniform (nth ks 9) [BT K]) E) mx/uint32))
(def x1  (bf16-arr (nth ks 10) [BT 1 1 HID] 1.0))               ; block input
(def xd  (bf16-arr (nth ks 11) [BT K 1 INTER] 1.0))             ; bare down input
(def topw (mx/astype (mx/full [BT K 1] 0.125) mx/bfloat16))

(apply mx/materialize! [wd sd bd wg sg bg wu su bu idx x1 xd topw])

(def r1 (rng/uniform (rng/fresh-key 111) [BT K 1 HID]))
(def r2 (rng/uniform (rng/fresh-key 222) [BT K 1 HID]))
(def r1s (rng/uniform (rng/fresh-key 333) [BT HID]))
(def r2s (rng/uniform (rng/fresh-key 444) [BT HID]))
(mx/materialize! r1 r2 r1s r2s)

(defn- fp [x ra rb]
  (let [xf (mx/astype x mx/float32)
        v [(mx/sum (mx/multiply xf ra)) (mx/sum (mx/multiply xf rb))
           (mx/sum xf) (mx/amax (mx/abs xf))]]
    (apply mx/materialize! v)
    (mapv mx/item v)))

(def qopts {:rhs-indices idx :bits BITS :group-size GS :transpose true :sorted? false})

(defn- run-variant [label f ra rb]
  (let [fps (mapv (fn [i]
                    (let [v (fp (f) ra rb)]
                      (mx/sweep-dead-arrays!)
                      v))
                  (range REPS))
        uniq (count (distinct fps))]
    (println (str "  " label ": " uniq " distinct fingerprint(s) over " REPS " reps"
                  (when (> uniq 1)
                    (str "  << NONDETERMINISTIC (p1 spread "
                         (.toExponential (- (apply max (map first fps))
                                            (apply min (map first fps))) 3) ")"))))
    uniq))

(println "== qmm-down standalone probe (real Ornith down_proj dims) ==")
(println (str "  x [" BT " " K " 1 " INTER "] bf16 . w [" E " " HID " "
              (quot INTER PER-U32) "] u32-packed . bits " BITS " gs " GS))

(def a (run-variant "A bare down-qmm      "
                    #(mx/gather-qmm xd wd sd bd qopts) r1 r2))

(def b (run-variant "B full expert block  "
                    #(let [g (mx/gather-qmm x1 wg sg bg qopts)
                           u (mx/gather-qmm x1 wu su bu qopts)
                           h (mx/multiply (mx/silu g) u)
                           y (mx/gather-qmm h wd sd bd qopts)          ; [BT K 1 HID]
                           o (mx/sum (mx/multiply (mx/reshape y [BT K HID])
                                                  (mx/astype topw (mx/dtype y)))
                                     [1] false)]                       ; [BT HID]
                       o)
                    r1s r2s))

;; control: the gate/up shape (x [BT 1 1 HID] -> [BT K 1 INTER]) — in situ
;; these never diverged with clean inputs; is that shape-specific?
(def r1g (rng/uniform (rng/fresh-key 555) [BT K 1 INTER]))
(def r2g (rng/uniform (rng/fresh-key 666) [BT K 1 INTER]))
(mx/materialize! r1g r2g)
(def c (run-variant "C bare gate-shape qmm"
                    #(mx/gather-qmm x1 wg sg bg qopts) r1g r2g))

;; Phase-2 lever probe: does sorted?=true change the down kernel's behavior?
;; (oracle Part 3 measured the flag as a performance hint on this backend)
(def idx-sorted (mx/sort-arr idx 1))
(def _ (mx/materialize! idx-sorted))
(def d (run-variant "D down-qmm sorted idx"
                    #(mx/gather-qmm xd wd sd bd
                                    (assoc qopts :rhs-indices idx-sorted :sorted? true))
                    r1 r2))

(println (str "\nVERDICT: bare-down=" (if (> a 1) "NONDET" "clean")
              " block=" (if (> b 1) "NONDET" "clean")
              " bare-gate=" (if (> c 1) "NONDET" "clean")
              " down-sorted=" (if (> d 1) "NONDET" "clean")))
(js/process.exit 0)
