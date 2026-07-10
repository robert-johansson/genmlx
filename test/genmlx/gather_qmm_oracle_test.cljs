;; @tier medium
(ns genmlx.gather-qmm-oracle-test
  "genmlx-q5uq (Ornith Phase 2): numerical oracle for the two new membrane ops
   mx/gather-qmm and mx/dequantize.

   Part 1 — native mx/dequantize vs the PURE floor-divide/remainder unpack
   (q3/dequantize-weights) on 2-D tensors, bits {4,8}: the two implementations
   must agree exactly. This pins that both sides use MLX's LSB-first packing.

   Part 2 — gather_qmm vs dequantize-then-matmul on synthetic [E,out,in] packed
   expert tensors, all of bits {3,4,8} x sorted {false,true-on-ascending}:
   internal consistency of the fused quantized kernel against the composed
   reference. bits=3 has no pure-CLJS reference (values straddle u32 words) —
   agreement between the two NATIVE paths is exactly the contract the owned
   MoE forward needs (loader hands checkpoint tensors straight to gather_qmm).

   Part 3 — the sorted? contract, MEASURED (spec open question 1): sorted?=true
   on genuinely UNSORTED indices. Printed, not asserted — this documents whether
   the flag is a hint or a hard contract.

   Part 4 — nondeterminism probe (spec open question 3): 100 iterations of the
   identical gather_qmm graph; prints max deviation from run 0. Decides whether
   the ba06 log-weight jitter originates in this kernel. Printed AND recorded
   in the bean; asserted only to be finite."
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.qwen3-forward :as q3]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

;; deterministic LCG so runs are reproducible without Math.random
(defn- lcg [seed]
  (let [state (atom seed)]
    (fn [] (let [s (mod (+ (* @state 1103515245) 12345) 2147483648)]
             (reset! state s)
             (/ s 2147483648)))))

(defn- rand-u32-array
  "[n] random u32 words as an mx uint32 array of the given shape."
  [rnd shape]
  (let [n (reduce * shape)]
    (mx/array (vec (repeatedly n #(js/Math.floor (* (rnd) 4294967295)))) shape mx/uint32)))

(defn- rand-f32-array [rnd shape lo hi]
  (let [n (reduce * shape)]
    (mx/array (vec (repeatedly n #(+ lo (* (rnd) (- hi lo))))) shape mx/float32)))

;; ---------------------------------------------------------------------------
;; Part 1 — native dequantize == pure CLJS unpack (bits 4, 8; 2-D)
;; ---------------------------------------------------------------------------

(println "\n== Part 1: mx/dequantize vs pure floor-divide/remainder unpack ==")

(doseq [bits [4 8]]
  (let [rnd    (lcg (+ 42 bits))
        out    8
        in     64
        gs     32
        packed (quot in (quot 32 bits))
        wq     (rand-u32-array rnd [out packed])
        scales (rand-f32-array rnd [out (quot in gs)] 0.01 0.1)
        biases (rand-f32-array rnd [out (quot in gs)] -0.5 0.5)
        pure   (get (q3/dequantize-weights
                     {"t.weight" wq "t.scales" scales "t.biases" biases}
                     {:bits bits :group-size gs :mode "affine"})
                    "t.weight")
        native (mx/dequantize wq scales biases {:group-size gs :bits bits})
        scale  (max 1.0 (mx/item (mx/amax (mx/abs pure))))
        diff   (/ (mx/item (mx/amax (mx/abs (mx/subtract pure native)))) scale)]
    ;; f32 rounding-order differs between the two implementations (pure:
    ;; broadcast multiply-add; native: fused) — agreement is to f32 epsilon
    ;; relative, not bit-exact.
    (assert-true (str "bits=" bits ": native == pure unpack (rel|diff| = "
                      (.toExponential diff 2) ")")
                 (< diff 1e-6))))

;; ---------------------------------------------------------------------------
;; Part 2 — gather_qmm vs dequantize+matmul (bits {3,4,8} x sorted {f,t})
;; ---------------------------------------------------------------------------

(println "\n== Part 2: gather_qmm vs dequantize-then-matmul oracle ==")

(def E 4)      ; experts
(def OUT 8)
(def IN 64)
(def GS 32)
(def NE 12)    ; rows of x

(defn- expert-fixture [bits seed]
  (let [rnd    (lcg seed)
        packed (quot IN (quot 32 bits))]
    {:wq     (rand-u32-array rnd [E OUT packed])
     :scales (rand-f32-array rnd [E OUT (quot IN GS)] 0.01 0.1)
     :biases (rand-f32-array rnd [E OUT (quot IN GS)] -0.2 0.2)
     :x      (rand-f32-array rnd [NE 1 IN] -1.0 1.0)}))

(defn- reference
  "dequantize all experts, gather the selected one per row, matmul."
  [{:keys [wq scales biases x]} idx-vec bits]
  (let [dq  (mx/dequantize wq scales biases {:group-size GS :bits bits :out-dtype 0})
        idx (mx/array idx-vec [NE] mx/uint32)
        sel (mx/take-idx dq idx 0)]                 ; [NE OUT IN]
    (mx/matmul x (mx/transpose sel [0 2 1]))))      ; [NE 1 IN]@[NE IN OUT] -> [NE 1 OUT]

(defn- fused [{:keys [wq scales biases x]} idx-vec bits sorted?]
  (mx/gather-qmm x wq scales biases
                 {:rhs-indices (mx/array idx-vec [NE] mx/uint32)
                  :group-size GS :bits bits :sorted? sorted?}))

(def unsorted-idx [3 0 2 1 0 3 1 2 0 0 3 1])
(def sorted-idx   [0 0 0 0 1 1 1 2 2 2 3 3])

(doseq [bits [3 4 8]]
  (let [fx (expert-fixture bits (* 7 bits))]
    (doseq [[idx sorted? label] [[unsorted-idx false "unsorted idx, sorted?=false"]
                                 [sorted-idx   false "ascending idx, sorted?=false"]
                                 [sorted-idx   true  "ascending idx, sorted?=true"]]]
      (let [ref  (reference fx idx bits)
            got  (fused fx idx bits sorted?)
            scale (max 1.0 (mx/item (mx/amax (mx/abs ref))))
            diff (/ (mx/item (mx/amax (mx/abs (mx/subtract ref got)))) scale)]
        (assert-true (str "bits=" bits " " label " (rel|diff| = " (.toExponential diff 2) ")")
                     (< diff 1e-2))))))

;; ---------------------------------------------------------------------------
;; Part 3 — sorted?=true on UNSORTED indices: measure, don't assert (Q1)
;; ---------------------------------------------------------------------------

(println "\n== Part 3: sorted?=true misuse — measured, not asserted ==")

(let [fx   (expert-fixture 8 99)
      ref  (reference fx unsorted-idx 8)
      got  (fused fx unsorted-idx 8 true)
      diff (mx/item (mx/amax (mx/abs (mx/subtract ref got))))]
  (println (str "  bits=8, unsorted idx forced through sorted?=true: max|diff| vs ref = " diff))
  (println (str "  => sorted? is " (if (< diff 1e-4)
                                     "apparently TOLERANT of unsorted input (hint only)"
                                     "a HARD CONTRACT (wrong results on unsorted input)")
                " on this backend"))
  (assert-true "sorted?-misuse probe ran" (js/isFinite diff)))

;; ---------------------------------------------------------------------------
;; Part 4 — kernel nondeterminism probe, 100 iterations (Q3)
;; ---------------------------------------------------------------------------

(println "\n== Part 4: gather_qmm nondeterminism probe (100 iterations) ==")

(let [fx  (expert-fixture 8 7)
      ;; fresh graph per iteration over identical leaves, like a real forward
      run (fn [] (mx/->clj (mx/reshape (fused fx unsorted-idx 8 false) [(* NE OUT)])))
      r0  (run)
      max-dev (reduce (fn [acc _]
                        (let [ri (run)]
                          (reduce (fn [a [x y]] (max a (js/Math.abs (- x y))))
                                  acc (map vector r0 ri))))
                      0.0 (range 99))]
  (println (str "  max deviation from run 0 across 99 re-runs: " max-dev))
  (println (str "  => gather_qmm on CUDA is "
                (if (zero? max-dev)
                  "bit-deterministic at this size (jitter must enter elsewhere)"
                  "NONDETERMINISTIC — the ba06 log-weight jitter can originate here")))
  (assert-true "nondeterminism probe ran (deviation finite)" (js/isFinite max-dev)))

(println (str "\n=== gather-qmm-oracle: " @pass " PASS, " @fail " FAIL ==="))
(when (pos? @fail) (js/process.exit 1))
