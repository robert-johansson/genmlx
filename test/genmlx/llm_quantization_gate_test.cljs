;; @tier fast
(ns genmlx.llm-quantization-gate-test
  "genmlx-a2mq (Ornith Phase 0): per-tensor quantization overrides.

   MLX checkpoints may carry per-tensor override sub-objects inside
   config.json's `quantization` block. Two real cases pinned here:
   - Ornith-1.0-35B-8bit: 80 overrides all byte-identical to the global
     {bits 8, group_size 64} — redundant metadata, previously a FALSE-POSITIVE
     rejection (the old :per-layer? flag fired on structure, not substance).
   - Ornith-1.0-35B-3bit: global {bits 3} with {bits 8} overrides on the
     router gates — GENUINELY mixed. Parseable now; still (correctly) not
     dequantizable, because 3-bit values straddle u32 words and the pure
     floor-divide/remainder unpack is power-of-two-only.

   Part A pins dequantize-weights numerics with a per-tensor override on
   synthetic packed tensors (exact expected values). Part B pins the
   dequantizable? gate logic on plain maps. Part C reads the real Ornith
   configs when cached (skips per-checkpoint otherwise)."
  (:require [genmlx.llm.qwen3-forward :as q3]
            [genmlx.llm.forward :as fwd]
            [genmlx.mlx :as mx]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))
(defn assert-close [label expected actual tol]
  (assert-true (str label " (expected " expected ", got " actual ")")
               (< (js/Math.abs (- expected actual)) tol)))

;; ---------------------------------------------------------------------------
;; Part A — dequantize-weights resolves per-tensor bits from :overrides
;; ---------------------------------------------------------------------------
;; Packing is LSB-first per u32 word (mlx ops.cpp: out |= v << (k*bits)), so
;; word = sum_k v_k * (2^bits)^k. Values are small enough that float64
;; arithmetic is exact.

(println "\n== Part A: per-tensor dequantize numerics (synthetic) ==")

(defn- pack-word [vals bits]
  (reduce + (map-indexed (fn [k v] (* v (js/Math.pow (js/Math.pow 2 bits) k))) vals)))

(let [;; tensor a: GLOBAL scheme, bits=4 (pf=8). [2,8] logical, one word/row.
      a-q0  [1 2 3 4 5 6 7 8]      ; scale 2.0 bias  1.0 -> [3 5 7 9 11 13 15 17]
      a-q1  [15 0 15 0 15 0 15 0]  ; scale 0.5 bias -1.0 -> [6.5 -1 ...]
      a-w   (mx/array [(pack-word a-q0 4) (pack-word a-q1 4)] [2 1] mx/uint32)
      a-s   (mx/array [2.0 0.5] [2 1] mx/float32)
      a-b   (mx/array [1.0 -1.0] [2 1] mx/float32)
      ;; tensor b: OVERRIDE scheme, bits=8 (pf=4). [1,4] logical, one word.
      b-q   [10 20 30 255]         ; scale 1.5 bias 0.5 -> [15.5 30.5 45.5 383.0]
      b-w   (mx/array [(pack-word b-q 8)] [1 1] mx/uint32)
      b-s   (mx/array [1.5] [1 1] mx/float32)
      b-b   (mx/array [0.5] [1 1] mx/float32)
      ;; tensor n: not quantized (no .scales sibling) — must pass through.
      n-w   (mx/array [42.0] [1] mx/float32)
      weights {"a.weight" a-w "a.scales" a-s "a.biases" a-b
               "b.weight" b-w "b.scales" b-s "b.biases" b-b
               "n.weight" n-w}
      qz    {:bits 4 :group-size 8 :mode "affine"
             :overrides {"b" {:bits 8 :group-size 4}}}
      dq    (q3/dequantize-weights weights qz)
      a-out (mx/->clj (get dq "a.weight"))
      b-out (mx/->clj (get dq "b.weight"))]
  (assert-true "scales/biases folded away"
               (and (not (contains? dq "a.scales")) (not (contains? dq "b.biases"))))
  (assert-true "unquantized tensor passes through"
               (= 42.0 (first (mx/->clj (get dq "n.weight")))))
  (assert-true "global-scheme tensor shape [2 8]" (= [2 8] (mx/shape (get dq "a.weight"))))
  (assert-true "override tensor shape [1 4]"      (= [1 4] (mx/shape (get dq "b.weight"))))
  (doseq [[j e] (map-indexed vector [3.0 5.0 7.0 9.0 11.0 13.0 15.0 17.0])]
    (assert-close (str "a row0 col" j) e (nth (nth a-out 0) j) 1e-6))
  (doseq [[j e] (map-indexed vector [6.5 -1.0 6.5 -1.0 6.5 -1.0 6.5 -1.0])]
    (assert-close (str "a row1 col" j) e (nth (nth a-out 1) j) 1e-6))
  (doseq [[j e] (map-indexed vector [15.5 30.5 45.5 383.0])]
    (assert-close (str "b (bits=8 override) col" j) e (nth (nth b-out 0) j) 1e-6)))

;; ---------------------------------------------------------------------------
;; Part B — dequantizable? gate logic
;; ---------------------------------------------------------------------------

(println "\n== Part B: dequantizable? gate ==")

(assert-true "uniform 8-bit, no overrides"
             (q3/dequantizable? {:bits 8 :mode "affine" :overrides {}}))
(assert-true "nil overrides (hand-built map)"
             (q3/dequantizable? {:bits 8 :mode "affine"}))
(assert-true "redundant overrides == global (the old false positive)"
             (q3/dequantizable? {:bits 8 :mode "affine"
                                 :overrides {"x" {:bits 8 :group-size 64}}}))
(assert-true "genuinely mixed but all power-of-two (4 global, 8 override)"
             (q3/dequantizable? {:bits 4 :mode "affine"
                                 :overrides {"x" {:bits 8 :group-size 64}}}))
(assert-true "odd-bit GLOBAL accepted (bits=3 — native mx/dequantize, genmlx-dlvi)"
             (q3/dequantizable? {:bits 3 :mode "affine"
                                 :overrides {"x" {:bits 8 :group-size 64}}}))
(assert-true "odd-bit OVERRIDE accepted (bits=5)"
             (q3/dequantizable? {:bits 8 :mode "affine"
                                 :overrides {"x" {:bits 5 :group-size 64}}}))
(assert-true "unknown bit width rejected (bits=7)"
             (not (q3/dequantizable? {:bits 7 :mode "affine" :overrides {}})))
(assert-true "non-affine mode rejected"
             (not (q3/dequantizable? {:bits 8 :mode "mxfp4" :overrides {}})))

;; ---------------------------------------------------------------------------
;; Part C — the real Ornith configs (skip per-checkpoint when absent)
;; ---------------------------------------------------------------------------

(println "\n== Part C: real Ornith configs ==")

(defn- snapshot-dir [model-name]
  (let [base (str (.-HOME js/process.env) "/.cache/huggingface/hub/" model-name "/snapshots")]
    (when (.existsSync fs base)
      (when-let [h (first (js->clj (.readdirSync fs base)))]
        (str base "/" h)))))

(if-let [d (snapshot-dir "models--mlx-community--Ornith-1.0-35B-8bit")]
  (let [qz (q3/load-quantization d)]
    (assert-true "8bit: global bits=8 group=64"
                 (and (= 8 (:bits qz)) (= 64 (:group-size qz))))
    (assert-true "8bit: 80 overrides parsed" (= 80 (count (:overrides qz))))
    (assert-true "8bit: all overrides == global (redundant)"
                 (every? #(and (= 8 (:bits %)) (= 64 (:group-size %)))
                         (vals (:overrides qz))))
    (assert-true "8bit: dequantizable? TRUE (gate 3 no longer fires)"
                 (q3/dequantizable? qz))
    ;; All three gates are down: gate 3 fell with per-tensor overrides
    ;; (genmlx-a2mq), gate 2 with sharded loading (genmlx-sbif), gate 1 with
    ;; the owned qwen3_5_moe forward (genmlx-g6vk).
    (assert-true "8bit: model_type owned (gate 1 fell — genmlx-g6vk)"
                 (contains? fwd/supported-model-types "qwen3_5_moe"))
    (assert-true "8bit: sharded layout loadable (gate 2 no longer fires)"
                 (fwd/loadable-weights? d))
    (assert-true "8bit: 8 shard files resolved from index.json"
                 (= 8 (count (q3/weight-files d))))
    (assert-true "8bit: fwd/supported? TRUE — Ornith runs the owned forward"
                 (fwd/supported? d)))
  (println "  SKIP: Ornith-1.0-35B-8bit not cached"))

(if-let [d (snapshot-dir "models--mlx-community--Ornith-1.0-35B-3bit")]
  (let [qz (q3/load-quantization d)]
    (assert-true "3bit: global bits=3" (= 3 (:bits qz)))
    (assert-true "3bit: 80 overrides parsed, all {bits 8, group 64} (genuinely mixed)"
                 (and (= 80 (count (:overrides qz)))
                      (every? #(and (= 8 (:bits %)) (= 64 (:group-size %)))
                              (vals (:overrides qz)))))
    (assert-true "3bit: dequantizable? TRUE (odd bits via native mx/dequantize, genmlx-dlvi)"
                 (q3/dequantizable? qz))
    (assert-true "3bit: fwd/supported? TRUE (all gates down, mixed 3/8-bit resolves per-tensor)"
                 (fwd/supported? d)))
  (println "  SKIP: Ornith-1.0-35B-3bit not cached"))

(if-let [d (snapshot-dir "models--mlx-community--Ornith-1.0-35B-4bit")]
  (let [qz (q3/load-quantization d)]
    (assert-true "4bit: global bits=4" (= 4 (:bits qz)))
    (assert-true "4bit: dequantizable? (all schemes power-of-two)"
                 (q3/dequantizable? qz)))
  (println "  SKIP: Ornith-1.0-35B-4bit not cached"))

(println (str "\n=== llm-quantization-gate: " @pass " PASS, " @fail " FAIL ==="))
(when (pos? @fail) (js/process.exit 1))
