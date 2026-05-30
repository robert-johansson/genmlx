(ns demo-value-semantics-gpu
  "DISTINCTIVE FEATURE: value-semantics through the GPU.

   MLX operations do NOT compute. mx/add, mx/multiply, mx/exp, mx/log ...
   build nodes in a LAZY computation graph. That graph is itself an
   immutable VALUE — like any Clojure data structure — and no Metal work
   happens until mx/eval! / mx/item forces it. So the whole probabilistic
   algebra (scores, weights, posteriors) composes as deferred values, and
   `eval!` is the SINGLE dispatch boundary where the device actually runs.

   This is why GenMLX's 'purely functional' story extends all the way
   through the GPU: device computation is a value until eval!."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn now [] (js/performance.now))
(defn ms [x] (.toFixed x 3))

;; ─────────────────────────────────────────────────────────────────────────
;; (a) A LARGE lazy graph is built as a VALUE — construction is ~free.
;;     Only eval!/item dispatches it to the GPU.
;; ─────────────────────────────────────────────────────────────────────────
(println "=== (a) Lazy graph IS a value: construction is ~free, eval! is the work ===")

(def n 1000000)
(def iters 50)

;; Build the source array first (this includes host-side data marshalling:
;; turning a Clojure vector into a Float32Array). We do NOT count this as
;; "graph construction" — it is data loading, a separate concern.
(def base (mx/array (vec (range n))))   ; 1,000,000-element lazy array

;; --- Time GRAPH CONSTRUCTION ---------------------------------------------
;; Chain 50 elementwise op-layers over the existing 1,000,000-element array.
;; Each op (mx/add, mx/multiply, mx/exp, mx/log) returns a NEW graph node;
;; nothing is computed — we are only building an immutable value.
(def t0 (now))
(def graph
  (loop [k 0
         acc base]
    (if (= k iters)
      acc
      ;; A bounded, numerically-safe op layer so values stay finite across
      ;; 50 stacked layers: soft log1p(exp(tiny * x)) increment.
      (let [scaled  (mx/multiply acc (mx/scalar 1.0000001))
            shifted (mx/add scaled (mx/scalar 0.5))
            nonlin  (mx/log (mx/add (mx/scalar 1.0)
                                    (mx/exp (mx/multiply shifted (mx/scalar 1e-7)))))]
        (recur (inc k) (mx/add acc nonlin))))))
(def t1 (now))
(def construct-ms (- t1 t0))

(println (str "  built a " iters "-layer op graph over " n " elements"))
(println (str "  (data load of the base array is excluded — that's marshalling, not graph build)"))
(println (str "  GRAPH CONSTRUCTION time: " (ms construct-ms) " ms"
              "  (no GPU work — just building an immutable value)"))
(println (str "  the graph is still just a value; (mx/shape graph) = "
              (mx/shape graph) "  <- known WITHOUT computing it"))

;; --- Time EVALUATION (the single GPU dispatch boundary) -------------------
;; mx/sum collapses to a scalar; mx/item is the eval! boundary that finally
;; sends the whole DAG to Metal.
(def reduced (mx/sum graph))   ; still lazy: one more node on the graph
(def t2 (now))
(def result (mx/item reduced)) ; <- THIS forces the GPU
(def t3 (now))
(def eval-ms (- t3 t2))

(println (str "  EVALUATION time (mx/item forces Metal): " (ms eval-ms) " ms"))
(println (str "  result (sum over " n " elements): " result))
(println (str "  ratio  eval / construct = "
              (.toFixed (/ eval-ms (max construct-ms 1e-6)) 1) "x"
              "   => construction << evaluation"))

;; Re-evaluating the SAME value is idempotent (a value, not an action):
(def t4 (now))
(def result2 (mx/item reduced))
(def t5 (now))
(println (str "  re-eval of the same value: " (ms (- t5 t4)) " ms (cached) ; equal? "
              (= result result2)))

;; ─────────────────────────────────────────────────────────────────────────
;; (b) A trace's :score is itself an UNEVALUATED MLX value.
;;     The probabilistic algebra is deferred just like raw array math.
;; ─────────────────────────────────────────────────────────────────────────
(println "\n=== (b) A trace :score is a deferred MLX value, not a number ===")

(def model
  (gen [mu0]
    (let [mu (trace :mu (dist/gaussian mu0 3.0))
          y  (trace :y  (dist/gaussian mu 1.0))]
      {:mu mu :y y})))

(def tr (p/simulate (dyn/auto-key model) [0.0]))
(def score (:score tr))

(println (str "  (.-name (type (:score tr))) = " (.-name (type score))
              "  <- the MLX array class"))
(println (str "  (mx/array? score)  = " (mx/array? score)
              "   <- it's a lazy MLX graph node, NOT a JS number"))
(println (str "  (mx/shape score)   = " (mx/shape score) " (a 0-d / scalar graph)"))

;; We can KEEP composing on the deferred score without evaluating it:
;; e.g. turn log-score into a probability-ish quantity, still lazy.
(def derived (mx/exp (mx/multiply score (mx/scalar 0.5))))
(println (str "  composed (mx/exp (* 0.5 score)) -> still MxArray? "
              (mx/array? derived) " (no GPU work yet)"))

;; Only NOW do we cross the eval boundary:
(println (str "  (mx/item (:score tr)) = " (mx/item score)
              "   <- THIS is where the device runs"))
(println (str "  (mx/item derived)     = " (mx/item derived)))

;; Compare a couple of address values, all forced at the boundary only:
(def mu-v (cm/get-value (cm/get-submap (:choices tr) :mu)))
(def y-v  (cm/get-value (cm/get-submap (:choices tr) :y)))
(println (str "  sampled mu (MxArray? " (mx/array? mu-v) "), forced = " (mx/item mu-v)))
(println (str "  sampled y  (MxArray? " (mx/array? y-v)  "), forced = " (mx/item y-v)))

;; ─────────────────────────────────────────────────────────────────────────
;; (c) Narration: the value-semantics story reaches all the way to Metal.
;; ─────────────────────────────────────────────────────────────────────────
(println "\n=== (c) Why this matters ===")
(println "  Layer A (CLJS data) and Layer B (MLX lazy graph) are BOTH values.")
(println "  Scores, weights, posteriors compose by building graph nodes —")
(println "  referentially transparent, no hidden device side effects.")
(println "  mx/eval!/mx/item is the SOLE boundary that dispatches to the GPU.")
(println "  So 'purely functional' holds end-to-end: the device computation")
(println "  itself is just an immutable value until you choose to force it.")

(println "\n=== done ===")
