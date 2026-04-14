(ns genmlx.mlx
  "The membrane between pure ClojureScript and MLX's GPU compute.

   GenMLX has a three-layer purity architecture:

     Layer A: Pure ClojureScript — GFI, handlers, inference (values)
     Layer B: Pure MLX Graphs    — lazy computation descriptions (also values!)
     Layer C: GPU Execution      — eval! dispatches the graph to Metal (side effect)

   This namespace is the membrane between Layers A and B. MLX operations build
   lazy computation graphs — (mx/add a b) returns a graph node, not a computed
   result. No GPU work occurs until eval! is called. This means most of this
   namespace is purely functional: graph construction is value manipulation.

   The only effectful operations are in the 'Effectful Operations' section:
   eval!, materialize!, item, ->clj, realize, async-eval!, training-step!.
   Everything else builds lazy graph nodes or reads metadata.

   Most ops are direct references to Rust NAPI exports (genmlx.rs).
   Rust's Either<&MxArray, f64> handles both array and JS number inputs,
   so no type coercion is needed on the ClojureScript side.")

;; =========================================================================
;; Module loading
;; =========================================================================

(defonce ^:private c (js/require "@mlx-node/core"))
(defonce ^:private M (.-MxArray c))

;; Forward declarations for functions referenced before definition.
(declare scalar array shape array? astype)

;; Scope tracking atoms — defined here because functional combinators
;; (grad, value-and-grad) reference grad-depth before the Memory
;; Management section where they logically belong.
(def ^:private tidy-depth (atom 0))
(def ^:private grad-depth (atom 0))

;; Backward-compat alias (some files reference mx/core)
(def core c)

;; =========================================================================
;; Dtypes
;;
;; IMPORTANT: MLX on Apple Silicon has no float64, int64, or bool dtypes.
;; The aliases below silently map to lower-precision types:
;;   float64 → float32  (loses precision above ~7 decimal digits)
;;   int64   → int32    (max value 2^31-1 instead of 2^63-1)
;;   bool    → int32    (0/1 representation)
;; Code using mx/float64 will get float32 arrays with NO runtime warning.
;; This matches MLX's hardware constraints — Apple GPUs operate in float32.
;; =========================================================================

(def float32 (.-Float32 (.-DType c)))
(def float64 (.-Float32 (.-DType c)))  ;; SILENT ALIAS: MLX has no float64
(def int32   (.-Int32   (.-DType c)))
(def int64   (.-Int32   (.-DType c)))  ;; SILENT ALIAS: MLX has no int64
(def bool-dt (.-Int32   (.-DType c)))  ;; SILENT ALIAS: MLX has no bool

;; =========================================================================
;; Internal helpers
;; =========================================================================

(defn- ->js
  "Convert clj collection to JS array, or pass nil through.
   Single numbers are wrapped in a 1-element array (for axes params)."
  [x]
  (when x
    (if (number? x) #js [x] (clj->js x))))

(defn- flatten-nested
  "Recursively flatten a nested collection to a flat vector of numbers."
  [coll]
  (if (or (vector? coll) (seq? coll) (sequential? coll) (js/Array.isArray coll))
    (let [first-el (first coll)]
      (if (or (vector? first-el) (seq? first-el) (sequential? first-el) (js/Array.isArray first-el))
        (into [] (mapcat flatten-nested coll))
        (vec coll)))
    [coll]))

(defn- infer-shape
  "Infer shape from nested collection. Returns [flat-data shape-vec]."
  [coll]
  (if (or (vector? coll) (seq? coll) (sequential? coll))
    (let [first-el (first coll)]
      (if (or (vector? first-el) (seq? first-el) (sequential? first-el))
        (let [[_ inner-shape] (infer-shape first-el)]
          [(flatten-nested coll) (into [(count coll)] inner-shape)])
        [(vec coll) [(count coll)]]))
    [[coll] []]))

(defn- unflatten
  "Reconstruct nested vector from flat data and shape."
  [flat-arr sh]
  (cond
    (empty? sh) (first flat-arr)
    (= 1 (count sh)) (vec flat-arr)
    :else
    (let [chunk-size (reduce * (rest sh))
          n (first sh)]
      (mapv (fn [i]
              (unflatten (subvec flat-arr (* i chunk-size) (* (inc i) chunk-size))
                         (vec (rest sh))))
            (range n)))))

;; =========================================================================
;; PURE GRAPH OPERATIONS (Layer B)
;;
;; Everything below until the "Effectful Operations" section builds lazy
;; MLX computation graph nodes. No GPU dispatch, no side effects.
;; (mx/add a b) returns a graph node describing "a + b" — a value.
;; =========================================================================

;; --- Unary ops (direct Rust NAPI references) ---

(def exp        (.-exp c))
(def expm1      (.-expm1 c))
(def log        (.-log c))
(def log2       (.-log2 c))
(def log10      (.-log10 c))
(def log1p      (.-log1p c))
(def sin        (.-sin c))
(def cos        (.-cos c))
(def tan        (.-tan c))
(def arccos     (.-arccos c))
(def tanh       (.-tanh c))
(def sigmoid    (.-sigmoid c))
(def erf        (.-erf c))
(def erfinv     (.-erfinv c))
(def lgamma     (.-lgamma c))
(def digamma    (.-digamma c))
(def bessel-i0e (.-besselI0e c))
(def bessel-i1e (.-besselI1e c))
(def floor      (.-floor c))
(def ceil       (.-ceil c))
(def round      (.-round c))
(def negative   (.-negative c))
(def square     (.-square c))
(def sqrt       (.-sqrt c))
(def abs        (.-abs c))
(def sign       (.-sign c))
(def reciprocal (.-reciprocal c))
(def flatten    (.-flatten c))
(def isnan      (.-isnan c))
(def isinf      (.-isinf c))

;; --- Binary ops (direct Rust NAPI references, non-variadic) ---

(def logaddexp    (.-logaddexp c))
(def divide       (.-div c))
(def power        (.-power c))
(def maximum      (.-maximum c))
(def minimum      (.-minimum c))
(def floor-divide (.-floorDivide c))
(def remainder    (.-remainder c))
(def matmul       (.-matmul c))
(def inner        (.-inner c))
(def outer        (.-outer c))

;; Variadic arithmetic -- CLJS reduce over Rust binary ops.
(def ^:private add* (.-add c))
(defn add
  ([a b] (add* a b))
  ([a b & more] (reduce add* (add* a b) more)))

(def ^:private sub* (.-sub c))
(defn subtract
  ([a b] (sub* a b))
  ([a b & more] (reduce sub* (sub* a b) more)))

(def ^:private mul* (.-mul c))
(defn multiply
  ([a b] (mul* a b))
  ([a b & more] (reduce mul* (mul* a b) more)))

;; --- Comparison / selection ---

(def equal         (.-equal c))
(def not-equal     (.-notEqual c))
(def greater       (.-greater c))
(def greater-equal (.-greaterEqual c))
(def less          (.-less c))
(def less-equal    (.-lessEqual c))
(def where         (.-where c))

;; Model-level helpers -- auto-promote integers, return float32.
(defn eq?  [a b] (.astype (equal (if (number? a) (scalar a int32) a)
                                 (if (number? b) (scalar b int32) b)) float32))
(defn neq? [a b] (.astype (not-equal (if (number? a) (scalar a int32) a)
                                     (if (number? b) (scalar b int32) b)) float32))
(defn gt?  [a b] (.astype (greater (if (number? a) (scalar a) a)
                                   (if (number? b) (scalar b) b)) float32))
(defn lt?  [a b] (.astype (less (if (number? a) (scalar a) a)
                                (if (number? b) (scalar b) b)) float32))
(defn and* [a b] (multiply a b))
(defn or*  [a b] (maximum a b))

;; --- Reductions (thin wrappers for clj->js axes conversion) ---

(def ^:private sum*       (.-sum c))
(def ^:private prod*      (.-prod c))
(def ^:private mean*      (.-mean c))
(def ^:private var*       (.-var c))
(def ^:private std*       (.-std c))
(def ^:private max*       (.-max c))
(def ^:private min*       (.-min c))
(def ^:private all*       (.-all c))
(def ^:private any*       (.-any c))
(def ^:private logsumexp* (.-logsumexp c))

(defn sum
  ([a] (sum* a))
  ([a axes] (sum* a (->js axes)))
  ([a axes keepdims] (sum* a (->js axes) keepdims)))
(defn prod
  ([a] (prod* a))
  ([a axes] (prod* a (->js axes))))
(defn mean
  ([a] (mean* a))
  ([a axes] (mean* a (->js axes))))
(defn variance
  ([a] (var* a))
  ([a axes] (var* a (->js axes))))
(defn std
  ([a] (std* a))
  ([a axes] (std* a (->js axes))))
(defn amax
  ([a] (max* a))
  ([a axes] (max* a (->js axes))))
(defn amin
  ([a] (min* a))
  ([a axes] (min* a (->js axes))))
(defn all
  ([a] (all* a))
  ([a axis] (all* a #js [axis])))
(defn any
  ([a] (any* a))
  ([a axis] (any* a #js [axis])))
(defn logsumexp
  ([a] (logsumexp* a))
  ([a axes] (logsumexp* a (->js axes)))
  ([a axes keepdims] (logsumexp* a (->js axes) keepdims)))

(def ^:private argmax* (.-argmax c))
(def ^:private argmin* (.-argmin c))
(defn argmax
  ([a] (argmax* a))
  ([a axis] (argmax* a axis)))
(defn argmin
  ([a] (argmin* a))
  ([a axis] (argmin* a axis)))

(def ^:private argsort* (.-argsort c))
(defn argsort
  ([a] (argsort* a))
  ([a axis] (argsort* a axis)))

(defn searchsorted
  ([sorted-arr values] (.searchsorted c sorted-arr values))
  ([sorted-arr values side] (.searchsorted c sorted-arr values (= side :right))))

(def ^:private sort* (.-sort c))
(defn sort-arr
  ([a] (sort* a))
  ([a axis] (sort* a axis)))

(defn topk [a k] (.topk c a k))

(def ^:private cumsum* (.-cumsum c))
(defn cumsum
  ([a] (cumsum* a))
  ([a axis] (cumsum* a axis)))

(def ^:private logcumsumexp* (.-logcumsumexp c))
(defn logcumsumexp
  ([a] (logcumsumexp* a))
  ([a axis] (logcumsumexp* a axis)))

;; --- Shape manipulation (thin wrappers for clj->js shape/axes conversion) ---

(defn reshape    [a sh]   (.reshape c a (clj->js sh)))
(defn squeeze
  ([a]      (.squeeze c a))
  ([a axes] (.squeeze c a (->js (vec axes)))))
(defn expand-dims [a axis] (.expandDims c a axis))
(defn transpose
  ([a]      (.transpose c a))
  ([a axes] (.transpose c a (->js axes))))
(defn broadcast-to [a sh] (.broadcastTo c a (clj->js sh)))
(defn tile       [a reps] (.tile c a (clj->js reps)))
(defn repeat-arr [a repeats axis] (.repeat c a repeats axis))
(defn stack
  ([arrs]      (.stack c (to-array arrs)))
  ([arrs axis] (.stack c (to-array arrs) axis)))
(defn concatenate
  ([arrs]      (.concatenate c (to-array arrs)))
  ([arrs axis] (.concatenate c (to-array arrs) axis)))
(defn split-arr
  ([a sections]      (vec (.split c a sections)))
  ([a sections axis] (vec (.split c a sections axis))))

;; --- Indexing ---

(defn take-idx
  ([a indices]
   (.take c a (if (number? indices) (scalar indices int32) indices) 0))
  ([a indices axis]
   (.take c a (if (number? indices) (scalar indices int32) indices) axis)))

(defn idx
  ([a i]      (take-idx a (if (number? i) (scalar i int32) i) 0))
  ([a i axis] (take-idx a (if (number? i) (scalar i int32) i) axis)))

(defn take-along-axis [a indices axis]
  (.takeAlongAxis c a indices axis))

(defn index [a i]
  (.take c a (if (number? i) (scalar i int32) i) 0))

(defn slice
  ([a start stop]
   (.slice c a (clj->js [start]) (clj->js [stop])))
  ([a start stop step]
   (let [sliced (.slice c a (clj->js [start]) (clj->js [stop]))]
     (if (= step 1)
       sliced
       (let [n (first (shape sliced))
             indices (array (vec (range 0 n step)) int32)]
         (.take c sliced indices 0))))))

(defn mat-get [a i j]
  (let [row (.take c a (scalar i int32) 0)]
    (.take c row (scalar j int32) 0)))

;; --- Array construction (creates graph leaf nodes — also pure) ---

(defn scalar
  "Create a scalar MLX array. Always float32 by default."
  ([v]   (.scalar c v))
  ([v dtype]
   (if (= dtype int32)
     (.scalarInt c (int v))
     (let [arr (.scalar c v)]
       (if (= dtype float32) arr (.astype c arr dtype))))))

(defn array
  ([v]
   (cond
     (array? v) v
     (or (vector? v) (seq? v) (sequential? v))
     (let [[flat-data sh] (infer-shape v)
           f32 (js/Float32Array.from (clj->js flat-data))]
       (.fromFloat32 c f32 (clj->js sh)))
     (js/Array.isArray v)
     (let [f32 (js/Float32Array.from v)]
       (.fromFloat32 c f32 #js [(.-length f32)]))
     :else (scalar v)))
  ([v shape-or-dtype]
   (if (or (vector? shape-or-dtype) (seq? shape-or-dtype))
     (let [[flat-data _] (infer-shape v)
           f32 (js/Float32Array.from (clj->js flat-data))]
       (.fromFloat32 c f32 (clj->js shape-or-dtype)))
     (let [[flat-data sh] (infer-shape v)]
       (if (= shape-or-dtype int32)
         (.fromInt32 c (js/Int32Array.from (clj->js flat-data)) (clj->js sh))
         (let [f32 (js/Float32Array.from (clj->js flat-data))
               arr (.fromFloat32 c f32 (clj->js sh))]
           (if (= shape-or-dtype float32) arr (.astype c arr shape-or-dtype)))))))
  ([v shape-vec dtype]
   (let [[flat-data _] (infer-shape v)]
     (if (= dtype int32)
       (.fromInt32 c (js/Int32Array.from (clj->js flat-data)) (clj->js shape-vec))
       (let [f32 (js/Float32Array.from (clj->js flat-data))
             arr (.fromFloat32 c f32 (clj->js shape-vec))]
         (if (= dtype float32) arr (.astype c arr dtype)))))))

(defn astype [a dtype] (.astype c a dtype))

(defn zeros
  ([sh]       (.zeros c (clj->js sh)))
  ([sh dtype] (.zeros c (clj->js sh) dtype)))
(defn ones
  ([sh]       (.ones c (clj->js sh)))
  ([sh dtype] (.ones c (clj->js sh) dtype)))
(defn full [sh val] (.full c (clj->js sh) val))
(defn eye
  ([n]       (.eye c n))
  ([n dtype] (.eye c n nil nil dtype)))
(defn arange
  ([stop]            (.arange c 0 stop))
  ([start stop]      (.arange c start stop))
  ([start stop step] (.arange c start stop step)))
(defn linspace [start stop num] (.linspace c start stop num))

(defn meshgrid [a b]
  (let [sa (shape a) sb (shape b)
        na (first sa) nb (first sb)
        a-col (.broadcastTo c (.reshape c a (clj->js [na 1])) (clj->js [na nb]))
        b-row (.broadcastTo c (.reshape c b (clj->js [1 nb])) (clj->js [na nb]))]
    #js [a-col b-row]))

;; --- Neural network ops ---

(defn softmax
  ([a]      (.softmax c a))
  ([a axis] (.softmax c a axis)))

(def clip (.-clip c))
(def nan-to-num (.-nanToNum c))
(def stop-gradient (.-stopGradient c))

;; =========================================================================
;; QUERY OPERATIONS (no side effects, no graph construction)
;;
;; These read array metadata without triggering evaluation or building
;; new graph nodes.
;; =========================================================================

(defn shape [a] (vec (.shapeOf c a)))
(defn ndim  [a] (.ndimOf c a))
(defn dtype [a] (.dtypeOf c a))
(defn size  [a] (js/Number (.sizeOf c a)))
(defn array? [x] (instance? M x))

;; =========================================================================
;; FUNCTIONAL COMBINATORS (Layer B — graph-to-graph transforms)
;;
;; Higher-order functions that transform functions. These are Layer B
;; operations: they produce new lazy graph builders. The results they
;; return are lazy graph nodes (not evaluated until eval!).
;; =========================================================================

(defn compile-fn
  "Identity pass-through. GenMLX's compilation uses noise transforms +
   the expression compiler (Level 1), not MLX's graph-caching compile.
   Direct use of MLX's compile would sever the autograd tape when model
   bodies contain eval!, returning silent zeros. The real graph caching
   happens in the Rust layer's compiled model forward passes.
   See: compiled.cljs, compiled_gen.cljs for the actual compilation strategy."
  ([f] f)
  ([f _shapeless?] f))

(defn compile-clear-cache!
  "Clear MLX's compiler cache (traced computation graphs).
   Effectful: frees GPU memory used by cached compiled graphs."
  [] (.compileClearCache c))

(defn vmap
  "Vectorized map: transforms f into a batched version that operates over
   an additional batch dimension. Pure graph-to-graph transformation."
  ([f]
   (fn [& args] (let [r (.vmap M f (to-array args))] (aget r 0))))
  ([f in-axes]
   (fn [& args] (let [r (.vmap M f (to-array args) (clj->js in-axes))] (aget r 0))))
  ([f in-axes out-axes]
   (fn [& args] (let [r (.vmap M f (to-array args) (clj->js in-axes) (clj->js out-axes))] (aget r 0)))))

(defn grad
  "Returns a function that computes gradients of f w.r.t. its arguments.
   The returned function builds a backward-pass graph lazily — gradient
   arrays are graph nodes until eval!'d."
  ([f]
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [grads (.computeGradients M f (to-array args))]
         (aget grads 0))
       (finally (swap! grad-depth dec)))))
  ([f argnums]
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [argnum-vec (if (vector? argnums) argnums [argnums])
             grads (.computeGradients M f (to-array args))]
         (if (= 1 (count argnum-vec))
           (aget grads (first argnum-vec))
           (mapv #(aget grads %) argnum-vec)))
       (finally (swap! grad-depth dec))))))

(defn value-and-grad
  "Returns a function that computes both f's value and its gradients.
   Results are lazy graph nodes until eval!'d."
  ([f]
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [result (.valueAndGrad M f (to-array args))]
         [(aget result 0) (aget result 1)])
       (finally (swap! grad-depth dec)))))
  ([f argnums]
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [result (.valueAndGrad M f (to-array args))
             v (aget result 0)
             argnum-vec (if (vector? argnums) argnums [argnums])
             g (if (= 1 (count argnum-vec))
                 (aget result (inc (first argnum-vec)))
                 (mapv #(aget result (inc %)) argnum-vec))]
         [v g])
       (finally (swap! grad-depth dec))))))

(defn jvp [f primals tangents]
  (let [result-val (apply f primals)
        eps 1e-5
        perturbed (mapv (fn [p t] (add p (multiply (scalar eps) t)))
                        (vec primals) (vec tangents))
        result-perturbed (apply f perturbed)]
    [result-val (divide (subtract result-perturbed result-val) (scalar eps))]))

(defn vjp [f primals cotangents]
  (let [cotangents-arr (vec cotangents)
        surrogate (fn [& xs]
                    (let [result (apply f xs)]
                      (if (sequential? cotangents-arr)
                        (sum (multiply result (first cotangents-arr)))
                        (sum (multiply result cotangents-arr)))))
        result (.valueAndGrad M surrogate (to-array (vec primals)))
        fval (apply f (vec primals))
        grads (let [gs #js []]
                (dotimes [i (dec (.-length result))]
                  (.push gs (aget result (inc i))))
                gs)]
    [fval grads]))

;; =========================================================================
;; EFFECTFUL OPERATIONS (Layer C — triggers GPU dispatch)
;;
;; These are the ONLY functions that cause side effects: GPU computation,
;; memory materialization, or data extraction from GPU to CPU.
;; Everything above this section is pure graph construction or metadata.
;; =========================================================================

(defn item
  "Extract scalar value from a 0-d or 1-element array.
   EFFECTFUL: fused eval + read in one NAPI call."
  [a]
  (.item c a))

(defn ->clj
  "Evaluate an MLX array and convert to nested ClojureScript data.
   EFFECTFUL: triggers eval, then transfers data from GPU to CPU."
  [a]
  (.eval a)
  (let [sh (shape a)
        flat (if (= (.dtypeOf c a) int32)
               (vec (js->clj (.toInt32 a)))
               (vec (js->clj (.toFloat32 a))))]
    (unflatten flat sh)))

(defn eval!
  "Evaluate one or more MLX arrays, materializing the computation graph.
   EFFECTFUL: this is the primary GPU dispatch point. Traverses the lazy
   computation DAG and executes all pending operations on Metal."
  [& arrs]
  (let [valid (filterv some? arrs)]
    (when (seq valid)
      (.evalArrays c (to-array valid)))))

(defn materialize!
  "Evaluate MLX arrays. Safely ignores non-MxArray values.
   EFFECTFUL: triggers GPU dispatch via eval!."
  [& arrs]
  (let [mx-arrs (filterv array? arrs)]
    (when (seq mx-arrs)
      (apply eval! mx-arrs))))

;; =========================================================================
;; MEMORY MANAGEMENT — the mutable boundary
;;
;; This is the one part of the membrane that violates pure functional
;; design. MLX lazy graphs accumulate memory — without periodic cleanup,
;; long inference loops (MCMC, SMC, optimization) build unbounded graphs
;; that exhaust Metal memory. There is no Rust-side memory pressure
;; callback, so ClojureScript-side heuristics are the pragmatic solution.
;;
;; Mutable state:
;;   tidy-depth, grad-depth — atoms tracking scope nesting
;;   ops-since-check, gfi-ops-count — counters for cleanup heuristics
;;
;; Phase 3 eval audit (2026-04-14) found 464 effectful operations across
;; 73 files: 237 in hot loops (essential), 217 at boundaries (correct),
;; 0 questionable. The codebase is well-structured for lazy evaluation.
;; =========================================================================

;; --- Scope tracking (atoms defined in Module loading, above) ---

(defn in-grad? [] (pos? @grad-depth))
(defn in-tidy? [] (pos? @tidy-depth))

(def ^:private jsc
  (when (exists? js/Bun) (js/require "bun:jsc")))

(defn jsc-cleanup! []
  (when jsc
    (.releaseWeakRefs jsc)
    (.drainMicrotasks jsc)
    (.gcAndSweep jsc)))

(def ^:private gc-fn
  (or (when (exists? js/Bun) (.-gc js/Bun))
      (.-gc js/globalThis)))

(defn tidy [f]
  (swap! tidy-depth inc)
  (try
    (let [result (f)]
      result)
    (finally
      (swap! tidy-depth dec)
      (when (zero? @tidy-depth)
        (jsc-cleanup!)
        (.clearCache c)))))

;; --- Memory queries (read-only, no side effects) ---

(defn get-active-memory  [] (.getActiveMemory c))
(defn get-cache-memory   [] (.getCacheMemory c))
(defn get-peak-memory    [] (.getPeakMemory c))
(defn reset-peak-memory! [] (.resetPeakMemory c))
(defn get-wrappers-count
  "Returns active GPU memory in bytes (not a wrapper object count).
   Named for backward compatibility; use get-active-memory for clarity."
  [] (.getActiveMemory c))

(defn sweep-dead-arrays! []
  (when-not (in-tidy?)
    (jsc-cleanup!)
    (.clearCache c)))

;; --- Memory control ---

(defn set-memory-limit! [n] (.setMemoryLimit c n))
(defn set-cache-limit!  [n] (.setCacheLimit c n))
(defn set-wired-limit!  [n] (.setWiredLimit c n))
(defn clear-cache!      []  (.clearCache c))


(defn metal-is-available? [] (.metalIsAvailable c))
(defn metal-device-info []
  (let [info (js/JSON.parse (.metalDeviceInfo c))]
    {:architecture (or (.-architecture info) "apple")
     :device-name  (or (.-device_name info) "apple-gpu")
     :memory-size  (or (.-max_recommended_working_set_size info) 0)
     :max-buffer-length (or (.-max_buffer_length info) 0)
     :max-recommended-working-set-size (or (.-max_recommended_working_set_size info) 0)}))

(defn memory-report []
  {:active-bytes (get-active-memory)
   :cache-bytes  (get-cache-memory)
   :peak-bytes   (get-peak-memory)})

;; --- Cleanup heuristics (global mutable counters) ---
;;
;; auto-cleanup! fires every 50 ops when active memory > 512 MB.
;; gfi-cleanup! fires every 10 GFI ops when active memory > 128 MB.
;; Together they form a multi-granularity safety net for long inference loops.
;; Called from: runtime.cljs (every sample), dist/core.cljs (batch sampling),
;; dynamic.cljs (after every GFI operation).

(def ^:private resource-pressure-threshold (* 512 1024 1024))
(def ^:private ^:mutable ops-since-check 0)
(def ^:private check-interval 50)

(defn auto-cleanup!
  ([] (auto-cleanup! false))
  ([aggressive?]
   (set! ops-since-check (inc ops-since-check))
   (when (>= ops-since-check check-interval)
     (set! ops-since-check 0)
     (when (and (not (in-tidy?))
                (> (get-active-memory) resource-pressure-threshold))
       (when aggressive?
         (jsc-cleanup!)
         (when gc-fn (gc-fn)))
       (sweep-dead-arrays!)
       (clear-cache!)))))

(def ^:private ^:mutable gfi-ops-count 0)
(def ^:private gfi-cleanup-interval 10)
(def ^:private gfi-pressure-threshold (* 128 1024 1024))

(defn gfi-cleanup! []
  (set! gfi-ops-count (inc gfi-ops-count))
  (when (>= gfi-ops-count gfi-cleanup-interval)
    (set! gfi-ops-count 0)
    (when (> (get-active-memory) gfi-pressure-threshold)
      (jsc-cleanup!)
      (sweep-dead-arrays!)
      (clear-cache!))))

(defn realize [x] (eval! x) (item x))
(defn realize-clj [x] (eval! x) (->clj x))

(def ^:private cache-pressure-threshold (* 512 1024 1024))

(defn tidy-materialize [f]
  (let [r (tidy f)] (eval! r) r))

(defn tidy-run [f collect-fn]
  (let [result-vol (volatile! nil)]
    (tidy (fn []
            (let [result (f)
                  arrays (collect-fn result)]
              (when (seq arrays) (apply eval! arrays))
              (vreset! result-vol result)
              (to-array arrays))))
    (when (> (get-cache-memory) cache-pressure-threshold)
      (jsc-cleanup!)
      (when gc-fn (gc-fn))
      (sweep-dead-arrays!)
      (clear-cache!))
    @result-vol))

(defn tidy-scalar [f]
  (let [result-vol (volatile! nil)]
    (tidy (fn []
            (let [arr (f)]
              (eval! arr)
              (vreset! result-vol (item arr))
              (to-array [arr]))))
    (when (> (get-cache-memory) cache-pressure-threshold)
      (jsc-cleanup!)
      (when gc-fn (gc-fn))
      (sweep-dead-arrays!)
      (clear-cache!))
    @result-vol))

(defn force-gc! []
  (jsc-cleanup!)
  (when gc-fn (gc-fn))
  (sweep-dead-arrays!)
  (clear-cache!)
  (.compileClearCache c))

(def ^:private DEFAULT-CACHE-LIMIT (* 256 1024 1024))
(set-cache-limit! DEFAULT-CACHE-LIMIT)

(defn with-resource-guard [f]
  (let [prev-limit (set-cache-limit! 0)]
    (try (f)
         (finally (clear-cache!) (set-cache-limit! prev-limit)))))

;; --- Linear algebra ---

(defn diag [a] (.diag c a))
(defn trace-mat
  ([a] (.trace c a 0 0 1))
  ([a offset] (.trace c a offset 0 1))
  ([a offset ax1 ax2] (.trace c a offset ax1 ax2)))
(defn einsum [subscripts & arrays]
  (.einsum c subscripts (to-array arrays)))

(def cholesky        (.-cholesky c))
(def solve           (.-linalgSolve c))
(defn solve-triangular [a b upper] (.solveTriangular c a b upper))
(def inv             (.-linalgInv c))
(defn tri-inv [a upper] (.triInv c a upper))
(defn cholesky-inv
  ([a]       (.choleskyInv c a false))
  ([a upper] (.choleskyInv c a upper)))
(defn qr [a]
  (let [r (.qr c a)] [(aget r 0) (aget r 1)]))
(defn svd [a]
  (let [r (.svd c a)] [(aget r 0) (aget r 1) (aget r 2)]))
(defn eigh [a]
  (let [r (.eigh c a)] [(aget r 0) (aget r 1)]))
(def eigvalsh (.-eigvalsh c))
(defn norm
  ([a]     (.linalgNorm c a))
  ([a ord] (.linalgNorm c a ord)))

(defn logdet [a]
  (let [L (cholesky a)]
    (multiply (scalar 2.0) (sum (log (diag L))))))
(defn det [a]
  (let [L (cholesky a)]
    (power (prod (diag L)) (scalar 2))))

;; --- Utilities ---

(defn ensure-array
  "Wrap JS numbers as MLX scalars; pass through arrays, fns, keywords, maps."
  ([x]
   (cond
     (array? x) x
     (fn? x) x
     (keyword? x) x
     (map? x) x
     (or (vector? x) (seq? x) (sequential? x)) (array x)
     :else (scalar x)))
  ([x dtype]
   (cond
     (array? x) (if (= (.dtypeOf c x) dtype) x (astype x dtype))
     (fn? x) x
     (keyword? x) x
     (map? x) x
     (or (vector? x) (seq? x) (sequential? x)) (array x dtype)
     :else (scalar x dtype))))

(defn async-eval!
  "Asynchronously evaluate arrays. EFFECTFUL: dispatches to GPU."
  [& arrays]
  (let [promises (mapv #(.evalAsync %) (filter some? arrays))]
    (js/Promise.all (to-array promises))))

(defn training-step!
  "One training step: forward, backward, update, extract loss.
   EFFECTFUL: triggers eval on module, loss, and extracts scalar."
  [module optim vg-fn & inputs]
  (let [[loss grads] (apply vg-fn inputs)]
    (.update optim module grads)
    (eval! module)
    (eval! loss)
    (item loss)))

;; =========================================================================
;; CONFIGURATION — device, constants
;; =========================================================================

(defn default-device
  "Always returns \"gpu\". MLX on Apple Silicon uses Metal GPU for all
   array operations; there is no CPU array backend. This function does
   not query actual device state — it returns a hardcoded constant."
  [] "gpu")
(defn set-default-device!
  "No-op. MLX on Apple Silicon has no per-operation device selection —
   Metal GPU is always used. Retained for API compatibility with code
   that wraps operations in device-switching blocks."
  [_d] nil)
(def cpu "cpu")
(def gpu "gpu")

(def pi    (.-PI js/Math))
(def e-val (.-E js/Math))
(def inf   js/Infinity)
(def nan   js/NaN)

