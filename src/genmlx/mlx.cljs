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
   eval!, materialize!, item, ->clj, realize, async-eval!.
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
(declare scalar array shape array? astype force-gc! clear-cache! maybe-count-sweep!)

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
(def float16  (.-Float16  (.-DType c)))
(def bfloat16 (.-BFloat16 (.-DType c)))
(def uint32   (.-Uint32   (.-DType c)))  ;; categorical/token indices
(def uint8    (.-Uint8    (.-DType c)))

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
  (if (or (sequential? coll) (js/Array.isArray coll))
    (let [first-el (first coll)]
      (if (or (sequential? first-el) (js/Array.isArray first-el))
        (into [] (mapcat flatten-nested coll))
        (vec coll)))
    [coll]))

(defn- infer-shape
  "Infer shape from nested collection. Returns [flat-data shape-vec].
   Handles JS arrays (#js [...]) as well as Clojure collections — mirrors
   flatten-nested's predicates so the dtype/shape arities of `array` shape
   JS-array inputs correctly (not as 0-dim scalars)."
  [coll]
  (if (or (sequential? coll) (js/Array.isArray coll))
    (let [first-el (first coll)]
      (if (or (sequential? first-el) (js/Array.isArray first-el))
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
(defn- ->cmp
  "Coerce a number to a comparison operand; pass MLX arrays through.
   eq?/neq? promote integers (dt int32) for exact comparison; a non-integer
   number stays float32 even when dt is int32 — an int32 scalar would
   truncate it ((eq? arr 2.5) must not compare against 2)."
  ([x] (if (number? x) (scalar x) x))
  ([x dt] (cond
            (not (number? x)) x
            (and (= dt int32) (not (js/Number.isInteger x))) (scalar x)
            :else (scalar x dt))))

(defn eq?
  "Element-wise a == b. Promotes integer operands (int32) and returns a
   float32 mask (1.0/0.0), so it composes with float arithmetic."
  [a b] (.astype (equal (->cmp a int32) (->cmp b int32)) float32))
(defn neq?
  "Element-wise a != b. Promotes integer operands (int32) and returns a
   float32 mask (1.0/0.0)."
  [a b] (.astype (not-equal (->cmp a int32) (->cmp b int32)) float32))
(defn gt?
  "Element-wise a > b. Promotes scalars at the float32 default and returns a
   float32 mask (1.0/0.0)."
  [a b] (.astype (greater (->cmp a) (->cmp b)) float32))
(defn lt?
  "Element-wise a < b. Promotes scalars at the float32 default and returns a
   float32 mask (1.0/0.0)."
  [a b] (.astype (less (->cmp a) (->cmp b)) float32))
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
  "Insertion indices for values into the sorted sorted-arr. side defaults to
   :left (index of the first element >= value); :right gives the index past
   the last element <= value. side maps to the native right? boolean."
  ([sorted-arr values] (.searchsorted c sorted-arr values))
  ([sorted-arr values side] (.searchsorted c sorted-arr values (= side :right))))

(def ^:private sort* (.-sort c))
(defn sort-arr
  ([a] (sort* a))
  ([a axis] (sort* a axis)))

(defn topk
  "Return the k largest *values* of a (not indices, not a value/index pair),
   along the last axis. The result is partition-ordered, not sorted."
  [a k] (.topk c a k))

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
(defn broadcast-to
  "Broadcast a to shape sh, preserving a's dtype.
   Reconstructed via the (correct) add-broadcasting onto a zeros of the target
   shape: the v0.31.2 native broadcast_to mis-fills a size-1 source dim,
   producing [v 0 0 …] instead of [v v v …]."
  [a sh]
  (let [a (if (instance? M a) a (.scalar c a))]
    (add (.zeros c (clj->js sh) (.dtypeOf c a)) a)))
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

(defn- ensure-int-indices
  "Ensure indices are integer dtype (int32). MLX take/gather crashes with
   float32 indices — Metal kernels expect integer index types."
  [indices]
  (cond
    (number? indices)            (scalar indices int32)
    ;; int32 is dtype code 1; float32 is code 0. Cast non-int to int32.
    (= (.dtypeOf c indices) 1)   indices
    :else                        (.astype indices int32)))

(defn take-idx
  "Gather elements of a at indices along axis (0-indexed; defaults to axis 0).
   indices are coerced to int32 — MLX gather crashes on float index dtypes."
  ([a indices]
   (.take c a (ensure-int-indices indices) 0))
  ([a indices axis]
   (.take c a (ensure-int-indices indices) axis)))

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

;; Catchable MLX resource exhaustion (bean genmlx-5ucd). The mlx-node shim now
;; catches a Metal allocation throw (e.g. "[metal::malloc] Resource limit (499000)
;; exceeded") and returns it as a thrown napi error instead of aborting the whole
;; process. Such an error is almost always dead-but-unfinalized MxArray wrappers
;; pinning their Metal buffers: force-gc! finalizes them (frees the buffers) and
;; clear-cache! drops the Metal cache, so a retry typically succeeds. Genuine
;; exhaustion rethrows for the caller to handle.
(defn- mlx-resource-error?
  [e]
  (let [msg (str (or (.-message e) e))]
    (boolean (or (re-find #"Resource limit" msg)
                 (re-find #"metal::malloc" msg)
                 (re-find #"out of memory" msg)))))

;; Layer 2 instrumentation (bean genmlx-x7cl). alloc-retry-count = times the
;; REACTIVE catch below fired (the ~499000 buffer wall WAS hit and we recovered).
;; proactive-sweep-count = times the COUNT-aware sweep pre-empted the wall (see
;; maybe-count-sweep! / auto-cleanup! / gfi-cleanup!). In a healthy hot loop the
;; proactive sweep keeps the live count under the limit, so the reactive path
;; should never trigger: alloc-retry-count stays 0 while proactive-sweep-count
;; goes positive. Public so regression tests can observe both.
(def alloc-retry-count (atom 0))
(def proactive-sweep-count (atom 0))

(defn- with-alloc-retry
  "Run thunk; opportunistically run the proactive count-aware sweep on success.
   On an MLX resource-exhaustion error, reclaim dead GPU buffers and retry ONCE.
   Any other error (or a second failure) propagates."
  [thunk]
  (try
    (let [r (thunk)] (maybe-count-sweep!) r)
    (catch :default e
      (if (mlx-resource-error? e)
        (do (swap! alloc-retry-count inc) (force-gc!) (clear-cache!) (thunk))
        (throw e)))))

(defn scalar
  "Create a scalar MLX array. Always float32 by default."
  ([v]   (with-alloc-retry #(.scalar c v)))
  ([v dtype]
   (with-alloc-retry
     #(if (= dtype int32)
        (.scalarInt c (int v))
        (let [arr (.scalar c v)]
          (if (= dtype float32) arr (.astype c arr dtype)))))))

(defn array
  ([v]
   (cond
     (array? v) v
     ;; JS arrays (#js [...]) included so NESTED JS arrays shape correctly via
     ;; infer-shape, not just flat ones. The old flat-only (js/Array.isArray v)
     ;; branch coerced #js [#js [..] #js [..]] to NaN — see infer-shape.
     (or (vector? v) (seq? v) (sequential? v) (js/Array.isArray v))
     (with-alloc-retry
       #(let [[flat-data sh] (infer-shape v)
              f32 (js/Float32Array.from (clj->js flat-data))]
          (.fromFloat32 c f32 (clj->js sh))))
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
  ;; Via broadcast-to, not native broadcastTo: the [na 1]->[na nb] and
  ;; [1 nb]->[na nb] expansions are exactly the size-1-source-dim case the
  ;; v0.31.2 native broadcast_to mis-fills (see broadcast-to).
  (let [sa (shape a) sb (shape b)
        na (first sa) nb (first sb)
        a-col (broadcast-to (.reshape c a (clj->js [na 1])) [na nb])
        b-row (broadcast-to (.reshape c b (clj->js [1 nb])) [na nb])]
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

(defn- grad-args
  "Marshal an arg vector for the autograd NAPI boundary. Every arg must be a
   concrete MLX value. The v0.31.2 binary rejects a nil arg (\"Failed to recover
   MxArray type from napi value\"), but silently substituting a 0-scalar is NOT
   safe: an arg that is not differentiated can still be consumed as DATA inside
   the function (e.g. a PRNG key threaded into value-and-grad), and a float
   0-scalar passed where a uint32 key is expected crashes Metal with a C++
   exception (SIGTRAP). So a nil arg is a caller bug — surface it loudly here,
   naming the offending index, rather than corrupting the computation downstream.
   (This intentionally retracts an earlier nil->0-scalar coercion whose \"inert\"
   premise was false: the nil was being consumed as a PRNG key. See genmlx-yo6y.)"
  [args]
  (to-array
   (vec (map-indexed
         (fn [i a]
           (when (nil? a)
             (throw (ex-info (str "value-and-grad/grad: argument " i " is nil. "
                                  "Pass a concrete MLX value — a nil PRNG key or "
                                  "placeholder is not supported (thread a real key "
                                  "via rng/ensure-key before the autograd boundary).")
                             {:arg-index i :args-count (count args)})))
           a)
         args))))

(defn- ->argnum-vec
  "Normalize an argnums spec (int or vector) to a vector of arg indices."
  [argnums]
  (if (vector? argnums) argnums [argnums]))

(defn grad
  "Returns a function that computes gradients of f w.r.t. its arguments.
   The returned function builds a backward-pass graph lazily — gradient
   arrays are graph nodes until eval!'d."
  ([f]
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [grads (.computeGradients M f (grad-args args))]
         (aget grads 0))
       (finally (swap! grad-depth dec)))))
  ([f argnums]
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [argnum-vec (->argnum-vec argnums)
             grads (.computeGradients M f (grad-args args))]
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
       (let [result (.valueAndGrad M f (grad-args args))]
         [(aget result 0) (aget result 1)])
       (finally (swap! grad-depth dec)))))
  ([f argnums]
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [result (.valueAndGrad M f (grad-args args))
             v (aget result 0)
             argnum-vec (->argnum-vec argnums)
             g (if (= 1 (count argnum-vec))
                 (aget result (inc (first argnum-vec)))
                 (mapv #(aget result (inc %)) argnum-vec))]
         [v g])
       (finally (swap! grad-depth dec))))))

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
  (with-alloc-retry #(.item c a)))

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
  ;; Keep only MxArrays (not just non-nil): a non-array value (a JS number,
  ;; a collection) reaching .evalArrays is rejected by the v0.31.2 binary
  ;; ("Failed to recover MxArray type from napi value"). The old binary
  ;; silently ignored them; this restores that, matching materialize!.
  (let [valid (filterv array? arrs)]
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
    (f)
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

(defn get-num-resources
  "Live Metal buffer allocation count (active + cached). Each counts toward the
   macOS resource limit (~499000). Read by the Layer-2 proactive sweep."
  [] (.getNumResources c))
(defn get-resource-limit
  "The Metal buffer resource limit — the count at which allocations fail."
  [] (.getResourceLimit c))
(defn get-buffer-count
  "Alias of get-num-resources (the bean-named API, genmlx-x7cl)."
  [] (.getNumResources c))

;; Proactive buffer-count sweep policy (bean genmlx-x7cl). The ~499000 count
;; limit is hit on tiny-array loops while MEMORY is trivially low, so the
;; memory-pressure heuristics below never fire for that class. We sweep when the
;; live count crosses ~80% of the limit, well before the wall. buffer-count-
;; threshold is a public atom so it can be tuned at runtime (and so regression
;; tests can lower/raise it deterministically). buffer-count-limit is resolved
;; once at load; a degraded-Metal host (query returns 0) falls back to 499000.
(def ^:private buffer-count-limit
  (let [l (try (get-resource-limit) (catch :default _ 0))]
    (if (pos? l) l 499000)))
(def buffer-count-threshold (atom (long (* 0.8 buffer-count-limit))))
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

;; --- Layer 2 proactive count-aware sweep (bean genmlx-x7cl) ---
(def ^:private ^:mutable allocs-since-count-check 0)
(def ^:private count-check-interval 4096)
(def ^:private ^:mutable proactive-armed? true)

(defn- buffer-count-pressure?
  "Hysteretic buffer-count pressure check. Returns true (and DISARMS) when the
   live Metal buffer count first crosses the HIGH watermark @buffer-count-
   threshold. Stays false until the count later falls below the LOW watermark
   (0.75 * high), then RE-ARMS. The hysteresis stops force-gc! from firing every
   interval when a high count is due to LIVE buffers a sweep cannot reclaim (it
   would drop little, stay above LOW, and stay disarmed) — while still firing
   once per genuine dead-buffer climb (a sweep frees them, count drops below LOW,
   re-arm). Shared by all three sweep sites; they cooperate via this one state."
  []
  (let [n  (get-num-resources)
        hi @buffer-count-threshold
        lo (* 0.75 hi)]
    (cond
      (and proactive-armed? (> n hi))       (do (set! proactive-armed? false) true)
      (and (not proactive-armed?) (< n lo)) (do (set! proactive-armed? true) false)
      :else false)))

(defn- count-sweep!
  "Reclaim dead Metal buffers (proactive Layer-2 sweep). force-gc! finalizes
   dead MxArray wrappers — freeing their pinned buffers — then clear-cache! drops
   the Metal free-buffer cache."
  []
  (swap! proactive-sweep-count inc)
  (force-gc!)
  (clear-cache!))

(defn- maybe-count-sweep!
  "Called from the allocation/read boundary (with-alloc-retry). Every
   count-check-interval guarded ops, if the live Metal buffer count crosses the
   threshold (hysteretic), reclaim dead buffers BEFORE the ~499000 wall is hit.
   Gated on (not in-tidy?): a tidy scope does its own cleanup on exit, and a
   force-gc! mid-scope would call jsc-cleanup! out of turn."
  []
  (set! allocs-since-count-check (inc allocs-since-count-check))
  (when (>= allocs-since-count-check count-check-interval)
    (set! allocs-since-count-check 0)
    (when (and (not (in-tidy?)) (buffer-count-pressure?))
      (count-sweep!))))

(defn auto-cleanup!
  ([] (auto-cleanup! false))
  ([aggressive?]
   (set! ops-since-check (inc ops-since-check))
   (when (>= ops-since-check check-interval)
     (set! ops-since-check 0)
     (when-not (in-tidy?)
       (cond
         ;; Count pressure: tiny-array loops hit the ~499000 buffer wall while
         ;; memory stays trivially low, so this branch fires where the memory
         ;; branch below never would. Hysteretic — see buffer-count-pressure?.
         (buffer-count-pressure?)
         (count-sweep!)

         (> (get-active-memory) resource-pressure-threshold)
         (do (when aggressive?
               (jsc-cleanup!)
               (when gc-fn (gc-fn)))
             (sweep-dead-arrays!)
             (clear-cache!)))))))

(def ^:private ^:mutable gfi-ops-count 0)
(def ^:private gfi-cleanup-interval 10)
(def ^:private gfi-pressure-threshold (* 128 1024 1024))

(defn gfi-cleanup! []
  (set! gfi-ops-count (inc gfi-ops-count))
  (when (>= gfi-ops-count gfi-cleanup-interval)
    (set! gfi-ops-count 0)
    (cond
      ;; Count pressure first — covers GFI inference loops that churn tiny
      ;; arrays (the ex-crashers) at low memory. Hysteretic; see auto-cleanup!.
      (and (not (in-tidy?)) (buffer-count-pressure?))
      (count-sweep!)

      ;; Same in-tidy? gate as the count branch and auto-cleanup!: a tidy
      ;; scope does its own cleanup on exit, and jsc-cleanup!/clear-cache!
      ;; mid-scope is the mid-tidy GC the design forbids.
      (and (not (in-tidy?)) (> (get-active-memory) gfi-pressure-threshold))
      (do (jsc-cleanup!)
          (sweep-dead-arrays!)
          (clear-cache!)))))

(defn realize [x] (eval! x) (item x))
(defn realize-clj [x] (eval! x) (->clj x))

(def ^:private cache-pressure-threshold (* 512 1024 1024))

(defn- cleanup-if-cache-pressure!
  "Run a GC/cache sweep when the Metal cache exceeds the pressure threshold."
  []
  (when (> (get-cache-memory) cache-pressure-threshold)
    (jsc-cleanup!)
    (when gc-fn (gc-fn))
    (sweep-dead-arrays!)
    (clear-cache!)))

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
    (cleanup-if-cache-pressure!)
    @result-vol))

(defn tidy-scalar [f]
  (let [result-vol (volatile! nil)]
    (tidy (fn []
            (let [arr (f)]
              (eval! arr)
              (vreset! result-vol (item arr))
              (to-array [arr]))))
    (cleanup-if-cache-pressure!)
    @result-vol))

(defn force-gc! []
  (jsc-cleanup!)
  (when gc-fn (gc-fn))
  (sweep-dead-arrays!)
  (clear-cache!)
  (.compileClearCache c))

(def ^:private DEFAULT-CACHE-LIMIT (* 256 1024 1024))
(set-cache-limit! DEFAULT-CACHE-LIMIT)

(defn with-resource-guard
  "Caps the cache limit to 0 for the scope (aggressive buffer eviction), then
   clears the buffer cache and restores the limit on exit. Light cleanup — for
   eager/kernel inference drivers that do not accumulate buffers across calls."
  [f]
  (let [prev-limit (set-cache-limit! 0)]
    (try (f)
         (finally (clear-cache!) (set-cache-limit! prev-limit)))))

(defn with-resource-guard-gc
  "Like with-resource-guard but does a full force-gc! on exit (sweep-dead-arrays!
   + jsc-cleanup! + clear-cache! + compile-clear-cache!). The compiled-inference
   drivers (compiled-mh/mala/hmc) build many short-lived MLX arrays per call;
   plain clear-cache! frees the buffer *cache* but not buffers still held by
   dead-but-unswept arrays, so the Metal live-buffer COUNT (~499000, independent
   of memory) climbs across calls until a native C++ exception / SIGTRAP fires
   (see genmlx-5ucd). Sweeping dead arrays on each exit bounds the count. Reserved
   for the compiled paths so the eager/kernel paths (which provably do not
   accumulate this way) avoid the per-call GC cost. Degrades gracefully where
   gc-fn is nil (bunx nbb): the sweep + native clears still run."
  [f]
  (let [prev-limit (set-cache-limit! 0)]
    (try (f)
         (finally (force-gc!) (set-cache-limit! prev-limit)))))

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

(defn- array-conversion-error
  "Helpful error for values that can't become MLX arrays. The usual culprit is
   bare ClojureScript arithmetic (+,-,*,/) on MLX arrays, which yields a string
   or NaN instead of a tensor — the handler path executes real cljs ops."
  [x]
  (ex-info
   (str "Cannot convert " (pr-str x) " to an MLX array — expected an MLX array "
        "or a number. A common cause is bare ClojureScript arithmetic "
        "(+, -, *, /) on MLX arrays in a model body; use mx/add, mx/subtract, "
        "mx/multiply, or mx/divide instead — e.g. (mx/multiply slope x), not "
        "(* slope x).")
   {:error ::not-array-convertible
    :value x
    :hint "replace bare +,-,*,/ with mx/add,mx/subtract,mx/multiply,mx/divide"}))

(defn ensure-array
  "Wrap JS numbers as MLX scalars; pass through arrays, fns, keywords, maps.
   Booleans coerce to 0/1 floats — MLX has no bool dtype, and this matches how
   compiled branch conditions (mx/where) read a boolean model arg. Throws a
   helpful error on strings/nil/NaN — usually a sign that bare ClojureScript
   arithmetic was used on an MLX array instead of an mx/ op."
  ([x]
   (cond
     (array? x) x
     (fn? x) x
     (keyword? x) x
     (map? x) x
     (or (vector? x) (seq? x) (sequential? x)) (array x)
     (string? x) (throw (array-conversion-error x))
     (nil? x) (throw (array-conversion-error x))
     (and (number? x) (js/isNaN x)) (throw (array-conversion-error x))
     (boolean? x) (scalar (if x 1.0 0.0))
     :else (scalar x)))
  ([x dtype]
   (cond
     (array? x) (if (= (.dtypeOf c x) dtype) x (astype x dtype))
     (fn? x) x
     (keyword? x) x
     (map? x) x
     (or (vector? x) (seq? x) (sequential? x)) (array x dtype)
     (string? x) (throw (array-conversion-error x))
     (nil? x) (throw (array-conversion-error x))
     (and (number? x) (js/isNaN x)) (throw (array-conversion-error x))
     (boolean? x) (scalar (if x 1.0 0.0) dtype)
     :else (scalar x dtype))))

(defn async-eval!
  "Asynchronously evaluate arrays. EFFECTFUL: dispatches to GPU."
  [& arrays]
  (let [promises (mapv #(.evalAsync %) (filter some? arrays))]
    (js/Promise.all (to-array promises))))

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

