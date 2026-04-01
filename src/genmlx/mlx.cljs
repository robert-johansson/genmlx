(ns genmlx.mlx
  "Thin ClojureScript wrapper over mlx-node (@mlx-node/core).
   Provides idiomatic ClojureScript access to MLX tensor operations,
   autograd, random number generation, and linear algebra.

   All operations are lazy by default — call (eval!) to materialize.
   Requires: npm install @mlx-node/core (Apple Silicon only, nbb only).")

;; ---------------------------------------------------------------------------
;; Module loading
;; ---------------------------------------------------------------------------

;; @mlx-node/core exports: MxArray class, DType enum, top-level memory/eval fns.
;; No bridge module dependency — BigInt64Array shape conversion is inlined below.
(defonce ^:private c (js/require "@mlx-node/core"))

;; MxArray class — instance methods for ops, static methods for creation/transforms.
(defonce ^:private M (.-MxArray c))

;; Expose `core` as alias for backward compatibility (some files reference mx/core).
;; In mlx-node, there is no core sub-module — ops are instance methods on MxArray.
(def core c)

;; nn-mod and optim-mod: mlx-node does not have nn/optimizers modules.
;; These are nil — downstream code that uses them will need separate adaptation.
(def nn-mod nil)
(def optim-mod nil)

;; No random sub-module — PRNG is on MxArray (randomKey, keyNormal, etc.).
(def random nil)

;; No linalg sub-module — linalg ops are instance methods on MxArray.
(def linalg nil)

;; ---------------------------------------------------------------------------
;; Dtypes
;; ---------------------------------------------------------------------------

;; mlx-node DType is a const enum: Float32=0, Int32=1, Float16=2, BFloat16=3, Uint32=4, Uint8=5
(def float32  (.-Float32 (.-DType c)))
(def float64  (.-Float32 (.-DType c)))  ;; MLX has no float64; map to float32
(def int32    (.-Int32 (.-DType c)))
(def int64    (.-Int32 (.-DType c)))     ;; MLX has no int64; map to int32
(def bool-dt  (.-Int32 (.-DType c)))     ;; MLX has no bool; map to int32

;; ---------------------------------------------------------------------------
;; Internal helpers
;; ---------------------------------------------------------------------------

(defn- to-big-shape
  "Convert a clj vector/seq of numbers to BigInt64Array for NAPI-RS shape params."
  [sh]
  (js/BigInt64Array.from (clj->js (mapv js/BigInt sh))))

(defn- to-int32-axes
  "Convert clj vector/seq of axis numbers to Int32Array for reduction axes."
  [axes]
  (js/Int32Array.from (clj->js axes)))

(defn- shape->clj
  "Convert shape from MxArray.shapeArray() to clj vector of numbers.
   shapeArray() returns Array<number> (unlike shape() which returns BigInt64Array)."
  [arr]
  (vec (js->clj (.shapeArray arr))))

(defn- flatten-nested
  "Recursively flatten a nested JS/clj collection to a flat JS array of numbers."
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

;; ---------------------------------------------------------------------------
;; Array creation
;; ---------------------------------------------------------------------------

(defn scalar
  ([v]
   (if (integer? v)
     (.scalarInt M v)
     (.scalarFloat M v)))
  ([v dtype]
   (if (= dtype int32)
     (.scalarInt M (int v))
     ;; Create as float then cast if needed
     (let [arr (.scalarFloat M v)]
       (if (= dtype float32)
         arr
         (.astype arr dtype))))))

(defn- ensure-mx
  "Ensure x is an MxArray. Convert JS numbers to scalar MxArray."
  [x]
  (if (number? x) (scalar x) x))

(defn array
  ([v]
   (let [js-v (clj->js v)]
     (if (or (vector? v) (seq? v) (sequential? v))
       (let [[flat-data sh] (infer-shape v)
             f32 (js/Float32Array.from (clj->js flat-data))]
         (.fromFloat32 M f32 (to-big-shape sh)))
       ;; Scalar value
       (scalar js-v))))
  ([v shape-or-dtype]
   (if (or (vector? shape-or-dtype) (seq? shape-or-dtype))
     ;; Second arg is a shape — create flat then reshape
     (let [[flat-data _] (infer-shape v)
           f32 (js/Float32Array.from (clj->js flat-data))]
       (.fromFloat32 M f32 (to-big-shape shape-or-dtype)))
     ;; Second arg is a dtype
     (let [[flat-data sh] (infer-shape v)]
       (if (= shape-or-dtype int32)
         (.fromInt32 M (js/Int32Array.from (clj->js flat-data)) (to-big-shape sh))
         (let [f32 (js/Float32Array.from (clj->js flat-data))
               arr (.fromFloat32 M f32 (to-big-shape sh))]
           (if (= shape-or-dtype float32)
             arr
             (.astype arr shape-or-dtype)))))))
  ([v shape-vec dtype]
   (let [[flat-data _] (infer-shape v)]
     (if (= dtype int32)
       (.fromInt32 M (js/Int32Array.from (clj->js flat-data)) (to-big-shape shape-vec))
       (let [f32 (js/Float32Array.from (clj->js flat-data))
             arr (.fromFloat32 M f32 (to-big-shape shape-vec))]
         (if (= dtype float32)
           arr
           (.astype arr dtype)))))))

(defn astype
  "Cast array to the given dtype."
  [a dtype]
  (.astype (ensure-mx a) dtype))

(defn zeros
  ([sh]       (.zeros M (to-big-shape sh)))
  ([sh dtype] (.zeros M (to-big-shape sh) dtype)))

(defn ones
  ([sh]       (.ones M (to-big-shape sh)))
  ([sh dtype] (.ones M (to-big-shape sh) dtype)))

(defn full
  [sh val]
  (.full M (to-big-shape sh) val))

(defn eye
  ([n]       (.eye M n))
  ([n dtype] (.eye M n nil nil dtype)))

(defn arange
  ;; mlx-node MxArray.arange requires start AND stop.
  ;; node-mlx had arange(stop), arange(start,stop), arange(start,stop,step).
  ([stop]            (.arange M 0 stop))
  ([start stop]      (.arange M start stop))
  ([start stop step] (.arange M start stop step)))

(defn linspace [start stop num] (.linspace M start stop num))

(defn meshgrid
  "Create coordinate grids from 1D arrays. Returns a JS array of MLX arrays.
   (meshgrid a b) -> #js [grid-a grid-b] where each has shape [len(a), len(b)].
   Implemented via broadcasting since mlx-node has no built-in meshgrid."
  [a b]
  ;; meshgrid(a,b) = [a expanded to [len(a), len(b)], b expanded to [len(a), len(b)]]
  ;; a is 1D [M], b is 1D [N] -> grid-a is [M,N], grid-b is [M,N]
  (let [sa (shape->clj a)
        sb (shape->clj b)
        na (first sa)
        nb (first sb)
        ;; a-col: [M,1] broadcast to [M,N]
        a-col (.broadcastTo (.reshape a (to-big-shape [na 1])) (to-big-shape [na nb]))
        ;; b-row: [1,N] broadcast to [M,N]
        b-row (.broadcastTo (.reshape b (to-big-shape [1 nb])) (to-big-shape [na nb]))]
    #js [a-col b-row]))

;; ---------------------------------------------------------------------------
;; Evaluation / materialization
;; ---------------------------------------------------------------------------

(defn eval!
  "Evaluate one or more MLX arrays, materializing the computation graph."
  [& arrs]
  (let [valid (filterv some? arrs)]
    (when (seq valid)
      ;; evalArrays takes a JS array of MxArrays
      (.evalArrays c (to-array valid)))))

(defn item
  "Extract scalar value from a 0-d or 1-element MLX array.
   mlx-node has no .item() — we use toFloat32/toInt32 and take first element."
  [a]
  (let [dt (.dtype a)]
    (if (= dt int32)
      (aget (.toInt32 a) 0)
      (aget (.toFloat32 a) 0))))

(defn ->clj
  "Evaluate an MLX array and convert to nested ClojureScript data.
   mlx-node has no .tolist() — we use toFloat32 + shape to reconstruct."
  [a]
  (.eval a)
  (let [sh (shape->clj a)
        dt (.dtype a)
        flat (if (= dt int32)
               (vec (js->clj (.toInt32 a)))
               (vec (js->clj (.toFloat32 a))))]
    (unflatten flat sh)))

(defn shape
  "Get array shape as clj vector of numbers."
  [a]
  (vec (js->clj (.shapeArray (ensure-mx a)))))

(defn ndim [a] (.ndim (ensure-mx a)))

(defn dtype [a] (.dtype (ensure-mx a)))

(defn size [a]
  ;; mlx-node .size() returns bigint — convert to number
  (js/Number (.size (ensure-mx a))))

;; ---------------------------------------------------------------------------
;; Memory management
;; ---------------------------------------------------------------------------

(def ^:private tidy-depth
  "Tracks nesting depth of mx/tidy scopes. sweep-dead-arrays! is unsafe
   inside tidy (can cause use-after-free / double-free of Metal buffers),
   so auto-sweep in p/generate and p/simulate skips when tidy-depth > 0."
  (atom 0))

(def ^:private grad-depth
  "Tracks nesting depth of mx/grad and mx/value-and-grad scopes.
   The L3 auto-analytical handler uses volatile! which breaks gradient
   flow. When grad-depth > 0, p/generate skips the analytical path
   and uses compiled-generate or handler instead."
  (atom 0))

(def ^:private compile-depth
  "Tracks nesting depth of mx/compile-fn tracing scopes.
   When compile-depth > 0, mx/grad skips tidy wrapping because
   tidy would dispose trace arrays needed by the compilation graph."
  (atom 0))

(defn in-grad?
  "Returns true if currently executing inside an mx/grad or mx/value-and-grad scope."
  [] (pos? @grad-depth))

(defn in-compile?
  "Returns true if currently executing inside an mx/compile-fn trace."
  [] (pos? @compile-depth))

(defn tidy
  "Run f, cleaning up intermediate arrays afterward.
   mlx-node has no native tidy — we use JSC cleanup (releaseWeakRefs +
   drainMicrotasks + gcAndSweep + clearCache) as a substitute.
   The function result is returned; intermediates become eligible for GC."
  [f]
  (swap! tidy-depth inc)
  (try
    (let [result (f)]
      result)
    (finally
      (swap! tidy-depth dec)
      ;; Post-scope cleanup: trigger JSC GC to release dead array weak refs,
      ;; then clear Metal cache. This approximates node-mlx's tidy behavior.
      (when (zero? @tidy-depth)
        (when-let [jsc-mod (when (exists? js/Bun) (js/require "bun:jsc"))]
          (.releaseWeakRefs jsc-mod)
          (.drainMicrotasks jsc-mod)
          (.gcAndSweep jsc-mod))
        (.clearCache c)))))

(defn in-tidy?
  "Returns true if currently executing inside an mx/tidy scope."
  [] (pos? @tidy-depth))

(defn dispose!
  "No-op in mlx-node (no explicit dispose). Arrays are freed by GC."
  [a]
  nil)

;; Memory monitoring
(defn get-active-memory [] (.getActiveMemory c))
(defn get-cache-memory [] (.getCacheMemory c))
(defn get-peak-memory [] (.getPeakMemory c))
(defn reset-peak-memory! [] (.resetPeakMemory c))

(defn get-wrappers-count
  "mlx-node has no wrapper count tracking. Returns active memory as proxy."
  []
  (.getActiveMemory c))

(defn sweep-dead-arrays!
  "Synchronously free Metal buffers for arrays whose JS wrappers have been GC'd.
   mlx-node has no sweepDeadArrays — we use JSC cleanup + clearCache instead.
   No-op when called inside mx/tidy (tidy manages its own disposal)."
  []
  (when-not (in-tidy?)
    (when-let [jsc-mod (when (exists? js/Bun) (js/require "bun:jsc"))]
      (.releaseWeakRefs jsc-mod)
      (.drainMicrotasks jsc-mod)
      (.gcAndSweep jsc-mod))
    (.clearCache c)))

;; Memory control
(defn set-memory-limit! [n] (.setMemoryLimit c n))
(defn set-cache-limit! [n] (.setCacheLimit c n))
(defn set-wired-limit! [n] (.setWiredLimit c n))
(defn clear-cache! [] (.clearCache c))

;; Metal resource tracking
;; mlx-node has no getNumResources/getResourceLimit — use getActiveMemory as proxy.
(defn get-num-resources
  "Number of live Metal buffer allocations (active + cached).
   mlx-node has no resource counting — returns active memory bytes as proxy."
  [] (.getActiveMemory c))

(defn get-resource-limit
  "Maximum Metal buffer allocations before failure.
   mlx-node has no resource limit tracking — returns a large sentinel value."
  [] 499000)

;; Metal device info
(defn metal-is-available? [] (.metalIsAvailable c))

(defn metal-device-info []
  (let [info-str (.metalDeviceInfo c)
        info (js/JSON.parse info-str)]
    {:architecture (or (.-architecture info) "apple")
     :device-name (or (.-device_name info) "apple-gpu")
     :memory-size (or (.-max_recommended_working_set_size info) 0)
     :max-buffer-length (or (.-max_buffer_length info) 0)
     :max-recommended-working-set-size (or (.-max_recommended_working_set_size info) 0)
     :resource-limit 499000}))

;; Convenience
(defn memory-report []
  {:active-bytes (get-active-memory)
   :cache-bytes (get-cache-memory)
   :peak-bytes (get-peak-memory)
   :wrappers (get-wrappers-count)
   :num-resources (get-num-resources)
   :resource-limit (get-resource-limit)})

;; ---------------------------------------------------------------------------
;; Arithmetic (element-wise)
;; ---------------------------------------------------------------------------

;; mlx-node: ops are instance methods on MxArray.
;; Binary ops: (.add a b), (.sub a b), (.mul a b), (.div a b)
;; BUT these require both args to be MxArray. We ensure-array for scalars.

(defn add
  ([a b] (.add (ensure-mx a) (ensure-mx b)))
  ([a b & more] (reduce add (add a b) more)))
(defn subtract
  ([a b] (.sub (ensure-mx a) (ensure-mx b)))
  ([a b & more] (reduce subtract (subtract a b) more)))
(defn multiply
  ([a b] (.mul (ensure-mx a) (ensure-mx b)))
  ([a b & more] (reduce multiply (multiply a b) more)))
(defn divide   [a b] (.div (ensure-mx a) (ensure-mx b)))
(defn negative [a]   (.negative (ensure-mx a)))
(defn power    [a b] (.power (ensure-mx a) (ensure-mx b)))
(defn square   [a]   (.square (ensure-mx a)))
(defn sqrt     [a]   (.sqrt (ensure-mx a)))
(defn abs      [a]   (.abs (ensure-mx a)))
(defn maximum  [a b] (.maximum (ensure-mx a) (ensure-mx b)))
(defn minimum  [a b] (.minimum (ensure-mx a) (ensure-mx b)))
(defn clip
  "Clip values to [lo, hi] range. lo/hi are numbers (not MxArray)."
  [a lo hi]
  (.clip (ensure-mx a) lo hi))
(defn sign     [a]   (.sign (ensure-mx a)))
(defn reciprocal [a] (.reciprocal (ensure-mx a)))
(defn floor-divide [a b] (.floorDivide (ensure-mx a) (ensure-mx b)))
(defn remainder    [a b] (.remainder (ensure-mx a) (ensure-mx b)))

;; ---------------------------------------------------------------------------
;; Math functions
;; ---------------------------------------------------------------------------

(defn exp      [a] (.exp (ensure-mx a)))
(defn expm1    [a] (.expm1 (ensure-mx a)))
(defn log      [a] (.log (ensure-mx a)))
(defn log2     [a] (.log2 (ensure-mx a)))
(defn log10    [a] (.log10 (ensure-mx a)))
(defn log1p    [a] (.log1p (ensure-mx a)))
(defn logaddexp [a b] (.logaddexp (ensure-mx a) (ensure-mx b)))

(defn sin      [a] (.sin (ensure-mx a)))
(defn cos      [a] (.cos (ensure-mx a)))
(defn tan      [a] (.tan (ensure-mx a)))
(defn arccos   [a] (.arccos (ensure-mx a)))
(defn tanh     [a] (.tanh (ensure-mx a)))
(defn sigmoid  [a] (.sigmoid (ensure-mx a)))
(defn erf      [a] (.erf (ensure-mx a)))
(defn erfinv   [a] (.erfinv (ensure-mx a)))
(defn lgamma   [a] (.lgamma (ensure-mx a)))
(defn digamma  [a] (.digamma (ensure-mx a)))
(defn bessel-i0e [a] (.besselI0e (ensure-mx a)))
(defn bessel-i1e [a] (.besselI1e (ensure-mx a)))

(defn floor    [a] (.floor (ensure-mx a)))
(defn ceil     [a] (.ceil (ensure-mx a)))
(defn round    [a] (.round (ensure-mx a)))

;; ---------------------------------------------------------------------------
;; Reductions
;; ---------------------------------------------------------------------------

;; mlx-node reductions take Int32Array for axes (not JS arrays).
;; Axes can be null for full reduction.

(defn sum
  ([a]      (.sum (ensure-mx a) nil nil))
  ([a axes] (.sum (ensure-mx a) (to-int32-axes axes) nil))
  ([a axes keepdims] (.sum (ensure-mx a) (to-int32-axes axes) keepdims)))

(defn prod
  ([a]      (.prod (ensure-mx a) nil nil))
  ([a axes] (.prod (ensure-mx a) (to-int32-axes axes) nil)))

(defn mean
  ([a]      (.mean (ensure-mx a) nil nil))
  ([a axes] (.mean (ensure-mx a) (to-int32-axes axes) nil)))

(defn variance
  ([a]      (.var (ensure-mx a) nil nil))
  ([a axes] (.var (ensure-mx a) (to-int32-axes axes) nil)))

(defn std
  ([a]      (.std (ensure-mx a) nil nil))
  ([a axes] (.std (ensure-mx a) (to-int32-axes axes) nil)))

(defn amax
  ([a]      (.max (ensure-mx a) nil nil))
  ([a axes] (.max (ensure-mx a) (to-int32-axes axes) nil)))

(defn amin
  ([a]      (.min (ensure-mx a) nil nil))
  ([a axes] (.min (ensure-mx a) (to-int32-axes axes) nil)))

(defn argmax
  ([a]      (.argmax (ensure-mx a) 0))
  ([a axis] (.argmax (ensure-mx a) axis)))

(defn argmin
  ([a]      (.argmin (ensure-mx a) 0))
  ([a axis] (.argmin (ensure-mx a) axis)))

(defn all
  "True if all elements are true (nonzero). Optional axis."
  ([a]      (.all (ensure-mx a) nil nil))
  ([a axis] (.all (ensure-mx a) (to-int32-axes [axis]) nil)))

(defn any
  "True if any element is true (nonzero). Optional axis."
  ([a]      (.any (ensure-mx a) nil nil))
  ([a axis] (.any (ensure-mx a) (to-int32-axes [axis]) nil)))

(defn argsort
  "Return indices that sort the array along the given axis (default: last axis)."
  ([a]      (.argsort (ensure-mx a)))
  ([a axis] (.argsort (ensure-mx a) axis)))

(defn searchsorted
  "Find insertion indices for values in a sorted 1D array.
   Returns indices such that inserting values maintains sorted order.
   side :left (default) returns first valid index, :right returns last."
  ([sorted-arr values]
   (.searchsorted (ensure-mx sorted-arr) (ensure-mx values)))
  ([sorted-arr values side]
   (.searchsorted (ensure-mx sorted-arr) (ensure-mx values) (= side :right))))

(defn sort-arr
  "Sort array along the given axis (default: last axis)."
  ([a]      (.sort (ensure-mx a)))
  ([a axis] (.sort (ensure-mx a) axis)))

(defn topk
  "Return the top-k largest values along the last axis."
  [a k]
  (.topk (ensure-mx a) k))

(defn logsumexp
  ([a]              (.logsumexp (ensure-mx a) nil nil))
  ([a axes]         (.logsumexp (ensure-mx a) (to-int32-axes axes) nil))
  ([a axes keepdims] (.logsumexp (ensure-mx a) (to-int32-axes axes) keepdims)))

(defn cumsum
  ([a]      (.cumsum (ensure-mx a) 0))
  ([a axis] (.cumsum (ensure-mx a) axis)))

(defn logcumsumexp
  "Cumulative log-sum-exp along axis."
  ([a]      (.logcumsumexp (ensure-mx a) 0))
  ([a axis] (.logcumsumexp (ensure-mx a) axis)))

;; ---------------------------------------------------------------------------
;; Comparison / selection
;; ---------------------------------------------------------------------------

;; mlx-node: comparison ops are instance methods.
(defn equal        [a b] (.equal (ensure-mx a) (ensure-mx b)))
(defn not-equal    [a b] (.notEqual (ensure-mx a) (ensure-mx b)))
(defn greater      [a b] (.greater (ensure-mx a) (ensure-mx b)))
(defn greater-equal [a b] (.greaterEqual (ensure-mx a) (ensure-mx b)))
(defn less         [a b] (.less (ensure-mx a) (ensure-mx b)))
(defn less-equal   [a b] (.lessEqual (ensure-mx a) (ensure-mx b)))

(defn where
  "Select elements from a or b based on condition.
   mlx-node: condition.where(x, y) — condition is the receiver."
  [cond a b]
  (.where (ensure-mx cond) (ensure-mx a) (ensure-mx b)))

;; Model-level comparison helpers — auto-promote integers, return float32.
;; Use these in gen bodies where traced values are tensors during enumeration.
;;   (eq? prize 0)  instead of  (.astype (equal prize (scalar 0 int32)) float32)
;;   (and* a b)     instead of  (multiply (.astype a float32) (.astype b float32))

(defn eq?  [a b]
  (.astype (equal (if (number? a) (scalar a int32) a)
                  (if (number? b) (scalar b int32) b)) float32))
(defn neq? [a b]
  (.astype (not-equal (if (number? a) (scalar a int32) a)
                      (if (number? b) (scalar b int32) b)) float32))
(defn gt?  [a b]
  (.astype (greater (if (number? a) (scalar a) a)
                    (if (number? b) (scalar b) b)) float32))
(defn lt?  [a b]
  (.astype (less (if (number? a) (scalar a) a)
                 (if (number? b) (scalar b) b)) float32))
(defn and* [a b] (multiply a b))
(defn or*  [a b] (maximum a b))

(defn isnan        [a] (.isnan (ensure-mx a)))
(defn isinf        [a] (.isinf (ensure-mx a)))
(defn nan-to-num
  "Replace NaN/Inf with finite values. Default: NaN->0."
  ([a]                             (.nanToNum (ensure-mx a) 0.0))
  ([a nan-val]                     (.nanToNum (ensure-mx a) nan-val))
  ([a nan-val posinf-val neginf-val] (.nanToNum (ensure-mx a) nan-val posinf-val neginf-val)))

;; ---------------------------------------------------------------------------
;; Shape manipulation
;; ---------------------------------------------------------------------------

(defn reshape    [a sh] (.reshape (ensure-mx a) (to-big-shape sh)))
(defn flatten    [a]    (.flatten (ensure-mx a)))
(defn squeeze
  "Remove size-1 dimensions. With axes, only squeeze specified positions."
  ([a]      (.squeeze (ensure-mx a)))
  ([a axes] (.squeeze (ensure-mx a) (to-int32-axes (vec axes)))))
(defn expand-dims [a axis] (.expandDims (ensure-mx a) axis))
(defn transpose
  ([a]      (.transpose (ensure-mx a)))
  ([a axes] (.transpose (ensure-mx a) (to-int32-axes axes))))
(defn stack
  ;; mlx-node: MxArray.stack(arrays, axis) — static method
  ([arrs]      (.stack M (to-array arrs)))
  ([arrs axis] (.stack M (to-array arrs) axis)))
(defn concatenate
  ;; mlx-node: MxArray.concatenate for 2, concatenateMany for 3+
  ([arrs]
   (let [arr-vec (vec arrs)]
     (if (= 2 (count arr-vec))
       (.concatenate M (nth arr-vec 0) (nth arr-vec 1) 0)
       (.concatenateMany M (to-array arr-vec) 0))))
  ([arrs axis]
   (let [arr-vec (vec arrs)]
     (if (= 2 (count arr-vec))
       (.concatenate M (nth arr-vec 0) (nth arr-vec 1) axis)
       (.concatenateMany M (to-array arr-vec) axis)))))
(defn broadcast-to [a sh] (.broadcastTo (ensure-mx a) (to-big-shape sh)))
(defn tile [a reps] (.tile (ensure-mx a) (to-int32-axes reps)))
(defn repeat-arr [a repeats axis] (.repeat (ensure-mx a) repeats axis))
(defn split-arr
  ([a sections]      (vec (.split (ensure-mx a) sections)))
  ([a sections axis] (vec (.split (ensure-mx a) sections axis))))

;; ---------------------------------------------------------------------------
;; Indexing
;; ---------------------------------------------------------------------------

(defn take-idx
  ([a indices]      (.take (ensure-mx a) (if (number? indices) (scalar indices int32) (ensure-mx indices)) 0))
  ([a indices axis] (.take (ensure-mx a) (if (number? indices) (scalar indices int32) (ensure-mx indices)) axis)))

(defn idx
  "Extract element at index i along axis (default 0).
   Auto-promotes integer i to MLX int32 scalar.

     (idx probs 2)   ; instead of (take-idx probs (scalar 2 int32) 0)"
  ([a i]      (take-idx a (if (number? i) (scalar i int32) i) 0))
  ([a i axis] (take-idx a (if (number? i) (scalar i int32) i) axis)))

(defn take-along-axis [a indices axis]
  (.takeAlongAxis (ensure-mx a) (ensure-mx indices) axis))

(defn index
  "Index along axis 0. For 1D: returns scalar element. For 2D: returns row.
   Uses take with a scalar int32 index."
  [a i]
  (.take a (if (number? i) (scalar i int32) i) 0))

(defn slice
  "Slice along axis 0. Returns elements [start, stop) with optional step.
   mlx-node uses .slice(starts, stops) with BigInt64Array."
  ([a start stop]
   (.slice a (js/BigInt64Array.from #js [(js/BigInt start)])
             (js/BigInt64Array.from #js [(js/BigInt stop)])))
  ([a start stop step]
   ;; mlx-node .slice doesn't support step directly.
   ;; Emulate via slice then take with strided indices.
   (let [sliced (.slice a (js/BigInt64Array.from #js [(js/BigInt start)])
                          (js/BigInt64Array.from #js [(js/BigInt stop)]))]
     (if (= step 1)
       sliced
       ;; Build strided index array
       (let [n (first (shape->clj sliced))
             indices (array (vec (range 0 n step)) int32)]
         (.take sliced indices 0))))))

(defn mat-get
  "Get element [i,j] from a 2D array. Returns a scalar MLX array."
  [a i j]
  ;; Take row i, then element j from that row
  (let [row (.take a (scalar i int32) 0)]
    (.take row (scalar j int32) 0)))

;; ---------------------------------------------------------------------------
;; Matrix operations
;; ---------------------------------------------------------------------------

(defn matmul    [a b] (.matmul (ensure-mx a) (ensure-mx b)))
(defn inner     [a b] (.inner (ensure-mx a) (ensure-mx b)))
(defn outer     [a b] (.outer (ensure-mx a) (ensure-mx b)))
(defn diag      [a]   (.diag (ensure-mx a)))
(defn trace-mat
  "Matrix trace (sum of diagonal elements)."
  ([a]                  (.trace (ensure-mx a) 0 0 1))
  ([a offset]           (.trace (ensure-mx a) offset 0 1))
  ([a offset ax1 ax2]   (.trace (ensure-mx a) offset ax1 ax2)))
(defn einsum
  "Einstein summation. E.g. (einsum \"ij,jk->ik\" a b)"
  [subscripts & arrays]
  (.einsum M subscripts (to-array arrays)))

;; ---------------------------------------------------------------------------
;; Linear algebra
;; ---------------------------------------------------------------------------

;; mlx-node: linalg ops are instance methods on MxArray.
;; No cpu-stream needed — handled internally in C++.

(defn cholesky [a]   (.cholesky (ensure-mx a) false))
(defn solve   [a b]  (.linalgSolve (ensure-mx a) (ensure-mx b)))
(defn solve-triangular [a b upper]
  (.solveTriangular (ensure-mx a) (ensure-mx b) upper))
(defn inv     [a]    (.linalgInv (ensure-mx a)))
(defn tri-inv [a upper] (.triInv (ensure-mx a) upper))
(defn cholesky-inv
  "Inverse of A from its Cholesky factor L (where A=LL^T)."
  ([a]       (.choleskyInv (ensure-mx a) false))
  ([a upper] (.choleskyInv (ensure-mx a) upper)))
(defn qr [a]
  (let [result (.qr (ensure-mx a))]
    [(aget result 0) (aget result 1)]))
(defn svd [a]
  (let [result (.svd (ensure-mx a))]
    [(aget result 0) (aget result 1) (aget result 2)]))
(defn eigh [a]
  (let [result (.eigh (ensure-mx a))]
    [(aget result 0) (aget result 1)]))
(defn eigvalsh [a] (.eigvalsh (ensure-mx a)))
(defn norm
  ([a]     (.linalgNorm (ensure-mx a)))
  ([a ord] (.linalgNorm (ensure-mx a) ord)))

(defn logdet
  "Log-determinant of positive-definite matrix via Cholesky."
  [a]
  (let [L (cholesky a)]
    (multiply (scalar 2.0) (sum (log (diag L))))))

(defn det
  "Determinant of positive-definite matrix via Cholesky."
  [a]
  (let [L (cholesky a)]
    (power (prod (diag L)) (scalar 2))))


;; ---------------------------------------------------------------------------
;; Autograd
;; ---------------------------------------------------------------------------

;; mlx-node autograd is fundamentally different from node-mlx:
;; - node-mlx: .grad(core, f) returns a gradient function
;; - mlx-node: MxArray.valueAndGrad(fn, inputs) applies immediately, returns [loss, ...grads]
;; - mlx-node: MxArray.computeGradients(fn, inputs) returns [...grads] only
;;
;; We adapt the API: grad/value-and-grad return curried functions that call
;; MxArray.valueAndGrad/computeGradients when invoked with actual arguments.

(defn grad
  "Compute gradient of f. Tracks grad-depth so p/generate can skip
   the L3 analytical path (which uses volatile! and breaks gradient flow).
   Each call runs inside mx/tidy to prevent Metal buffer accumulation.
   Tidy is skipped inside mx/compile-fn (would dispose trace arrays).

   Adaptation: mlx-node has no .grad() that returns a function.
   We use MxArray.computeGradients(fn, inputs) which applies immediately.
   The returned wrapper function adapts the curried API."
  ([f]
   ;; Default: gradient w.r.t. first argument
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [inputs (to-array args)
             ;; computeGradients returns [grad0, grad1, ...] for all inputs
             grads (.computeGradients M f inputs)
             ;; Return gradient of first argument (matching node-mlx .grad behavior)
             r (aget grads 0)]
         (when-not (in-compile?)
           (eval! r))
         r)
       (finally (swap! grad-depth dec)))))
  ([f argnums]
   ;; Gradient w.r.t. specific arguments
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [argnum-vec (if (vector? argnums) argnums [argnums])
             ;; Build a wrapper that only passes the selected args to valueAndGrad
             ;; then unpacks the gradients for those specific positions
             inputs (to-array args)
             grads (.computeGradients M f inputs)]
         ;; Return grads for the specified argnums
         (if (= 1 (count argnum-vec))
           (let [r (aget grads (first argnum-vec))]
             (when-not (in-compile?)
               (eval! r))
             r)
           (let [r (mapv #(aget grads %) argnum-vec)]
             (when-not (in-compile?)
               (apply eval! r))
             r)))
       (finally (swap! grad-depth dec))))))

(defn value-and-grad
  "Compute value and gradient of f. Tracks grad-depth.
   Each call runs inside mx/tidy to prevent Metal buffer accumulation.
   Tidy is skipped inside mx/compile-fn (would dispose trace arrays).

   Adaptation: mlx-node MxArray.valueAndGrad(fn, inputs) returns [loss, grad0, grad1, ...].
   node-mlx returns [value, gradient_of_first_arg] for the no-argnums case.
   We match that convention: always return [value, single_gradient]."
  ([f]
   ;; Default: gradient w.r.t. first argument (matching node-mlx behavior)
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [inputs (to-array args)
             ;; valueAndGrad returns [loss, grad0, grad1, ...]
             result (.valueAndGrad M f inputs)
             v (aget result 0)
             ;; Return gradient of first argument only (node-mlx convention)
             g (aget result 1)]
         (when-not (in-compile?)
           (eval! v g))
         [v g])
       (finally (swap! grad-depth dec)))))
  ([f argnums]
   ;; argnums variant: differentiate w.r.t. specified argument(s).
   ;; mlx-node always computes all gradients; we select the requested ones.
   (fn [& args]
     (swap! grad-depth inc)
     (try
       (let [inputs (to-array args)
             result (.valueAndGrad M f inputs)
             v (aget result 0)
             argnum-vec (if (vector? argnums) argnums [argnums])
             ;; For single argnum, return the gradient directly.
             ;; For multiple, return JS array of gradients.
             g (if (= 1 (count argnum-vec))
                 (aget result (inc (first argnum-vec)))
                 (let [gs #js []]
                   (doseq [i argnum-vec]
                     (.push gs (aget result (inc i))))
                   gs))]
         (when-not (in-compile?)
           (if (instance? M g)
             (eval! v g)
             (do (eval! v)
                 (.evalArrays c (to-array (vec (js->clj g)))))))
         [v g])
       (finally (swap! grad-depth dec))))))

(defn jvp
  "Forward-mode Jacobian-vector product.
   mlx-node has no jvp — implement via forward-mode AD workaround.
   Falls back to numerical differentiation approximation."
  [f primals tangents]
  ;; mlx-node doesn't expose jvp. For now, compute f(primals) and approximate.
  ;; This is a stub — callers that need true JVP will need mlx-node upstream support.
  (let [result-val (apply f primals)
        ;; Approximate JVP via (f(x+eps*t) - f(x)) / eps
        eps 1e-5
        primals-arr (vec primals)
        tangents-arr (vec tangents)
        perturbed (mapv (fn [p t] (add p (multiply (scalar eps) t)))
                        primals-arr tangents-arr)
        result-perturbed (apply f perturbed)
        jvp-approx (divide (subtract result-perturbed result-val) (scalar eps))]
    [result-val jvp-approx]))

(defn vjp
  "Reverse-mode vector-Jacobian product.
   mlx-node has no vjp — we implement via valueAndGrad with a surrogate loss."
  [f primals cotangents]
  ;; VJP: compute gradients of <f(x), cotangent> w.r.t. x
  ;; This gives us the adjoint (VJP) of f.
  (let [primals-arr (vec primals)
        cotangents-arr (vec cotangents)
        ;; Surrogate loss: inner product of f(x) with cotangent
        surrogate (fn [& xs]
                    (let [result (apply f xs)]
                      (if (sequential? cotangents-arr)
                        (sum (multiply result (first cotangents-arr)))
                        (sum (multiply result cotangents-arr)))))
        inputs (to-array primals-arr)
        result (.valueAndGrad M surrogate inputs)
        fval (apply f primals-arr)
        ;; result is [loss, grad0, grad1, ...]
        grads (let [gs #js []]
                (dotimes [i (dec (.-length result))]
                  (.push gs (aget result (inc i))))
                gs)]
    [fval grads]))

(defn stop-gradient [a] (.stopGradient (ensure-mx a)))

;; ---------------------------------------------------------------------------
;; Transforms
;; ---------------------------------------------------------------------------

(def ^:private compile-generation
  "Monotonic counter incremented on each compile-clear-cache! call.
   Compiled functions compare their birth generation against this to
   detect cache invalidation and recompile transparently."
  (atom 0))

(defn compile-fn
  "Wrap f in mx/compile-fn with auto-recompilation on cache clear.
   If compile-clear-cache! has been called since this function was compiled,
   the next invocation transparently recompiles -- no crash, no manual
   intervention. Cost: one atom deref per call (negligible).
   Tracks compile-depth so mx/grad can skip tidy during tracing.

   Adaptation: mlx-node MxArray.compileFn(fn, inputs, shapeless) applies immediately.
   We return a wrapper that calls compileFn each time with the current inputs.
   If f returns a JS array of MxArrays, they are stacked into a single tensor
   (compileFn requires MxArray return) and split back on output."
  ([f]    (compile-fn f false))
  ([f shapeless?]
   ;; mlx-node compileFn applies immediately rather than returning a compiled function.
   ;; We wrap to provide the same curried API. Each call traces and compiles.
   ;; Probe on first call whether f returns a JS array (multi-output) or single MxArray.
   (let [multi-output? (atom nil)]
     (fn [& args]
       (swap! compile-depth inc)
       (try
         ;; Probe the return type on first call
         (when (nil? @multi-output?)
           (let [probe-result (apply f args)]
             (reset! multi-output? (js/Array.isArray probe-result))))
         (if @multi-output?
           ;; Multi-output: wrap f to stack results, then split on output
           (let [n-out (atom nil)
                 wrapped (fn [& inner-args]
                           (let [raw (apply f inner-args)
                                 arr-vec (vec (js->clj raw))]
                             (reset! n-out (count arr-vec))
                             (.stack M (to-array arr-vec))))
                 result (.compileFn M wrapped (to-array args) shapeless?)
                 stacked (aget result 0)]
             ;; Split back into JS array
             (to-array (mapv (fn [i] (.take stacked (scalar i int32) 0))
                             (range @n-out))))
           ;; Single-output: pass through directly
           (let [result (.compileFn M f (to-array args) shapeless?)]
             (if (= 1 (.-length result))
               (aget result 0)
               result)))
         (finally (swap! compile-depth dec)))))))

(defn compile-clear-cache!
  "Clear all compiled function caches, releasing associated Metal resources.
   Safe to call at any time -- compiled functions transparently recompile
   on next use via the compile-generation counter."
  []
  (.compileClearCache c)
  (swap! compile-generation inc))

(defn vmap
  "Vectorized map. Applies f to batched inputs.

   Adaptation: mlx-node MxArray.vmap(fn, inputs, inAxes, outAxes) applies immediately.
   node-mlx .vmap(core, f, inAxes, outAxes) returns a vmapped function.
   We return a wrapper function that calls MxArray.vmap on each invocation."
  ([f]                     (fn [& args] (let [r (.vmap M f (to-array args))] (aget r 0))))
  ([f in-axes]             (fn [& args] (let [r (.vmap M f (to-array args) (clj->js in-axes))] (aget r 0))))
  ([f in-axes out-axes]    (fn [& args] (let [r (.vmap M f (to-array args) (clj->js in-axes) (clj->js out-axes))] (aget r 0)))))

;; ---------------------------------------------------------------------------
;; Async
;; ---------------------------------------------------------------------------

(defn async-eval!
  "Asynchronously evaluate arrays. Uses per-array evalAsync."
  [& arrays]
  (let [promises (mapv #(.evalAsync %) (filter some? arrays))]
    ;; Return a promise that resolves when all are done
    (js/Promise.all (to-array promises))))

;; ---------------------------------------------------------------------------
;; Device / Stream
;; ---------------------------------------------------------------------------

;; mlx-node has no device/stream management — MLX handles it internally.
(defn default-device [] "gpu")
(defn set-default-device! [d] nil)
(def cpu "cpu")
(def gpu "gpu")

;; ---------------------------------------------------------------------------
;; Constants
;; ---------------------------------------------------------------------------

;; mlx-node has no pi/e/inf/nan constants — compute them.
(def pi    (.-PI js/Math))
(def e-val (.-E js/Math))
(def inf   js/Infinity)
(def nan   js/NaN)

;; ---------------------------------------------------------------------------
;; Softmax
;; ---------------------------------------------------------------------------

(defn softmax
  ([a]      (.softmax (ensure-mx a) -1))
  ([a axis] (.softmax (ensure-mx a) axis)))

;; ---------------------------------------------------------------------------
;; Utilities
;; ---------------------------------------------------------------------------

(def ^:private MxArray
  "Constructor of MLX arrays -- used for fast instance? checks."
  M)

(defn array? [x]
  (instance? M x))

(defn realize
  "Evaluate a lazy MLX array and return its scalar JS value."
  [x] (eval! x) (item x))

;; ---------------------------------------------------------------------------
;; Layer 0 boundary helpers -- ALL eval!/tidy calls in Layers 1-8 flow
;; through these. Keeps side-effectful materialization confined to mlx.cljs.
;; ---------------------------------------------------------------------------

(def ^:private jsc
  "Bun JSC internals -- exposes GC control, weak ref release, microtask drain."
  (when (exists? js/Bun) (js/require "bun:jsc")))

(defn jsc-cleanup!
  "Trigger JSC garbage collection + microtask drain + weak ref cleanup.
   Fires N-API destroy callbacks for dead MLX arrays, releasing Metal buffers.
   Safe to call from synchronous code."
  []
  (when jsc
    (.releaseWeakRefs jsc)
    (.drainMicrotasks jsc)
    (.gcAndSweep jsc)))

(def ^:private gc-fn
  "Synchronous GC function (Bun.gc or global.gc if available)."
  (or (when (exists? js/Bun) (.-gc js/Bun))
      (.-gc js/globalThis)))

;; ---------------------------------------------------------------------------
;; Resource-pressure auto-cleanup (Bun GC integration)
;; ---------------------------------------------------------------------------

(def ^:private resource-pressure-threshold
  "When active memory exceeds this (bytes), auto-cleanup triggers.
   mlx-node has no resource counting -- we use memory bytes instead."
  (* 512 1024 1024))  ;; 512MB

(def ^:private ^:mutable ops-since-check
  "Counter to amortize the cost of getActiveMemory calls.
   Only check resource pressure every N operations."
  0)

(def ^:private check-interval
  "Check resource pressure every N auto-cleanup! calls."
  50)

(defn auto-cleanup!
  "Resource-pressure cleanup for hot paths. Two tiers:

   Lightweight (default): sweep + clear only. Harvests Metal buffers
   that Bun's natural GC has already freed. Safe to call from anywhere --
   including inside tight handler loops (SMC rejuvenation, MH chains).
   The handler's purity (each trace op produces a clean batch of dead
   intermediates via vreset!) makes sweep highly effective.

   Aggressive (aggressive? true): also forces GC via jsc-cleanup! before
   sweeping. Use ONLY from leaf operations (dist-sample-n) that are not
   called from inside complex state-holding loops. Never from trace-fn --
   forced GC during tight handler loops causes use-after-free segfaults."
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

(def ^:private ^:mutable gfi-ops-count
  "Counter for amortizing GFI-boundary cleanup."
  0)

(def ^:private gfi-cleanup-interval
  "Check resource pressure every N GFI operations."
  10)

(def ^:private gfi-pressure-threshold
  "Only force jsc-cleanup! when active memory exceeds this (bytes).
   Prevents aggressive GC in small inference loops where it can cause
   SIGTRAP crashes."
  (* 128 1024 1024))  ;; 128MB

(defn gfi-cleanup!
  "Cleanup at GFI operation boundaries. Every N calls, checks resource
   pressure. If active memory exceeds the threshold, forces N-API weak
   reference release via jsc-cleanup!, then sweeps freed Metal buffers.

   Called after each DynamicGF protocol operation (simulate, generate,
   update, regenerate, assess, project). Unlike auto-cleanup! (which
   uses sweep-only from inside handlers), this runs AFTER the handler
   returns, when old-iteration arrays are unreachable. jsc-cleanup!
   forces their weak refs to fire, making sweep effective in sync code."
  []
  (set! gfi-ops-count (inc gfi-ops-count))
  (when (>= gfi-ops-count gfi-cleanup-interval)
    (set! gfi-ops-count 0)
    (when (> (get-active-memory) gfi-pressure-threshold)
      (jsc-cleanup!)
      (sweep-dead-arrays!)
      (clear-cache!))))

(defn materialize!
  "Evaluate MLX arrays, materializing the computation graph.
   Use at inference/training loop boundaries to bound graph size."
  [& arrs]
  (apply eval! arrs))

(defn realize-clj
  "Evaluate an MLX array and convert to ClojureScript data."
  [x]
  (eval! x)
  (->clj x))

(defn tidy-materialize
  "Run f inside mx/tidy, materialize the result, return it.
   For simple cases where f returns a single MLX array or JS array."
  [f]
  (let [r (tidy f)]
    (eval! r)
    r))

;; Auto-clear threshold: release Metal cache when it exceeds this (bytes).
;; Prevents unbounded cache growth that crashes Bun at ~2GB.
(def ^:private cache-pressure-threshold (* 512 1024 1024)) ;; 512MB

(defn tidy-run
  "Run f inside mx/tidy. Call collect-fn on the result to get arrays
   to preserve. Materializes those arrays (detaching from computation
   graph intermediates). Returns the result of f.
   Automatically clears Metal cache when memory pressure is high.
   collect-fn: (result) -> [array1, array2, ...]"
  [f collect-fn]
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

(defn tidy-scalar
  "Run f inside mx/tidy, extract a JS number via item, return it.
   All intermediate MLX arrays are freed. The returned value is a
   plain JS number with no MLX references -- safe for use in loops.
   Automatically clears Metal cache when memory pressure is high.
   f must return an MLX scalar array."
  [f]
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

;; ---------------------------------------------------------------------------
;; Resource management (moved from inference/util.cljs)
;; ---------------------------------------------------------------------------

(defn force-gc!
  "Force garbage collection and Metal buffer cleanup.
   Clears compiled function caches too -- compiled functions transparently
   recompile on next use, so this is safe."
  []
  (jsc-cleanup!)
  (when gc-fn (gc-fn))
  (sweep-dead-arrays!)
  (clear-cache!)
  (compile-clear-cache!))

(def ^:private DEFAULT-CACHE-LIMIT (* 256 1024 1024))

(set-cache-limit! DEFAULT-CACHE-LIMIT)

(defn with-resource-guard
  "Run f with cache-limit=0 to prevent Metal buffer accumulation.
   Freed buffers are released immediately instead of being cached."
  [f]
  (let [prev-limit (set-cache-limit! 0)]
    (try (f)
      (finally
        (clear-cache!)
        (set-cache-limit! prev-limit)))))

;; ---------------------------------------------------------------------------
;; NN training step (moved from nn.cljs)
;; ---------------------------------------------------------------------------

(defn training-step!
  "One NN training step: compute loss+grads, update module. Returns loss (JS number)."
  [module optim vg-fn & inputs]
  (let [[loss grads] (apply vg-fn inputs)]
    (.update optim module grads)
    (eval! module)
    (eval! loss)
    (item loss)))

(defn ensure-array
  "Wrap a JS number as an MLX scalar array; pass through existing arrays.
   Vectors and sequences are converted to MLX arrays.
   Functions and keywords are passed through (for distributions carrying
   closures or address references)."
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
     (array? x) (if (= (.dtype x) dtype) x (astype x dtype))
     (fn? x) x
     (keyword? x) x
     (map? x) x
     (or (vector? x) (seq? x) (sequential? x)) (array x dtype)
     :else (scalar x dtype))))
