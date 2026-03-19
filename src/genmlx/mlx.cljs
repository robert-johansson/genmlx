(ns genmlx.mlx
  "Thin ClojureScript wrapper over node-mlx (@frost-beta/mlx).
   Provides idiomatic ClojureScript access to MLX tensor operations,
   autograd, random number generation, and linear algebra.

   All operations are lazy by default — call (eval!) to materialize.
   Requires: npm install @frost-beta/mlx (Apple Silicon only, nbb only).")

;; ---------------------------------------------------------------------------
;; Module loading
;; ---------------------------------------------------------------------------

(defonce ^:private mlx-mod (js/require "@frost-beta/mlx"))
(defonce core    (.-core mlx-mod))
(defonce random  (.-random core))
(defonce linalg  (.-linalg core))
(defonce nn-mod  (.-nn mlx-mod))
(defonce optim-mod (.-optimizers mlx-mod))

;; CPU stream needed for linalg ops (cholesky, solve, etc.)
(defonce ^:private cpu-stream (.newStream core (.-cpu core)))

;; ---------------------------------------------------------------------------
;; Dtypes
;; ---------------------------------------------------------------------------

(def float32  (.-float32 core))
(def float64  (.-float64 core))
(def int32    (.-int32 core))
(def int64    (.-int64 core))
(def bool-dt  (.-bool_ core))

;; ---------------------------------------------------------------------------
;; Array creation
;; ---------------------------------------------------------------------------

(defn scalar
  ([v]      (.array core v))
  ([v dtype] (.array core v dtype)))

(defn array
  ([v]      (.array core (clj->js v)))
  ([v dtype] (.array core (clj->js v) dtype)))

(defn astype
  "Cast array to the given dtype."
  [a dtype]
  (.astype a dtype))

(defn zeros
  ([sh]       (.zeros core (clj->js sh)))
  ([sh dtype] (.zeros core (clj->js sh) dtype)))

(defn ones
  ([sh]       (.ones core (clj->js sh)))
  ([sh dtype] (.ones core (clj->js sh) dtype)))

(defn full [sh val] (.full core (clj->js sh) val))

(defn eye
  ([n]       (.eye core n))
  ([n dtype] (.eye core n dtype)))

(defn arange
  ([stop]            (.arange core stop))
  ([start stop]      (.arange core start stop))
  ([start stop step] (.arange core start stop step)))

(defn linspace [start stop num] (.linspace core start stop num))

;; ---------------------------------------------------------------------------
;; Evaluation / materialization
;; ---------------------------------------------------------------------------

(defn eval! [& arrs]
  (apply (.-eval core) arrs))

(defn item [a]
  (.item a))

(defn ->clj [a]
  (.eval core a)
  (js->clj (.tolist a)))

(defn shape [a] (vec (.-shape a)))

(defn ndim [a] (count (.-shape a)))

(defn dtype [a] (.-dtype a))

(defn size [a] (.-size a))

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

(defn in-grad?
  "Returns true if currently executing inside an mx/grad or mx/value-and-grad scope."
  [] (pos? @grad-depth))

(defn tidy [f]
  (swap! tidy-depth inc)
  (try
    (.tidy core f)
    (finally
      (swap! tidy-depth dec))))

(defn in-tidy?
  "Returns true if currently executing inside an mx/tidy scope."
  [] (pos? @tidy-depth))

(defn dispose! [a] (.dispose core a))

;; Memory monitoring
(defn get-active-memory [] (.getActiveMemory core))
(defn get-cache-memory [] (.getCacheMemory core))
(defn get-peak-memory [] (.getPeakMemory core))
(defn reset-peak-memory! [] (.resetPeakMemory core))
(defn get-wrappers-count [] (.getWrappersCount core))
(defn sweep-dead-arrays!
  "Synchronously free Metal buffers for arrays whose JS wrappers have been GC'd
   but whose deferred N-API finalizers haven't run yet. Returns count swept.
   Call this in synchronous loops where event loop yields are rare.
   No-op when called inside mx/tidy (tidy manages its own disposal)."
  []
  (when-not (in-tidy?)
    (.sweepDeadArrays core)))

;; Memory control
(defn set-memory-limit! [n] (.setMemoryLimit core n))
(defn set-cache-limit! [n] (.setCacheLimit core n))
(defn set-wired-limit! [n] (.setWiredLimit core n))
(defn clear-cache! [] (.clearCache core))

;; Metal device info
(defn metal-is-available? [] (.isAvailable (.-metal core)))

(defn metal-device-info []
  (let [info (.deviceInfo (.-metal core))]
    {:architecture (.-architecture info)
     :device-name (.-device_name info)
     :memory-size (.-memory_size info)
     :max-buffer-length (.-max_buffer_length info)
     :max-recommended-working-set-size (.-max_recommended_working_set_size info)
     :resource-limit (.-resource_limit info)}))

;; Convenience
(defn memory-report []
  {:active-bytes (get-active-memory)
   :cache-bytes (get-cache-memory)
   :peak-bytes (get-peak-memory)
   :wrappers (get-wrappers-count)
   :resource-limit (:resource-limit (metal-device-info))})

;; ---------------------------------------------------------------------------
;; Arithmetic (element-wise)
;; ---------------------------------------------------------------------------

(defn add
  ([a b] (.add core a b))
  ([a b & more] (reduce add (add a b) more)))
(defn subtract
  ([a b] (.subtract core a b))
  ([a b & more] (reduce subtract (subtract a b) more)))
(defn multiply
  ([a b] (.multiply core a b))
  ([a b & more] (reduce multiply (multiply a b) more)))
(defn divide   [a b] (.divide core a b))
(defn negative [a]   (.negative core a))
(defn power    [a b] (.power core a b))
(defn square   [a]   (.square core a))
(defn sqrt     [a]   (.sqrt core a))
(defn abs      [a]   (.abs core a))
(defn maximum  [a b] (.maximum core a b))
(defn minimum  [a b] (.minimum core a b))
(defn clip     [a lo hi] (.clip core a lo hi))
(defn sign     [a]   (.sign core a))
(defn reciprocal [a] (.reciprocal core a))
(defn floor-divide [a b] (.floorDivide core a b))

;; ---------------------------------------------------------------------------
;; Math functions
;; ---------------------------------------------------------------------------

(defn exp      [a] (.exp core a))
(defn expm1    [a] (.expm1 core a))
(defn log      [a] (.log core a))
(defn log2     [a] (.log2 core a))
(defn log10    [a] (.log10 core a))
(defn log1p    [a] (.log1p core a))
(defn logaddexp [a b] (.logaddexp core a b))

(defn sin      [a] (.sin core a))
(defn cos      [a] (.cos core a))
(defn tan      [a] (.tan core a))
(defn arccos   [a] (.arccos core a))
(defn tanh     [a] (.tanh core a))
(defn sigmoid  [a] (.sigmoid core a))
(defn erf      [a] (.erf core a))
(defn erfinv   [a] (.erfinv core a))
(defn lgamma   [a] (.lgamma core a))
(defn digamma  [a] (.digamma core a))
(defn bessel-i0e [a] (.besselI0e core a))
(defn bessel-i1e [a] (.besselI1e core a))

(defn floor    [a] (.floor core a))
(defn ceil     [a] (.ceil core a))
(defn round    [a] (.round core a))

;; ---------------------------------------------------------------------------
;; Reductions
;; ---------------------------------------------------------------------------

(defn sum
  ([a]      (.sum core a))
  ([a axes] (.sum core a (clj->js axes)))
  ([a axes keepdims] (.sum core a (clj->js axes) keepdims)))

(defn prod
  ([a]      (.prod core a))
  ([a axes] (.prod core a (clj->js axes))))

(defn mean
  ([a]      (.mean core a))
  ([a axes] (.mean core a (clj->js axes))))

(defn variance
  ([a]      (.variance core a))
  ([a axes] (.variance core a (clj->js axes))))

(defn std
  ([a]      (.std core a))
  ([a axes] (.std core a (clj->js axes))))

(defn amax
  ([a]      (.max core a))
  ([a axes] (.max core a (clj->js axes))))

(defn amin
  ([a]      (.min core a))
  ([a axes] (.min core a (clj->js axes))))

(defn argmax
  ([a]      (.argmax core a))
  ([a axis] (.argmax core a axis)))

(defn argmin
  ([a]      (.argmin core a))
  ([a axis] (.argmin core a axis)))

(defn argsort
  "Return indices that sort the array along the given axis (default: last axis).
   Requires custom MLX build with argsort support."
  ([a]      (.argsort core a))
  ([a axis] (.argsort core a axis)))

(defn sort-arr
  "Sort array along the given axis (default: last axis).
   Requires custom MLX build with sort support."
  ([a]      (.sort core a))
  ([a axis] (.sort core a axis)))

(defn topk
  "Return the top-k largest values along the last axis.
   Requires custom MLX build with topk support."
  [a k]
  (.topk core a k))

(defn logsumexp
  ([a]      (.logsumexp core a))
  ([a axes] (.logsumexp core a (clj->js axes))))

(defn cumsum
  ([a]      (.cumsum core a))
  ([a axis] (.cumsum core a axis)))

(defn logcumsumexp
  "Cumulative log-sum-exp along axis."
  ([a]      (.logcumsumexp core a))
  ([a axis] (.logcumsumexp core a axis)))

;; ---------------------------------------------------------------------------
;; Comparison / selection
;; ---------------------------------------------------------------------------

(defn equal        [a b] (.equal core a b))
(defn not-equal    [a b] (.notEqual core a b))
(defn greater      [a b] (.greater core a b))
(defn greater-equal [a b] (.greaterEqual core a b))
(defn less         [a b] (.less core a b))
(defn less-equal   [a b] (.lessEqual core a b))
(defn where        [cond a b] (.where core cond a b))
(defn isnan        [a] (.isnan core a))
(defn isinf        [a] (.isinf core a))
(defn nan-to-num
  "Replace NaN/Inf with finite values. Default: NaN→0."
  ([a]                             (.nanToNum core a 0.0))
  ([a nan-val]                     (.nanToNum core a nan-val))
  ([a nan-val posinf-val neginf-val] (.nanToNum core a nan-val posinf-val neginf-val)))

;; ---------------------------------------------------------------------------
;; Shape manipulation
;; ---------------------------------------------------------------------------

(defn reshape    [a sh] (.reshape core a (clj->js sh)))
(defn flatten    [a]    (.flatten core a))
(defn squeeze
  "Remove size-1 dimensions. With axes, only squeeze specified positions."
  ([a]      (.squeeze core a))
  ([a axes] (.squeeze core a (clj->js (vec axes)))))
(defn expand-dims [a axis] (.expandDims core a axis))
(defn transpose
  ([a]      (.transpose core a))
  ([a axes] (.transpose core a (clj->js axes))))
(defn stack
  ([arrs]      (.stack core (clj->js arrs)))
  ([arrs axis] (.stack core (clj->js arrs) axis)))
(defn concatenate
  ([arrs]      (.concatenate core (clj->js arrs)))
  ([arrs axis] (.concatenate core (clj->js arrs) axis)))
(defn broadcast-to [a sh] (.broadcastTo core a (clj->js sh)))
(defn tile [a reps] (.tile core a (clj->js reps)))
(defn repeat-arr [a repeats axis] (.repeat core a repeats axis))
(defn split-arr
  ([a sections]      (vec (js->clj (.split core a sections))))
  ([a sections axis] (vec (js->clj (.split core a sections axis)))))

;; ---------------------------------------------------------------------------
;; Indexing
;; ---------------------------------------------------------------------------

(defn take-idx
  ([a indices]      (.take core a indices))
  ([a indices axis] (.take core a indices axis)))

(defn take-along-axis [a indices axis]
  (.takeAlongAxis core a indices axis))

(defn index
  "Index along axis 0. For 1D: returns scalar element. For 2D: returns row."
  [a i]
  (.index a i))

(defn slice
  "Slice along axis 0. Returns elements [start, stop) with optional step."
  ([a start stop]      (.index a (.Slice core start stop)))
  ([a start stop step] (.index a (.Slice core start stop step))))

(defn mat-get
  "Get element [i,j] from a 2D array. Returns a scalar MLX array."
  [a i j]
  (.index (.index a i) j))

;; ---------------------------------------------------------------------------
;; Matrix operations
;; ---------------------------------------------------------------------------

(defn matmul    [a b] (.matmul core a b))
(defn inner     [a b] (.inner core a b))
(defn outer     [a b] (.outer core a b))
(defn diag      [a]   (.diag core a))
(defn trace-mat
  "Matrix trace (sum of diagonal elements)."
  ([a]                  (.trace core a 0 0 1))
  ([a offset]           (.trace core a offset 0 1))
  ([a offset ax1 ax2]   (.trace core a offset ax1 ax2)))
(defn einsum
  "Einstein summation. E.g. (einsum \"ij,jk->ik\" a b)"
  [subscripts & arrays]
  (.einsum core subscripts (to-array arrays)))

;; ---------------------------------------------------------------------------
;; Linear algebra (CPU stream)
;; ---------------------------------------------------------------------------

(defn cholesky [a]   (.cholesky linalg a false cpu-stream))
(defn solve   [a b]  (.solve linalg a b cpu-stream))
(defn solve-triangular [a b upper]
  (.solveTriangular linalg a b upper cpu-stream))
(defn inv     [a]    (.inv linalg a cpu-stream))
(defn tri-inv [a upper] (.triInv linalg a upper cpu-stream))
(defn cholesky-inv
  "Inverse of A from its Cholesky factor L (where A=LL^T)."
  ([a]       (.choleskyInv linalg a false cpu-stream))
  ([a upper] (.choleskyInv linalg a upper cpu-stream)))
(defn qr [a]
  (let [result (.qr linalg a cpu-stream)]
    [(aget result 0) (aget result 1)]))
(defn svd [a]
  (let [result (.svd linalg a cpu-stream)]
    [(aget result 0) (aget result 1) (aget result 2)]))
(defn eigh [a]
  (let [result (.eigh linalg a cpu-stream)]
    [(aget result 0) (aget result 1)]))
(defn eigvalsh [a] (.eigvalsh linalg a cpu-stream))
(defn norm
  ([a]     (.norm linalg a))
  ([a ord] (.norm linalg a ord)))

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

(defn grad
  "Compute gradient of f. Tracks grad-depth so p/generate can skip
   the L3 analytical path (which uses volatile! and breaks gradient flow)."
  ([f]
   (let [gf (.grad core f)]
     (fn [& args]
       (swap! grad-depth inc)
       (try (apply gf args)
            (finally (swap! grad-depth dec))))))
  ([f argnums]
   (let [gf (.grad core f (clj->js argnums))]
     (fn [& args]
       (swap! grad-depth inc)
       (try (apply gf args)
            (finally (swap! grad-depth dec)))))))

(defn value-and-grad
  "Compute value and gradient of f. Tracks grad-depth."
  ([f]
   (let [vg (.valueAndGrad core f)]
     (fn [& args]
       (swap! grad-depth inc)
       (try (let [result (apply vg args)]
              [(aget result 0) (aget result 1)])
            (finally (swap! grad-depth dec))))))
  ([f argnums]
   (let [vg (.valueAndGrad core f (clj->js argnums))]
     (fn [& args]
       (swap! grad-depth inc)
       (try (let [result (apply vg args)]
              [(aget result 0) (aget result 1)])
            (finally (swap! grad-depth dec)))))))

(defn jvp [f primals tangents]
  (let [result (.jvp core f (clj->js primals) (clj->js tangents))]
    [(aget result 0) (aget result 1)]))

(defn vjp [f primals cotangents]
  (let [result (.vjp core f (clj->js primals) (clj->js cotangents))]
    [(aget result 0) (aget result 1)]))

(defn stop-gradient [a] (.stopGradient core a))

;; ---------------------------------------------------------------------------
;; Transforms
;; ---------------------------------------------------------------------------

(defn compile-fn
  ([f]    (.compile core f))
  ([f shapeless?]
   (if shapeless?
     (.compile core f true)
     (.compile core f))))

(defn vmap
  ([f]                     (.vmap core f))
  ([f in-axes]             (.vmap core f (clj->js in-axes)))
  ([f in-axes out-axes]    (.vmap core f (clj->js in-axes) (clj->js out-axes))))

;; ---------------------------------------------------------------------------
;; Async
;; ---------------------------------------------------------------------------

(defn async-eval! [& arrays]
  (apply (.-asyncEval core) arrays))

;; ---------------------------------------------------------------------------
;; Device / Stream
;; ---------------------------------------------------------------------------

(defn default-device [] (.defaultDevice core))
(defn set-default-device! [d] (.setDefaultDevice core d))
(def cpu (.-cpu core))
(def gpu (.-gpu core))

;; ---------------------------------------------------------------------------
;; Constants
;; ---------------------------------------------------------------------------

(def pi    (.-pi core))
(def e-val (.-e core))
(def inf   (.-inf core))
(def nan   (.-nan core))

;; ---------------------------------------------------------------------------
;; Softmax
;; ---------------------------------------------------------------------------

(defn softmax
  ([a]      (.softmax core a))
  ([a axis] (.softmax core a axis)))

;; ---------------------------------------------------------------------------
;; Utilities
;; ---------------------------------------------------------------------------

(def ^:private MxArray
  "Constructor of MLX arrays — used for fast instance? checks."
  (.-constructor (.array core 0)))

(defn array? [x]
  (instance? MxArray x))

(defn realize
  "Evaluate a lazy MLX array and return its scalar JS value."
  [x] (eval! x) (item x))

;; ---------------------------------------------------------------------------
;; Layer 0 boundary helpers — ALL eval!/tidy calls in Layers 1-8 flow
;; through these. Keeps side-effectful materialization confined to mlx.cljs.
;; ---------------------------------------------------------------------------

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
      (clear-cache!))
    @result-vol))

(defn tidy-scalar
  "Run f inside mx/tidy, extract a JS number via item, return it.
   All intermediate MLX arrays are freed. The returned value is a
   plain JS number with no MLX references — safe for use in loops.
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
      (clear-cache!))
    @result-vol))

;; ---------------------------------------------------------------------------
;; Resource management (moved from inference/util.cljs)
;; ---------------------------------------------------------------------------

(def ^:private gc-fn
  "Synchronous GC function (Bun.gc or global.gc if available)."
  (or (when (exists? js/Bun) (.-gc js/Bun))
      (.-gc js/globalThis)))

(defn force-gc!
  "Force a synchronous garbage collection cycle and immediately sweep dead
   array wrappers to release Metal buffers. The sweep step is critical because
   N-API finalizers are deferred to the event loop — without sweeping, Metal
   buffers accumulate even after GC marks their JS wrappers as dead.
   Uses Bun.gc(true) or global.gc() if available; sweep always runs."
  []
  (when gc-fn (gc-fn true))
  (sweep-dead-arrays!))

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
     (array? x) (if (= (.-dtype x) dtype) x (astype x dtype))
     (fn? x) x
     (keyword? x) x
     (map? x) x
     (or (vector? x) (seq? x) (sequential? x)) (array x dtype)
     :else (scalar x dtype))))
