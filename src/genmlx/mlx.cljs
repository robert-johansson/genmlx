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

(def ^:dynamic *batched-exec?* false)

(defn eval! [& arrs]
  (when *batched-exec?*
    (js/console.warn
      (str "mx/eval! called during batched execution. This materializes the computation "
           "graph and may produce incorrect results or break vectorization. Move eval!/item "
           "calls outside the gen body, or use scalar execution instead of vsimulate/vgenerate.")))
  (apply (.-eval core) arrs))

(defn item [a]
  (when *batched-exec?*
    (js/console.warn
      (str "mx/item called during batched execution. This materializes the computation "
           "graph and may produce incorrect results or break vectorization. Move eval!/item "
           "calls outside the gen body, or use scalar execution instead of vsimulate/vgenerate.")))
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

(defn tidy [f] (.tidy core f))

(defn dispose! [a] (.dispose core a))

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
(defn tanh     [a] (.tanh core a))
(defn sigmoid  [a] (.sigmoid core a))
(defn erf      [a] (.erf core a))
(defn erfinv   [a] (.erfinv core a))

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

(defn logsumexp
  ([a]      (.logsumexp core a))
  ([a axes] (.logsumexp core a (clj->js axes))))

(defn cumsum
  ([a]      (.cumsum core a))
  ([a axis] (.cumsum core a axis)))

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

;; ---------------------------------------------------------------------------
;; Shape manipulation
;; ---------------------------------------------------------------------------

(defn reshape    [a sh] (.reshape core a (clj->js sh)))
(defn flatten    [a]    (.flatten core a))
(defn squeeze    [a]    (.squeeze core a))
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

(defn index [a i]
  (.take core a (scalar i int32)))

(defn slice
  ([a start stop]
   (.index a (new (.-Slice core) start stop)))
  ([a start stop step]
   (.index a (new (.-Slice core) start stop step))))

;; ---------------------------------------------------------------------------
;; Matrix operations
;; ---------------------------------------------------------------------------

(defn matmul    [a b] (.matmul core a b))
(defn inner     [a b] (.inner core a b))
(defn outer     [a b] (.outer core a b))
(defn diag      [a]   (.diag core a))

;; ---------------------------------------------------------------------------
;; Linear algebra (CPU stream)
;; ---------------------------------------------------------------------------

(defn cholesky [a]   (.cholesky linalg a false cpu-stream))
(defn solve   [a b]  (.solve linalg a b cpu-stream))
(defn solve-triangular [a b upper]
  (.solveTriangular linalg a b upper cpu-stream))
(defn inv     [a]    (.inv linalg a cpu-stream))
(defn tri-inv [a upper] (.triInv linalg a upper cpu-stream))
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

;; ---------------------------------------------------------------------------
;; Random number generation (stateful — for convenience outside handlers)
;; ---------------------------------------------------------------------------

(defn random-seed! [seed] (.seed random seed))
(defn random-normal
  ([shape]       (.normal random (clj->js shape)))
  ([shape dtype] (.normal random (clj->js shape) dtype)))
(defn random-uniform
  ([shape]            (.uniform random (scalar 0) (scalar 1) (clj->js shape)))
  ([lo hi shape]      (.uniform random (scalar lo) (scalar hi) (clj->js shape))))
(defn random-bernoulli [p shape] (.bernoulli random (scalar p) (clj->js shape)))
(defn random-categorical
  ([logits]             (.categorical random logits))
  ([logits num-samples] (.categorical random logits num-samples)))
(defn random-randint [lo hi shape] (.randint random lo hi (clj->js shape)))
(defn random-gumbel [shape] (.gumbel random (clj->js shape)))
(defn random-laplace [shape] (.laplace random (clj->js shape)))

;; ---------------------------------------------------------------------------
;; Autograd
;; ---------------------------------------------------------------------------

(defn grad
  ([f]         (.grad core f))
  ([f argnums] (.grad core f (clj->js argnums))))

(defn value-and-grad
  ([f]
   (let [vg (.valueAndGrad core f)]
     (fn [& args]
       (let [result (apply vg args)]
         [(aget result 0) (aget result 1)]))))
  ([f argnums]
   (let [vg (.valueAndGrad core f (clj->js argnums))]
     (fn [& args]
       (let [result (apply vg args)]
         [(aget result 0) (aget result 1)])))))

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

(defn array? [x]
  (and (some? x)
       (object? x)
       (some? (.-shape x))
       (fn? (.-item x))))

(defn realize
  "Evaluate a lazy MLX array and return its scalar JS value."
  [x] (eval! x) (item x))

(defn ensure-array
  "Wrap a JS number as an MLX scalar array; pass through existing arrays.
   Vectors and sequences are converted to MLX arrays."
  ([x]
   (cond
     (array? x) x
     (or (vector? x) (seq? x) (sequential? x)) (array x)
     :else (scalar x)))
  ([x dtype]
   (cond
     (array? x) x
     (or (vector? x) (seq? x) (sequential? x)) (array x dtype)
     :else (scalar x dtype))))
