(ns mx)

;; ============================================================
;; mx contract — Scittle bindings to jax-js/WebGPU
;; Generated from the mx contract. Each function delegates to
;; a global js/mx_<name> function set up by the host page.
;; ============================================================

;; --- Array creation ---
(defn array [v] (.call js/mx_array nil (clj->js v)))
(defn scalar
  ([v] (.call js/mx_scalar nil v))
  ([v dtype] (.call js/mx_scalar nil v dtype)))
(defn zeros
  ([sh] (.call js/mx_zeros nil (clj->js sh)))
  ([sh dtype] (.call js/mx_zeros nil (clj->js sh) dtype)))
(defn ones
  ([sh] (.call js/mx_ones nil (clj->js sh)))
  ([sh dtype] (.call js/mx_ones nil (clj->js sh) dtype)))
(defn eye [n] (.call js/mx_eye nil n))
(defn arange
  ([stop] (.call js/mx_arange nil stop))
  ([start stop] (.call js/mx_arange nil start stop))
  ([start stop step] (.call js/mx_arange nil start stop step)))
(defn linspace [a b n] (.call js/mx_linspace nil a b n))
(defn full [sh v] (.call js/mx_full nil (clj->js sh) v))
(defn meshgrid [a b] (.call js/mx_meshgrid nil a b))

;; --- Dtypes ---
(def float32 "float32")
(def float64 "float64")
(def int32 "int32")
(def bool-dt "bool")

;; --- Eval & inspect ---
(defn item [a] (.call js/mx_item nil a))
(defn to-vec [a] (js->clj (.call js/mx_toVec nil a)))
(defn ->clj [a] (js->clj (.call js/mx_toVec nil a)))
(defn shape [a] (js->clj (.call js/mx_shape nil a)))
(defn ndim [a] (.call js/mx_ndim nil a))
(defn dtype [a] (.call js/mx_dtype nil a))
(defn size [a] (reduce * 1 (shape a)))
(defn array? [x] (and (object? x) (some? (.-shape x)) (some? (.-dtype x))))
(defn eval! [& arrs] nil)
(defn materialize! [& arrs] nil)
(defn realize [x] (item x))
(defn realize-clj [x] (->clj x))
(defn astype [a dtype] (.call js/mx_astype nil a dtype))

;; --- Arithmetic ---
(defn add
  ([a b] (.call js/mx_add nil a b))
  ([a b & more] (reduce add (add a b) more)))
(defn subtract
  ([a b] (.call js/mx_subtract nil a b))
  ([a b & more] (reduce subtract (subtract a b) more)))
(defn multiply
  ([a b] (.call js/mx_multiply nil a b))
  ([a b & more] (reduce multiply (multiply a b) more)))
(defn divide [a b] (.call js/mx_divide nil a b))
(defn negative [a] (.call js/mx_negative nil a))
(defn power [a b] (.call js/mx_power nil a b))
(defn square [a] (.call js/mx_square nil a))
(defn sqrt [a] (.call js/mx_sqrt nil a))
(defn abs [a] (.call js/mx_abs nil a))
(defn maximum [a b] (.call js/mx_maximum nil a b))
(defn minimum [a b] (.call js/mx_minimum nil a b))
(defn clip [a lo hi] (.call js/mx_clip nil a lo hi))
(defn sign [a] (.call js/mx_sign nil a))
(defn reciprocal [a] (.call js/mx_reciprocal nil a))
(defn floor-divide [a b] (.call js/mx_floorDivide nil a b))
(defn remainder [a b] (.call js/mx_remainder nil a b))

;; --- Math functions ---
(defn exp [a] (.call js/mx_exp nil a))
(defn expm1 [a] (.call js/mx_expm1 nil a))
(defn log [a] (.call js/mx_log nil a))
(defn log2 [a] (.call js/mx_log2 nil a))
(defn log10 [a] (.call js/mx_log10 nil a))
(defn log1p [a] (.call js/mx_log1p nil a))
(defn logaddexp [a b] (.call js/mx_logaddexp nil a b))
(defn sin [a] (.call js/mx_sin nil a))
(defn cos [a] (.call js/mx_cos nil a))
(defn tan [a] (.call js/mx_tan nil a))
(defn arccos [a] (.call js/mx_arccos nil a))
(defn tanh [a] (.call js/mx_tanh nil a))
(defn sigmoid [a] (.call js/mx_sigmoid nil a))
(defn erf [a] (.call js/mx_erf nil a))
(defn erfinv [a] (.call js/mx_erfinv nil a))
(defn lgamma [a] (.call js/mx_lgamma nil a))
(defn digamma [a] (.call js/mx_digamma nil a))
(defn bessel-i0e [a] (.call js/mx_besselI0e nil a))
(defn bessel-i1e [a] (.call js/mx_besselI1e nil a))
(defn floor [a] (.call js/mx_floor nil a))
(defn ceil [a] (.call js/mx_ceil nil a))
(defn round [a] (.call js/mx_round nil a))
(defn softmax
  ([a] (.call js/mx_softmax nil a))
  ([a axis] (.call js/mx_softmax nil a axis)))

;; --- Reductions ---
(defn sum
  ([a] (.call js/mx_sum nil a))
  ([a axes] (.call js/mx_sum nil a (clj->js axes)))
  ([a axes keepdims] (.call js/mx_sum nil a (clj->js axes) keepdims)))
(defn prod
  ([a] (.call js/mx_prod nil a))
  ([a axes] (.call js/mx_prod nil a (clj->js axes))))
(defn mean
  ([a] (.call js/mx_mean nil a))
  ([a axes] (.call js/mx_mean nil a (clj->js axes))))
(defn variance
  ([a] (.call js/mx_variance nil a))
  ([a axes] (.call js/mx_variance nil a (clj->js axes))))
(defn std
  ([a] (.call js/mx_std nil a))
  ([a axes] (.call js/mx_std nil a (clj->js axes))))
(defn amax
  ([a] (.call js/mx_amax nil a))
  ([a axes] (.call js/mx_amax nil a (clj->js axes))))
(defn amin
  ([a] (.call js/mx_amin nil a))
  ([a axes] (.call js/mx_amin nil a (clj->js axes))))
(defn argmax
  ([a] (.call js/mx_argmax nil a))
  ([a axis] (.call js/mx_argmax nil a axis)))
(defn argmin
  ([a] (.call js/mx_argmin nil a))
  ([a axis] (.call js/mx_argmin nil a axis)))
(defn all
  ([a] (.call js/mx_all nil a))
  ([a axis] (.call js/mx_all nil a axis)))
(defn any
  ([a] (.call js/mx_any nil a))
  ([a axis] (.call js/mx_any nil a axis)))
(defn argsort
  ([a] (.call js/mx_argsort nil a))
  ([a axis] (.call js/mx_argsort nil a axis)))
(defn sort-arr
  ([a] (.call js/mx_sortArr nil a))
  ([a axis] (.call js/mx_sortArr nil a axis)))
(defn topk [a k] (.call js/mx_topk nil a k))
(defn logsumexp
  ([a] (.call js/mx_logsumexp nil a))
  ([a axes] (.call js/mx_logsumexp nil a (clj->js axes)))
  ([a axes keepdims] (.call js/mx_logsumexp nil a (clj->js axes) keepdims)))
(defn cumsum
  ([a] (.call js/mx_cumsum nil a))
  ([a axis] (.call js/mx_cumsum nil a axis)))
(defn logcumsumexp
  ([a] (.call js/mx_logcumsumexp nil a))
  ([a axis] (.call js/mx_logcumsumexp nil a axis)))
(defn searchsorted
  ([sorted-arr values] (.call js/mx_searchsorted nil sorted-arr values))
  ([sorted-arr values side] (.call js/mx_searchsorted nil sorted-arr values side)))

;; --- Comparison & selection ---
(defn equal [a b] (.call js/mx_equal nil a b))
(defn not-equal [a b] (.call js/mx_notEqual nil a b))
(defn greater [a b] (.call js/mx_greater nil a b))
(defn greater-equal [a b] (.call js/mx_greaterEqual nil a b))
(defn less [a b] (.call js/mx_less nil a b))
(defn less-equal [a b] (.call js/mx_lessEqual nil a b))
(defn where [cond a b] (.call js/mx_where nil cond a b))
(defn eq? [a b] (astype (equal (if (number? a) (scalar a int32) a)
                                (if (number? b) (scalar b int32) b)) float32))
(defn neq? [a b] (astype (not-equal (if (number? a) (scalar a int32) a)
                                     (if (number? b) (scalar b int32) b)) float32))
(defn gt? [a b] (astype (greater (if (number? a) (scalar a) a)
                                  (if (number? b) (scalar b) b)) float32))
(defn lt? [a b] (astype (less (if (number? a) (scalar a) a)
                               (if (number? b) (scalar b) b)) float32))
(defn and* [a b] (multiply a b))
(defn or* [a b] (maximum a b))
(defn isnan [a] (.call js/mx_isnan nil a))
(defn isinf [a] (.call js/mx_isinf nil a))
(defn nan-to-num
  ([a] (.call js/mx_nanToNum nil a))
  ([a nan-val] (.call js/mx_nanToNum nil a nan-val))
  ([a nan-val posinf neginf] (.call js/mx_nanToNum nil a nan-val posinf neginf)))

;; --- Shape manipulation ---
(defn reshape [a sh] (.call js/mx_reshape nil a (clj->js sh)))
(defn flatten [a] (.call js/mx_flatten nil a))
(defn squeeze
  ([a] (.call js/mx_squeeze nil a))
  ([a axes] (.call js/mx_squeeze nil a (clj->js (vec axes)))))
(defn expand-dims [a axis] (.call js/mx_expandDims nil a axis))
(defn transpose
  ([a] (.call js/mx_transpose nil a))
  ([a axes] (.call js/mx_transpose nil a (clj->js axes))))
(defn stack
  ([arrs] (.call js/mx_stack nil (clj->js (vec arrs))))
  ([arrs axis] (.call js/mx_stack nil (clj->js (vec arrs)) axis)))
(defn concatenate
  ([arrs] (.call js/mx_concatenate nil (clj->js (vec arrs))))
  ([arrs axis] (.call js/mx_concatenate nil (clj->js (vec arrs)) axis)))
(defn broadcast-to [a sh] (.call js/mx_broadcastTo nil a (clj->js sh)))
(defn tile [a reps] (.call js/mx_tile nil a (clj->js reps)))
(defn repeat-arr [a repeats axis] (.call js/mx_repeatArr nil a repeats axis))
(defn split-arr
  ([a sections] (js->clj (.call js/mx_splitArr nil a sections)))
  ([a sections axis] (js->clj (.call js/mx_splitArr nil a sections axis))))

;; --- Indexing (take-idx uses raw gather; idx defined after matmul for differentiability) ---
(defn take-idx
  ([a indices] (.call js/mx_take nil a indices 0))
  ([a indices axis] (.call js/mx_take nil a indices axis)))
(defn take-along-axis [a indices axis] (.call js/mx_takeAlongAxis nil a indices axis))
(defn slice
  ([a start stop] (.call js/mx_slice nil a start stop))
  ([a start stop step] (.call js/mx_slice nil a start stop step)))

;; --- Matrix operations ---
(defn matmul [a b] (.call js/mx_matmul nil a b))
(defn inner [a b] (.call js/mx_inner nil a b))
(defn outer [a b] (.call js/mx_outer nil a b))
(defn diag [a] (.call js/mx_diag nil a))
(defn trace-mat
  ([a] (.call js/mx_traceMat nil a))
  ([a offset] (.call js/mx_traceMat nil a offset))
  ([a offset ax1 ax2] (.call js/mx_traceMat nil a offset ax1 ax2)))
(defn einsum [subscripts & arrays] (apply js/mx_einsum subscripts arrays))
(defn dot [a b] (.call js/mx_dot nil a b))

;; --- Differentiable indexing (after matmul is defined) ---
(defn idx
  "Differentiable indexing via one-hot selection.
   Works inside mx/grad (jax-js lacks gather VJP, so we use
   one-hot * array -> sum, which has full autodiff support)."
  ([a i]
   (let [n    (first (shape a))
         oh   (.call js/mx_oneHot nil i n)]
     (if (= 1 (count (shape a)))
       (sum (multiply oh a))
       (squeeze (matmul (reshape oh [1 n]) a)))))
  ([a i axis]
   (let [n  (nth (shape a) axis)
         oh (.call js/mx_oneHot nil i n)]
     (if (= axis 0)
       (squeeze (matmul (reshape oh [1 n]) a))
       (squeeze (matmul a (reshape oh [n 1])))))))
(defn index [a i] (idx a i))
(defn mat-get [a i j] (idx (idx a i 0) j 0))

;; --- Linear algebra ---
(defn cholesky [a] (.call js/mx_cholesky nil a))
(defn solve [a b] (.call js/mx_solve nil a b))
(defn solve-triangular [a b upper] (.call js/mx_solveTriangular nil a b upper))
(defn inv [a] (.call js/mx_inv nil a))
(defn tri-inv [a upper] (.call js/mx_triInv nil a upper))
(defn cholesky-inv
  ([a] (.call js/mx_choleskyInv nil a))
  ([a upper] (.call js/mx_choleskyInv nil a upper)))
(defn qr [a] (.call js/mx_qr nil a))
(defn svd [a] (.call js/mx_svd nil a))
(defn eigh [a] (.call js/mx_eigh nil a))
(defn eigvalsh [a] (.call js/mx_eigvalsh nil a))
(defn norm
  ([a] (.call js/mx_norm nil a))
  ([a ord] (.call js/mx_norm nil a ord)))
(defn logdet [a] (.call js/mx_logdet nil a))
(defn det [a] (.call js/mx_det nil a))

;; --- Autograd ---
(defn grad
  ([f] (.call js/mx_grad nil f))
  ([f argnums] (.call js/mx_grad nil f (clj->js argnums))))
(defn value-and-grad
  ([f] (.call js/mx_valueAndGrad nil f))
  ([f argnums] (.call js/mx_valueAndGrad nil f (clj->js argnums))))
(defn jvp [f primals tangents]
  (.call js/mx_jvp nil f (clj->js primals) (clj->js tangents)))
(defn vjp [f primals cotangents]
  (.call js/mx_vjp nil f primals cotangents))
(defn stop-gradient [a] (.call js/mx_stopGradient nil a))

;; --- Transforms ---
(defn compile-fn
  ([f] (.call js/mx_jit nil f))
  ([f shapeless?] (.call js/mx_jit nil f)))
(defn compile-clear-cache! [] nil)
(defn vmap
  ([f] (.call js/mx_vmap nil f))
  ([f in-axes] (.call js/mx_vmap nil f (clj->js in-axes)))
  ([f in-axes out-axes] (.call js/mx_vmap nil f (clj->js in-axes) (clj->js out-axes))))

;; --- Random PRNG ---
(defn fresh-key
  ([] (.call js/mx_freshKey nil))
  ([seed] (.call js/mx_freshKey nil seed)))
(defn split-key [k] (js->clj (.call js/mx_splitKey nil k)))
(defn normal [key shape] (.call js/mx_normal nil key (clj->js shape)))
(defn uniform [key shape] (.call js/mx_uniform nil key (clj->js shape)))
(defn bernoulli [key p shape] (.call js/mx_bernoulli nil key p (clj->js shape)))
(defn categorical [key logits] (.call js/mx_categorical nil key logits))
(defn gumbel [key shape] (.call js/mx_gumbel nil key (clj->js shape)))
(defn laplace [key shape] (.call js/mx_laplace nil key (clj->js shape)))
(defn multivariate-normal
  ([key mean cov] (.call js/mx_multivariateNormal nil key mean cov))
  ([key mean cov shape] (.call js/mx_multivariateNormal nil key mean cov (clj->js shape))))
(defn permutation [key n] (.call js/mx_permutation nil key n))

;; --- Constants ---
(def pi js/Math.PI)
(def e-val js/Math.E)
(def inf js/Infinity)
(def nan js/NaN)

;; --- Device ---
(defn default-device [] "webgpu")
(defn set-default-device! [d] nil)
(def cpu "cpu")
(def gpu "webgpu")

;; --- Memory management (no-ops for WebGPU) ---
(defn tidy [f] (f))
(defn dispose! [a] (when (and a (object? a) (.-dispose a)) (.dispose a)))
(defn tidy-materialize [f] (f))
(defn tidy-run [f collect-fn] (let [r (f)] (collect-fn r) r))
(defn tidy-scalar [f] (item (f)))
(defn with-resource-guard [f] (f))
(defn auto-cleanup! ([] nil) ([aggressive?] nil))
(defn gfi-cleanup! [] nil)
(defn force-gc! [] nil)
(defn jsc-cleanup! [] nil)
(defn sweep-dead-arrays! [] nil)
(defn clear-cache! [] nil)
(defn set-memory-limit! [n] nil)
(defn set-cache-limit! [n] nil)
(defn set-wired-limit! [n] nil)
(defn get-active-memory [] 0)
(defn get-cache-memory [] 0)
(defn get-peak-memory [] 0)
(defn reset-peak-memory! [] nil)
(defn get-wrappers-count [] 0)
(defn get-num-resources [] 0)
(defn get-resource-limit [] 999999)
(defn metal-is-available? [] false)
(defn metal-device-info [] {})
(defn memory-report [] {:active-bytes 0 :cache-bytes 0 :peak-bytes 0
                         :wrappers 0 :num-resources 0 :resource-limit 999999})
(defn async-eval! [& arrs] nil)
(defn in-grad? [] false)
(defn in-compile? [] false)
(defn in-tidy? [] false)

;; --- Utilities ---
(defn ensure-array
  ([x] (cond (array? x) x (number? x) (scalar x) (or (vector? x) (sequential? x)) (array x) :else x))
  ([x dtype] (cond (array? x) (if (= (.-dtype x) dtype) x (astype x dtype))
                    (number? x) (scalar x dtype)
                    (or (vector? x) (sequential? x)) (array x) :else x)))
(defn training-step! [& args] (throw (js/Error. "training-step! not available in WebGPU")))
