# MLX Operations

Thin ClojureScript wrapper over Apple's MLX framework (`@frost-beta/mlx`).
All operations produce lazy computation graphs -- call `mx/eval!` to materialize.
Values stay as MLX arrays from sampling through scoring through gradient computation.

Source: `src/genmlx/mlx.cljs`, `src/genmlx/mlx/random.cljs`

---

## Array Creation

### `scalar`

```clojure
(mx/scalar 3.14)
(mx/scalar 3.14 mx/float64)
```

Create a scalar (0-dimensional) MLX array from a JS number.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | number | JS number value |
| `dtype` | dtype (optional) | Data type (default: inferred) |

**Returns:** MLX scalar array with shape `[]`

---

### `array`

```clojure
(mx/array [1 2 3])
(mx/array [1 2 3 4 5 6] [2 3])
(mx/array [1 2 3] mx/float32)
(mx/array [1 2 3 4] [2 2] mx/float64)
```

Create an MLX array from a Clojure collection. Optionally reshape and/or cast to a dtype.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | collection | Clojure vector or sequence |
| `shape` | vector (optional) | Target shape (triggers reshape) |
| `dtype` | dtype (optional) | Data type |

**Returns:** MLX array

---

### `zeros`

```clojure
(mx/zeros [3 4])
(mx/zeros [3 4] mx/float64)
```

Create an array filled with zeros. Like `numpy.zeros`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | vector | Shape of the output array |
| `dtype` | dtype (optional) | Data type (default: float32) |

**Returns:** MLX array

---

### `ones`

```clojure
(mx/ones [3 4])
(mx/ones [2 2] mx/int32)
```

Create an array filled with ones. Like `numpy.ones`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | vector | Shape of the output array |
| `dtype` | dtype (optional) | Data type (default: float32) |

**Returns:** MLX array

---

### `full`

```clojure
(mx/full [3 3] 7.0)
```

Create an array filled with a constant value. Like `numpy.full`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | vector | Shape of the output array |
| `val` | number | Fill value |

**Returns:** MLX array

---

### `eye`

```clojure
(mx/eye 3)
(mx/eye 4 mx/float64)
```

Create an identity matrix. Like `numpy.eye`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | integer | Size of the square matrix |
| `dtype` | dtype (optional) | Data type (default: float32) |

**Returns:** MLX array with shape `[n n]`

---

### `arange`

```clojure
(mx/arange 5)          ;; [0 1 2 3 4]
(mx/arange 2 7)        ;; [2 3 4 5 6]
(mx/arange 0 10 2)     ;; [0 2 4 6 8]
```

Create an array of evenly spaced values. Like `numpy.arange`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | number (optional) | Start value (default: 0) |
| `stop` | number | End value (exclusive) |
| `step` | number (optional) | Step size (default: 1) |

**Returns:** MLX 1-D array

---

### `linspace`

```clojure
(mx/linspace 0 1 5)    ;; [0.0 0.25 0.5 0.75 1.0]
```

Create an array of evenly spaced values over an interval. Like `numpy.linspace`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | number | Start value |
| `stop` | number | End value (inclusive) |
| `num` | integer | Number of samples |

**Returns:** MLX 1-D array

---

### `meshgrid`

```clojure
(mx/meshgrid (mx/arange 3) (mx/arange 4))
```

Create coordinate grids from 1-D arrays. Like `numpy.meshgrid`. Returns a JS array of two MLX arrays, each with shape `[len(a) len(b)]`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First 1-D coordinate array |
| `b` | MLX array | Second 1-D coordinate array |

**Returns:** JS array `#js [grid-a grid-b]`

---

## Array Properties

### `shape`

```clojure
(mx/shape a)    ;; => [3 4]
```

Get the shape of an array as a Clojure vector.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** Clojure vector of integers

---

### `ndim`

```clojure
(mx/ndim a)     ;; => 2
```

Get the number of dimensions of an array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** integer

---

### `dtype`

```clojure
(mx/dtype a)    ;; => mx/float32
```

Get the data type of an array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** dtype object

---

### `size`

```clojure
(mx/size a)     ;; => 12
```

Get the total number of elements in an array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** integer

---

### `item`

```clojure
(mx/item a)     ;; => 3.14
```

Extract a JS number from a scalar (0-dimensional) array. Triggers evaluation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Scalar array |

**Returns:** JS number

> **Warning:** Do not call `mx/item` inside model bodies during batched execution -- it breaks vectorization by materializing intermediate values.

---

### `array?`

```clojure
(mx/array? x)   ;; => true
```

Check whether `x` is an MLX array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | any | Value to test |

**Returns:** boolean

---

### `realize`

```clojure
(mx/realize a)  ;; => 3.14
```

Evaluate a lazy MLX array and return its scalar JS value. Equivalent to `(do (mx/eval! x) (mx/item x))`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | MLX array | Scalar array |

**Returns:** JS number

---

### `ensure-array`

```clojure
(mx/ensure-array 3.14)              ;; JS number -> scalar array
(mx/ensure-array [1 2 3])           ;; vector -> MLX array
(mx/ensure-array some-array)        ;; pass through
(mx/ensure-array 3.14 mx/float64)   ;; with dtype
```

Wrap a JS number as a scalar MLX array, convert vectors/sequences to arrays, and pass through existing MLX arrays unchanged. Functions, keywords, and maps pass through unchanged.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | any | Value to convert |
| `dtype` | dtype (optional) | Target dtype (casts if needed) |

**Returns:** MLX array (or pass-through for fn/keyword/map)

---

### `->clj`

```clojure
(mx/->clj a)    ;; => [1 2 3]
```

Evaluate an MLX array and convert to a ClojureScript value (number or nested vector).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** ClojureScript value

---

### `realize-clj`

```clojure
(mx/realize-clj a)    ;; => [1 2 3]
```

Evaluate an MLX array and convert to ClojureScript data. Same as `->clj` (evaluates then converts).

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | MLX array | Input array |

**Returns:** ClojureScript value

---

## Type Conversion

### `astype`

```clojure
(mx/astype a mx/float64)
(mx/astype a mx/int32)
```

Cast an array to a different dtype. Like `numpy.ndarray.astype`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `dtype` | dtype | Target data type |

**Returns:** MLX array with new dtype

### Data type constants

| Constant | MLX dtype |
|----------|-----------|
| `mx/float32` | 32-bit float (default) |
| `mx/float64` | 64-bit float |
| `mx/int32` | 32-bit integer |
| `mx/int64` | 64-bit integer |
| `mx/bool-dt` | Boolean |

---

## Arithmetic

All arithmetic operations are element-wise and support broadcasting.

### `add`

```clojure
(mx/add a b)
(mx/add a b c d)
```

Element-wise addition. Variadic -- accepts two or more arguments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |
| `more` | MLX arrays (optional) | Additional operands |

**Returns:** MLX array

---

### `subtract`

```clojure
(mx/subtract a b)
(mx/subtract a b c)
```

Element-wise subtraction. Variadic -- accepts two or more arguments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |
| `more` | MLX arrays (optional) | Additional operands |

**Returns:** MLX array

---

### `multiply`

```clojure
(mx/multiply a b)
(mx/multiply a b c)
```

Element-wise multiplication. Variadic -- accepts two or more arguments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |
| `more` | MLX arrays (optional) | Additional operands |

**Returns:** MLX array

---

### `divide`

```clojure
(mx/divide a b)
```

Element-wise division.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Dividend |
| `b` | MLX array | Divisor |

**Returns:** MLX array

---

### `negative`

```clojure
(mx/negative a)
```

Element-wise negation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `power`

```clojure
(mx/power a b)
```

Element-wise exponentiation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Base |
| `b` | MLX array | Exponent |

**Returns:** MLX array

---

### `square`

```clojure
(mx/square a)
```

Element-wise square.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `sqrt`

```clojure
(mx/sqrt a)
```

Element-wise square root.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `abs`

```clojure
(mx/abs a)
```

Element-wise absolute value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `maximum`

```clojure
(mx/maximum a b)
```

Element-wise maximum of two arrays.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |

**Returns:** MLX array

---

### `minimum`

```clojure
(mx/minimum a b)
```

Element-wise minimum of two arrays.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |

**Returns:** MLX array

---

### `clip`

```clojure
(mx/clip a lo hi)
```

Clamp values to the range `[lo, hi]`. Like `numpy.clip`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `lo` | MLX array / number | Lower bound |
| `hi` | MLX array / number | Upper bound |

**Returns:** MLX array

---

### `sign`

```clojure
(mx/sign a)
```

Element-wise sign function. Returns -1, 0, or +1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `reciprocal`

```clojure
(mx/reciprocal a)
```

Element-wise reciprocal (1/a).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `floor-divide`

```clojure
(mx/floor-divide a b)
```

Element-wise integer division (floor of a/b). Like Python's `//` operator.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Dividend |
| `b` | MLX array | Divisor |

**Returns:** MLX array

---

### `remainder`

```clojure
(mx/remainder a b)
```

Element-wise remainder after division. Like Python's `%` operator.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Dividend |
| `b` | MLX array | Divisor |

**Returns:** MLX array

---

## Math Functions

### `exp`

```clojure
(mx/exp a)
```

Element-wise exponential.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `expm1`

```clojure
(mx/expm1 a)
```

Element-wise exp(a) - 1. Numerically stable for small values of a.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `log`

```clojure
(mx/log a)
```

Element-wise natural logarithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `log2`

```clojure
(mx/log2 a)
```

Element-wise base-2 logarithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `log10`

```clojure
(mx/log10 a)
```

Element-wise base-10 logarithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `log1p`

```clojure
(mx/log1p a)
```

Element-wise log(1 + a). Numerically stable for small values of a.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `logaddexp`

```clojure
(mx/logaddexp a b)
```

Element-wise log(exp(a) + exp(b)). Numerically stable log-space addition.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |

**Returns:** MLX array

---

### `sin`

```clojure
(mx/sin a)
```

Element-wise sine.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array (radians) |

**Returns:** MLX array

---

### `cos`

```clojure
(mx/cos a)
```

Element-wise cosine.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array (radians) |

**Returns:** MLX array

---

### `tan`

```clojure
(mx/tan a)
```

Element-wise tangent.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array (radians) |

**Returns:** MLX array

---

### `arccos`

```clojure
(mx/arccos a)
```

Element-wise inverse cosine (arccosine). Returns values in [0, pi].

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array (values in [-1, 1]) |

**Returns:** MLX array (radians)

---

### `tanh`

```clojure
(mx/tanh a)
```

Element-wise hyperbolic tangent.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `sigmoid`

```clojure
(mx/sigmoid a)
```

Element-wise sigmoid function: 1 / (1 + exp(-a)).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `softmax`

```clojure
(mx/softmax a)
(mx/softmax a axis)
```

Softmax along the given axis (default: all elements).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer (optional) | Axis for softmax |

**Returns:** MLX array

---

### `erf`

```clojure
(mx/erf a)
```

Element-wise error function. Used in Gaussian CDF computations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `erfinv`

```clojure
(mx/erfinv a)
```

Element-wise inverse error function. Useful for computing quantiles of the normal distribution.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array (values in (-1, 1)) |

**Returns:** MLX array

---

### `lgamma`

```clojure
(mx/lgamma a)
```

Element-wise log-gamma function: ln(Gamma(a)). Essential for log-probability computations involving Beta, Gamma, and Dirichlet distributions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array (positive values) |

**Returns:** MLX array

---

### `digamma`

```clojure
(mx/digamma a)
```

Element-wise digamma function: psi(a) = Gamma'(a) / Gamma(a). The derivative of the log-gamma function.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array (positive values) |

**Returns:** MLX array

---

### `bessel-i0e`

```clojure
(mx/bessel-i0e a)
```

Element-wise exponentially scaled modified Bessel function of the first kind, order 0. Returns I0(a) * exp(-|a|). Used in von Mises distribution.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `bessel-i1e`

```clojure
(mx/bessel-i1e a)
```

Element-wise exponentially scaled modified Bessel function of the first kind, order 1. Returns I1(a) * exp(-|a|).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `floor`

```clojure
(mx/floor a)
```

Element-wise floor (round toward negative infinity).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `ceil`

```clojure
(mx/ceil a)
```

Element-wise ceiling (round toward positive infinity).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

### `round`

```clojure
(mx/round a)
```

Element-wise rounding to nearest integer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array

---

## Comparison

### `equal`

```clojure
(mx/equal a b)
```

Element-wise equality test. Returns boolean array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |

**Returns:** MLX boolean array

---

### `not-equal`

```clojure
(mx/not-equal a b)
```

Element-wise inequality test. Returns boolean array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |

**Returns:** MLX boolean array

---

### `greater`

```clojure
(mx/greater a b)
```

Element-wise greater-than test.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |

**Returns:** MLX boolean array

---

### `greater-equal`

```clojure
(mx/greater-equal a b)
```

Element-wise greater-or-equal test.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |

**Returns:** MLX boolean array

---

### `less`

```clojure
(mx/less a b)
```

Element-wise less-than test.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |

**Returns:** MLX boolean array

---

### `less-equal`

```clojure
(mx/less-equal a b)
```

Element-wise less-or-equal test.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand |
| `b` | MLX array | Second operand |

**Returns:** MLX boolean array

---

### `eq?`

```clojure
(mx/eq? a b)
(mx/eq? prize 0)
```

Model-level equality helper. Auto-promotes integers to scalars, returns `float32` (not bool). Designed for use in gen bodies where traced values are tensors during enumeration.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array / number | First operand |
| `b` | MLX array / number | Second operand |

**Returns:** MLX float32 array (1.0 for true, 0.0 for false)

---

### `neq?`

```clojure
(mx/neq? a b)
```

Model-level inequality helper. Auto-promotes integers, returns `float32`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array / number | First operand |
| `b` | MLX array / number | Second operand |

**Returns:** MLX float32 array (1.0 for true, 0.0 for false)

---

### `gt?`

```clojure
(mx/gt? a b)
```

Model-level greater-than helper. Auto-promotes integers, returns `float32`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array / number | First operand |
| `b` | MLX array / number | Second operand |

**Returns:** MLX float32 array (1.0 for true, 0.0 for false)

---

### `lt?`

```clojure
(mx/lt? a b)
```

Model-level less-than helper. Auto-promotes integers, returns `float32`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array / number | First operand |
| `b` | MLX array / number | Second operand |

**Returns:** MLX float32 array (1.0 for true, 0.0 for false)

---

### `and*`

```clojure
(mx/and* a b)
```

Differentiable logical AND for model bodies. Implemented as `(mx/multiply a b)` so that both arguments contribute to gradients.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand (float32, 0.0/1.0) |
| `b` | MLX array | Second operand (float32, 0.0/1.0) |

**Returns:** MLX float32 array

---

### `or*`

```clojure
(mx/or* a b)
```

Differentiable logical OR for model bodies. Implemented as `(mx/maximum a b)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First operand (float32, 0.0/1.0) |
| `b` | MLX array | Second operand (float32, 0.0/1.0) |

**Returns:** MLX float32 array

---

### `where`

```clojure
(mx/where cond a b)
```

Element-wise conditional selection. Like `numpy.where`. Use this instead of `if` for conditional values in models under vectorized inference, since different particles may take different branches.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cond` | MLX array | Boolean condition array |
| `a` | MLX array | Values where condition is true |
| `b` | MLX array | Values where condition is false |

**Returns:** MLX array

---

### `isnan`

```clojure
(mx/isnan a)
```

Element-wise NaN check.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX boolean array

---

### `isinf`

```clojure
(mx/isinf a)
```

Element-wise infinity check.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX boolean array

---

### `nan-to-num`

```clojure
(mx/nan-to-num a)
(mx/nan-to-num a 0.0)
(mx/nan-to-num a 0.0 1e30 -1e30)
```

Replace NaN and Inf values with finite numbers. Like `numpy.nan_to_num`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `nan-val` | number (optional) | Replacement for NaN (default: 0.0) |
| `posinf-val` | number (optional) | Replacement for +Inf |
| `neginf-val` | number (optional) | Replacement for -Inf |

**Returns:** MLX array

---

### `all`

```clojure
(mx/all a)
(mx/all a axis)
```

True if all elements are nonzero. With optional axis, reduces along that axis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer (optional) | Axis to reduce along |

**Returns:** MLX boolean array

---

### `any`

```clojure
(mx/any a)
(mx/any a axis)
```

True if any element is nonzero. With optional axis, reduces along that axis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer (optional) | Axis to reduce along |

**Returns:** MLX boolean array

---

## Reduction

All reductions accept optional `axes` (integer or vector) and some accept `keepdims` (boolean).

### `sum`

```clojure
(mx/sum a)
(mx/sum a [0])
(mx/sum a [0 1] true)
```

Sum of array elements. Like `numpy.sum`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | int/vector (optional) | Axes to reduce |
| `keepdims` | boolean (optional) | Keep reduced dimensions |

**Returns:** MLX array

---

### `prod`

```clojure
(mx/prod a)
(mx/prod a [0])
```

Product of array elements. Like `numpy.prod`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | int/vector (optional) | Axes to reduce |

**Returns:** MLX array

---

### `mean`

```clojure
(mx/mean a)
(mx/mean a [0])
```

Mean of array elements. Like `numpy.mean`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | int/vector (optional) | Axes to reduce |

**Returns:** MLX array

---

### `variance`

```clojure
(mx/variance a)
(mx/variance a [0])
```

Variance of array elements. Like `numpy.var`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | int/vector (optional) | Axes to reduce |

**Returns:** MLX array

---

### `std`

```clojure
(mx/std a)
(mx/std a [0])
```

Standard deviation of array elements. Like `numpy.std`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | int/vector (optional) | Axes to reduce |

**Returns:** MLX array

---

### `amax`

```clojure
(mx/amax a)
(mx/amax a [0])
```

Maximum value of array elements. Like `numpy.amax`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | int/vector (optional) | Axes to reduce |

**Returns:** MLX array

---

### `amin`

```clojure
(mx/amin a)
(mx/amin a [0])
```

Minimum value of array elements. Like `numpy.amin`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | int/vector (optional) | Axes to reduce |

**Returns:** MLX array

---

### `argmax`

```clojure
(mx/argmax a)
(mx/argmax a 0)
```

Index of the maximum value. Like `numpy.argmax`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer (optional) | Axis to reduce along |

**Returns:** MLX integer array

---

### `argmin`

```clojure
(mx/argmin a)
(mx/argmin a 0)
```

Index of the minimum value. Like `numpy.argmin`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer (optional) | Axis to reduce along |

**Returns:** MLX integer array

---

### `logsumexp`

```clojure
(mx/logsumexp a)
(mx/logsumexp a [0])
(mx/logsumexp a [0] true)
```

Numerically stable log-sum-exp: log(sum(exp(a))). Essential for log-probability normalization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | int/vector (optional) | Axes to reduce |
| `keepdims` | boolean (optional) | Keep reduced dimensions |

**Returns:** MLX array

---

### `logcumsumexp`

```clojure
(mx/logcumsumexp a)
(mx/logcumsumexp a 0)
```

Cumulative log-sum-exp along an axis. Numerically stable cumulative version of `logsumexp`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer (optional) | Axis for cumulative operation |

**Returns:** MLX array

---

### `cumsum`

```clojure
(mx/cumsum a)
(mx/cumsum a 0)
```

Cumulative sum along an axis. Like `numpy.cumsum`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer (optional) | Axis for cumulative sum |

**Returns:** MLX array

---

## Shape Manipulation

### `reshape`

```clojure
(mx/reshape a [2 3])
```

Reshape an array to a new shape. Like `numpy.reshape`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `shape` | vector | Target shape |

**Returns:** MLX array

---

### `flatten`

```clojure
(mx/flatten a)
```

Flatten an array to 1-D. Like `numpy.ndarray.flatten`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX 1-D array

---

### `squeeze`

```clojure
(mx/squeeze a)
(mx/squeeze a [0 2])
```

Remove dimensions of size 1. Like `numpy.squeeze`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | vector (optional) | Specific axes to squeeze |

**Returns:** MLX array

---

### `expand-dims`

```clojure
(mx/expand-dims a 0)
```

Add a dimension of size 1 at the given axis. Like `numpy.expand_dims`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer | Position for new axis |

**Returns:** MLX array

---

### `transpose`

```clojure
(mx/transpose a)
(mx/transpose a [1 0 2])
```

Transpose an array. Without axes, reverses all dimensions. Like `numpy.transpose`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axes` | vector (optional) | Permutation of axes |

**Returns:** MLX array

---

### `stack`

```clojure
(mx/stack [a b c])
(mx/stack [a b c] 1)
```

Stack arrays along a new axis. Like `numpy.stack`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `arrs` | vector | Arrays to stack (must have same shape) |
| `axis` | integer (optional) | Axis for the new dimension (default: 0) |

**Returns:** MLX array

---

### `concatenate`

```clojure
(mx/concatenate [a b c])
(mx/concatenate [a b c] 1)
```

Concatenate arrays along an existing axis. Like `numpy.concatenate`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `arrs` | vector | Arrays to concatenate |
| `axis` | integer (optional) | Axis to concatenate along (default: 0) |

**Returns:** MLX array

---

### `broadcast-to`

```clojure
(mx/broadcast-to a [3 4])
```

Broadcast an array to a target shape. Like `numpy.broadcast_to`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `shape` | vector | Target shape |

**Returns:** MLX array

---

### `tile`

```clojure
(mx/tile a [2 3])
```

Tile an array by repeating it along each axis. Like `numpy.tile`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `reps` | vector | Number of repetitions along each axis |

**Returns:** MLX array

---

### `repeat-arr`

```clojure
(mx/repeat-arr a 3 0)
```

Repeat elements of an array along an axis. Like `numpy.repeat`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `repeats` | integer | Number of repetitions per element |
| `axis` | integer | Axis along which to repeat |

**Returns:** MLX array

---

### `split-arr`

```clojure
(mx/split-arr a 3)
(mx/split-arr a 3 1)
```

Split an array into sections. Like `numpy.split`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `sections` | integer | Number of equal sections |
| `axis` | integer (optional) | Axis to split along (default: 0) |

**Returns:** Clojure vector of MLX arrays

---

## Indexing

### `take-idx`

```clojure
(mx/take-idx a indices)
(mx/take-idx a indices 1)
```

Gather elements by index along an axis. Like `numpy.take`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `indices` | MLX int array | Index array |
| `axis` | integer (optional) | Axis to take along |

**Returns:** MLX array

---

### `idx`

```clojure
(mx/idx probs 2)
(mx/idx a 0 1)
```

Extract element at index `i` along axis (default 0). Convenience wrapper around `take-idx` that auto-promotes integers to MLX int32 scalars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `i` | integer / MLX array | Index |
| `axis` | integer (optional) | Axis (default: 0) |

**Returns:** MLX array

---

### `take-along-axis`

```clojure
(mx/take-along-axis a indices 1)
```

Gather elements along an axis using an index array (same shape as input). Like `numpy.take_along_axis`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `indices` | MLX int array | Index array |
| `axis` | integer | Axis to gather along |

**Returns:** MLX array

---

### `index`

```clojure
(mx/index a 0)
```

Index along axis 0. For 1-D arrays returns a scalar element. For 2-D arrays returns a row.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `i` | integer | Index |

**Returns:** MLX array

---

### `slice`

```clojure
(mx/slice a 2 5)
(mx/slice a 0 10 2)
```

Slice along axis 0. Returns elements `[start, stop)` with optional step.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `start` | integer | Start index (inclusive) |
| `stop` | integer | Stop index (exclusive) |
| `step` | integer (optional) | Step size |

**Returns:** MLX array

---

### `mat-get`

```clojure
(mx/mat-get a 1 2)
```

Get element `[i, j]` from a 2-D array. Returns a scalar MLX array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | 2-D input array |
| `i` | integer | Row index |
| `j` | integer | Column index |

**Returns:** MLX scalar array

---

### `searchsorted`

```clojure
(mx/searchsorted sorted-arr values)
(mx/searchsorted sorted-arr values :right)
```

Find insertion indices for values in a sorted 1-D array. Like `numpy.searchsorted`. Returns indices such that inserting `values` at those positions maintains sorted order.

| Parameter | Type | Description |
|-----------|------|-------------|
| `sorted-arr` | MLX array | Sorted 1-D array |
| `values` | MLX array | Values to insert |
| `side` | keyword (optional) | `:left` (default) or `:right` |

**Returns:** MLX integer array

---

## Linear Algebra

### `matmul`

```clojure
(mx/matmul a b)
```

Matrix multiplication. Like `numpy.matmul` or the `@` operator.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Left matrix |
| `b` | MLX array | Right matrix |

**Returns:** MLX array

---

### `inner`

```clojure
(mx/inner a b)
```

Inner (dot) product of two arrays. Like `numpy.inner`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First array |
| `b` | MLX array | Second array |

**Returns:** MLX array

---

### `outer`

```clojure
(mx/outer a b)
```

Outer product of two 1-D arrays. Like `numpy.outer`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | First 1-D array |
| `b` | MLX array | Second 1-D array |

**Returns:** MLX 2-D array

---

### `diag`

```clojure
(mx/diag a)
```

Extract diagonal from a matrix, or create a diagonal matrix from a 1-D array. Like `numpy.diag`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | 2-D matrix (extract) or 1-D array (create) |

**Returns:** MLX array

---

### `trace-mat`

```clojure
(mx/trace-mat a)
(mx/trace-mat a 1)
(mx/trace-mat a 0 0 1)
```

Matrix trace (sum of diagonal elements). Like `numpy.trace`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input matrix |
| `offset` | integer (optional) | Diagonal offset (default: 0) |
| `ax1` | integer (optional) | First axis (default: 0) |
| `ax2` | integer (optional) | Second axis (default: 1) |

**Returns:** MLX scalar array

---

### `einsum`

```clojure
(mx/einsum "ij,jk->ik" a b)
(mx/einsum "ii->" a)
```

Einstein summation convention. Like `numpy.einsum`. Supports arbitrary contraction patterns via subscript notation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `subscripts` | string | Einstein summation subscript string |
| `arrays` | MLX arrays | Input arrays (variadic) |

**Returns:** MLX array

---

### `cholesky`

```clojure
(mx/cholesky a)
```

Cholesky decomposition of a positive-definite matrix. Returns lower triangular factor L where A = LL^T. Runs on CPU stream.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Positive-definite matrix |

**Returns:** MLX lower triangular array

---

### `solve`

```clojure
(mx/solve a b)
```

Solve the linear system Ax = b. Runs on CPU stream. Like `numpy.linalg.solve`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Coefficient matrix |
| `b` | MLX array | Right-hand side |

**Returns:** MLX array (solution x)

---

### `solve-triangular`

```clojure
(mx/solve-triangular a b true)
```

Solve a triangular linear system. Runs on CPU stream.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Triangular matrix |
| `b` | MLX array | Right-hand side |
| `upper` | boolean | `true` for upper triangular, `false` for lower |

**Returns:** MLX array (solution)

---

### `inv`

```clojure
(mx/inv a)
```

Matrix inverse. Runs on CPU stream. Like `numpy.linalg.inv`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Square matrix |

**Returns:** MLX array

---

### `tri-inv`

```clojure
(mx/tri-inv L false)
(mx/tri-inv U true)
```

Triangular matrix inverse. More efficient than general `inv` for triangular matrices. Runs on CPU stream.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Triangular matrix |
| `upper` | boolean | `true` for upper triangular, `false` for lower |

**Returns:** MLX array

---

### `cholesky-inv`

```clojure
(mx/cholesky-inv L)
(mx/cholesky-inv U true)
```

Inverse of A from its Cholesky factor L (where A = LL^T). More efficient than computing `(inv A)` when you already have the Cholesky factor. Runs on CPU stream.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Cholesky factor |
| `upper` | boolean (optional) | `true` if upper triangular (default: `false`) |

**Returns:** MLX array (A^-1)

---

### `qr`

```clojure
(let [[Q R] (mx/qr a)] ...)
```

QR decomposition. Runs on CPU stream. Like `numpy.linalg.qr`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input matrix |

**Returns:** vector `[Q R]` of MLX arrays

---

### `svd`

```clojure
(let [[U S V] (mx/svd a)] ...)
```

Singular value decomposition. Runs on CPU stream. Like `numpy.linalg.svd`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input matrix |

**Returns:** vector `[U S V]` of MLX arrays

---

### `eigh`

```clojure
(let [[eigenvalues eigenvectors] (mx/eigh a)] ...)
```

Eigendecomposition of a symmetric (Hermitian) matrix. Runs on CPU stream. Like `numpy.linalg.eigh`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Symmetric matrix |

**Returns:** vector `[eigenvalues eigenvectors]` of MLX arrays

---

### `eigvalsh`

```clojure
(mx/eigvalsh a)
```

Eigenvalues of a symmetric (Hermitian) matrix. Runs on CPU stream. Like `numpy.linalg.eigvalsh`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Symmetric matrix |

**Returns:** MLX array of eigenvalues

---

### `logdet`

```clojure
(mx/logdet a)
```

Log-determinant of a positive-definite matrix. Computed via Cholesky: `2 * sum(log(diag(L)))`. More numerically stable than `log(det(A))`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Positive-definite matrix |

**Returns:** MLX scalar array

---

### `det`

```clojure
(mx/det a)
```

Determinant of a positive-definite matrix. Computed via Cholesky: `prod(diag(L))^2`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Positive-definite matrix |

**Returns:** MLX scalar array

---

### `norm`

```clojure
(mx/norm a)
(mx/norm a 2)
```

Matrix or vector norm. Like `numpy.linalg.norm`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `ord` | number (optional) | Norm order (default: Frobenius/L2) |

**Returns:** MLX scalar array

---

## Autograd

### `grad`

```clojure
(def grad-f (mx/grad (fn [x] (mx/multiply x x))))
(mx/item (grad-f (mx/scalar 3.0)))  ;; => 6.0

;; Differentiate with respect to specific arguments
(def grad-f (mx/grad f [0 2]))
```

Create a gradient function. The returned function computes the gradient of `f` with respect to its arguments. Internally tracks grad-depth so that `p/generate` can skip the L3 analytical path (which uses `volatile!` and breaks gradient flow).

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Scalar-valued function to differentiate |
| `argnums` | vector (optional) | Which argument indices to differentiate |

**Returns:** gradient function

---

### `value-and-grad`

```clojure
(def vg (mx/value-and-grad (fn [x] (mx/power x (mx/scalar 3)))))
(let [[val grad] (vg (mx/scalar 2.0))]
  (println "f(2)=" (mx/item val) "f'(2)=" (mx/item grad)))
;; f(2)= 8.0 f'(2)= 12.0
```

Create a function that returns both the value and gradient. More efficient than calling `f` and `(grad f)` separately. Tracks grad-depth.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Scalar-valued function to differentiate |
| `argnums` | vector (optional) | Which argument indices to differentiate |

**Returns:** function returning `[value gradient]`

---

### `jvp`

```clojure
(let [[primals tangents] (mx/jvp f [x] [dx])] ...)
```

Jacobian-vector product (forward-mode AD). Computes `f(primals)` and the directional derivative along `tangents`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Function to differentiate |
| `primals` | vector | Input values |
| `tangents` | vector | Tangent vectors (directions) |

**Returns:** vector `[primals tangents-out]`

---

### `vjp`

```clojure
(let [[primals cotangents] (mx/vjp f [x] [dy])] ...)
```

Vector-Jacobian product (reverse-mode AD). Computes `f(primals)` and propagates `cotangents` backward.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Function to differentiate |
| `primals` | vector | Input values |
| `cotangents` | vector | Cotangent vectors |

**Returns:** vector `[primals cotangents-out]`

---

### `stop-gradient`

```clojure
(mx/stop-gradient a)
```

Stop gradient flow through an array. The returned array has the same value but is treated as a constant during differentiation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |

**Returns:** MLX array (detached from gradient tape)

---

### `in-grad?`

```clojure
(mx/in-grad?)  ;; => false (or true inside grad/value-and-grad)
```

Returns `true` if currently executing inside an `mx/grad` or `mx/value-and-grad` scope. Used by `p/generate` to skip the L3 analytical path.

**Returns:** boolean

---

## Vectorization

### `vmap`

```clojure
(def batched-f (mx/vmap f))
(def batched-f (mx/vmap f [0 nil]))
(def batched-f (mx/vmap f [0] [0]))
```

Vectorize a function over a batch dimension. Like JAX's `vmap`. Automatically maps `f` over the leading axis of inputs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Function to vectorize |
| `in-axes` | vector (optional) | Which axis to map over per input (`nil` = broadcast) |
| `out-axes` | vector (optional) | Which axis the output batch dim appears on |

**Returns:** vectorized function

---

## Compilation

### `compile-fn`

```clojure
(def fast-f (mx/compile-fn f))
(def fast-f (mx/compile-fn f true))  ;; shapeless mode
```

Compile a function into a fused MLX computation graph for faster execution. The compiled function transparently recompiles if `compile-clear-cache!` has been called since compilation.

Set `shapeless?` to `true` if input shapes may vary between calls.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Function to compile |
| `shapeless?` | boolean (optional) | Allow varying shapes (default: `false`) |

**Returns:** compiled function (same interface as `f`)

---

### `compile-clear-cache!`

```clojure
(mx/compile-clear-cache!)
```

Clear all compiled function caches, releasing associated Metal resources. Safe to call at any time -- compiled functions transparently recompile on next use via the compile-generation counter.

**Returns:** nil

---

## Random Number Generation

Source: `src/genmlx/mlx/random.cljs`

Functional PRNG with key-based splitting. No global mutable state -- all randomness flows through explicit keys. Every sample consumes a key and produces a deterministic result.

### Key Management

#### `fresh-key`

```clojure
(rng/fresh-key)
(rng/fresh-key 42)
```

Create a fresh PRNG key. Without a seed, uses a random integer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | integer (optional) | Deterministic seed |

**Returns:** MLX array (PRNG key)

---

#### `split`

```clojure
(let [[k1 k2] (rng/split key)] ...)
```

Split a key into two independent sub-keys. The original key should not be reused after splitting.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |

**Returns:** vector `[k1 k2]`

---

#### `split-n`

```clojure
(let [keys (rng/split-n key 10)] ...)
```

Split a key into `n` independent sub-keys.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `n` | integer | Number of sub-keys |

**Returns:** vector of `n` PRNG keys

---

#### `ensure-key`

```clojure
(rng/ensure-key key)
```

Return `key` if non-nil, otherwise create a fresh random key.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array / nil | PRNG key or nil |

**Returns:** MLX array (PRNG key)

---

#### `key->seed`

```clojure
(rng/key->seed key)  ;; => 1234567
```

Derive a non-negative integer seed from a PRNG key. Combines both uint32 elements via XOR, then masks to 31 bits to ensure non-negative.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |

**Returns:** non-negative integer

---

#### `seed!`

```clojure
(rng/seed! key)
```

Seed the global MLX PRNG state from a key array. MLX random functions require this for deterministic output.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |

**Returns:** nil

---

#### `split-or-nils`

```clojure
(let [[k1 k2] (rng/split-or-nils key)] ...)
```

Split key into `[k1 k2]` if non-nil, otherwise return `[nil nil]`. Useful for optional key threading.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array / nil | PRNG key or nil |

**Returns:** vector `[k1 k2]` or `[nil nil]`

---

#### `split-n-or-nils`

```clojure
(rng/split-n-or-nils key 5)
```

Split key into `n` sub-keys if non-nil, otherwise return a vector of `n` nils.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array / nil | PRNG key or nil |
| `n` | integer | Number of sub-keys |

**Returns:** vector of keys or nils

---

### Sampling Functions

#### `normal`

```clojure
(rng/normal key [])       ;; scalar
(rng/normal key [100])    ;; 100 samples
(rng/normal key [3 4])    ;; 3x4 matrix
```

Sample from the standard normal distribution N(0, 1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `shape` | vector | Output shape (`[]` for scalar) |

**Returns:** MLX array

---

#### `uniform`

```clojure
(rng/uniform key [100])
```

Sample from the uniform distribution on [0, 1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `shape` | vector | Output shape |

**Returns:** MLX array

---

#### `bernoulli`

```clojure
(rng/bernoulli key 0.3 [100])
```

Sample from the Bernoulli distribution with probability `p`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `p` | number / MLX array | Success probability |
| `shape` | vector | Output shape |

**Returns:** MLX boolean array

---

#### `categorical`

```clojure
(rng/categorical key log-probs)
```

Sample from a categorical distribution parameterized by log-probabilities.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `logits` | MLX array | Unnormalized log-probabilities |

**Returns:** MLX integer array

---

#### `randint`

```clojure
(rng/randint key 0 10 [5])
```

Sample random integers uniformly from [lo, hi).

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `lo` | integer | Lower bound (inclusive) |
| `hi` | integer | Upper bound (exclusive) |
| `shape` | vector | Output shape |

**Returns:** MLX integer array

---

#### `gumbel`

```clojure
(rng/gumbel key [100])
```

Sample from the standard Gumbel distribution. Useful for the Gumbel-softmax trick.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `shape` | vector | Output shape |

**Returns:** MLX array

---

#### `laplace`

```clojure
(rng/laplace key [100])
```

Sample from the standard Laplace distribution (location=0, scale=1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `shape` | vector | Output shape |

**Returns:** MLX array

---

#### `truncated-normal`

```clojure
(rng/truncated-normal key -2.0 2.0 [100])
```

Sample from a truncated normal distribution. Values are clipped to [lower, upper].

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `lower` | number / MLX array | Lower bound |
| `upper` | number / MLX array | Upper bound |
| `shape` | vector | Output shape |

**Returns:** MLX array

---

#### `multivariate-normal`

```clojure
(rng/multivariate-normal key mean cov)
(rng/multivariate-normal key mean cov [10])
```

Sample from the multivariate normal distribution N(mean, cov). Runs on CPU stream because MLX's internal Cholesky requires it. For high dimensions (k > 10), manual Cholesky + matmul is faster.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `mean` | MLX array / vector | Mean vector of shape `[k]` |
| `cov` | MLX array / vector | Covariance matrix of shape `[k k]` (positive definite) |
| `shape` | vector (optional) | Batch shape prefix |

**Returns:** MLX array of shape `[...shape k]`

---

#### `permutation`

```clojure
(rng/permutation key 10)          ;; random permutation of [0..9]
(rng/permutation key arr 0)       ;; shuffle array along axis 0
```

Return a random permutation of integers [0, n) or shuffle an array along an axis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `n` / `arr` | integer / MLX array | Length or array to shuffle |
| `axis` | integer (optional) | Axis to shuffle along (for array input) |

**Returns:** MLX array

---

## Memory Management

### `materialize!`

```clojure
(mx/materialize! a b c)
```

Evaluate MLX arrays, materializing the computation graph. Use at inference/training loop boundaries to bound graph size. Alias for `eval!`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `arrs` | MLX arrays (variadic) | Arrays to evaluate |

**Returns:** nil

---

### `eval!`

```clojure
(mx/eval! a b c)
```

Evaluate (materialize) one or more lazy MLX arrays. Forces computation of the accumulated graph.

| Parameter | Type | Description |
|-----------|------|-------------|
| `arrs` | MLX arrays (variadic) | Arrays to evaluate |

**Returns:** nil

---

### `async-eval!`

```clojure
(mx/async-eval! a b)
```

Asynchronous evaluation. Triggers computation without blocking.

| Parameter | Type | Description |
|-----------|------|-------------|
| `arrays` | MLX arrays (variadic) | Arrays to evaluate asynchronously |

**Returns:** nil

---

### `tidy`

```clojure
(mx/tidy (fn [] (let [a (mx/ones [100 100])] (mx/sum a))))
```

Execute a function with automatic memory management. Arrays created inside the scope are freed unless returned. Use for bounding memory in tight loops.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Zero-argument function to execute |

**Returns:** return value of `f`

---

### `tidy-run`

```clojure
(mx/tidy-run
  (fn [] (compute-something))
  (fn [result] [(:array1 result) (:array2 result)]))
```

Run `f` inside `mx/tidy`, then call `collect-fn` on the result to identify which arrays to preserve. Materializes those arrays (detaching from computation graph intermediates), frees everything else. Automatically clears Metal cache when memory pressure is high.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Zero-argument function to execute |
| `collect-fn` | function | `(result) -> [arrays to preserve]` |

**Returns:** return value of `f`

---

### `tidy-scalar`

```clojure
(mx/tidy-scalar (fn [] (mx/sum (mx/ones [1000]))))  ;; => 1000.0
```

Run `f` inside `mx/tidy`, extract a JS number via `item`, return it. All intermediate MLX arrays are freed. The returned value is a plain JS number with no MLX references. Automatically clears Metal cache when memory pressure is high.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Zero-argument function returning an MLX scalar |

**Returns:** JS number

---

### `tidy-materialize`

```clojure
(mx/tidy-materialize (fn [] (mx/add (mx/ones [3]) (mx/ones [3]))))
```

Run `f` inside `mx/tidy`, materialize the result, return it. For simple cases where `f` returns a single MLX array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Zero-argument function returning an MLX array |

**Returns:** MLX array

---

### `in-tidy?`

```clojure
(mx/in-tidy?)  ;; => false
```

Returns `true` if currently executing inside an `mx/tidy` scope.

**Returns:** boolean

---

### `dispose!`

```clojure
(mx/dispose! a)
```

Explicitly free an array's memory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Array to free |

**Returns:** nil

---

### `with-resource-guard`

```clojure
(mx/with-resource-guard (fn [] (run-inference ...)))
```

Run `f` with cache-limit set to 0, preventing Metal buffer accumulation. Freed buffers are released immediately instead of being cached. Restores the previous cache limit on exit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | function | Zero-argument function to execute |

**Returns:** return value of `f`

---

### `force-gc!`

```clojure
(mx/force-gc!)
```

Force garbage collection and Metal buffer cleanup. Triggers JSC GC, sweeps dead arrays, clears cache, and clears compiled function caches. Compiled functions transparently recompile on next use, so this is safe to call at any time.

**Returns:** nil

---

### `auto-cleanup!`

```clojure
(mx/auto-cleanup!)
(mx/auto-cleanup! true)  ;; aggressive mode
```

Resource-pressure cleanup for hot paths. Checks Metal buffer count periodically (every 50 calls) and cleans up when pressure exceeds threshold (~200K buffers).

- **Lightweight** (default): sweep + clear only. Safe to call from anywhere including tight handler loops.
- **Aggressive** (`true`): also forces GC via `jsc-cleanup!`. Use only from leaf operations, never from tight handler loops.

| Parameter | Type | Description |
|-----------|------|-------------|
| `aggressive?` | boolean (optional) | Force GC before sweep (default: `false`) |

**Returns:** nil

---

### `sweep-dead-arrays!`

```clojure
(mx/sweep-dead-arrays!)
```

Synchronously free Metal buffers for arrays whose JS wrappers have been GC'd but whose deferred N-API finalizers have not run yet. No-op when called inside `mx/tidy`.

**Returns:** count of arrays swept (or nil inside tidy)

---

### Memory Monitoring

#### `get-active-memory`

```clojure
(mx/get-active-memory)  ;; => 1048576
```

Current active GPU memory in bytes.

**Returns:** integer (bytes)

---

#### `get-cache-memory`

```clojure
(mx/get-cache-memory)
```

Current Metal cache memory in bytes.

**Returns:** integer (bytes)

---

#### `get-peak-memory`

```clojure
(mx/get-peak-memory)
```

Peak memory usage since last reset.

**Returns:** integer (bytes)

---

#### `reset-peak-memory!`

```clojure
(mx/reset-peak-memory!)
```

Reset the peak memory counter to current active memory.

**Returns:** nil

---

#### `get-wrappers-count`

```clojure
(mx/get-wrappers-count)
```

Number of live JS wrapper objects for MLX arrays.

**Returns:** integer

---

#### `get-num-resources`

```clojure
(mx/get-num-resources)
```

Number of live Metal buffer allocations (active + cached). When this hits the resource limit (~499K), allocations fail.

**Returns:** integer

---

#### `get-resource-limit`

```clojure
(mx/get-resource-limit)
```

Maximum Metal buffer allocations before failure.

**Returns:** integer

---

#### `memory-report`

```clojure
(mx/memory-report)
;; => {:active-bytes 1048576
;;     :cache-bytes 524288
;;     :peak-bytes 2097152
;;     :wrappers 150
;;     :num-resources 300
;;     :resource-limit 499000}
```

Comprehensive memory usage report as a Clojure map.

**Returns:** map with keys `:active-bytes`, `:cache-bytes`, `:peak-bytes`, `:wrappers`, `:num-resources`, `:resource-limit`

---

### Memory Limits

#### `set-memory-limit!`

```clojure
(mx/set-memory-limit! (* 4 1024 1024 1024))  ;; 4 GB
```

Set the maximum GPU memory limit in bytes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | integer | Memory limit in bytes |

**Returns:** nil

---

#### `set-cache-limit!`

```clojure
(mx/set-cache-limit! (* 256 1024 1024))  ;; 256 MB
```

Set the Metal cache limit in bytes. GenMLX defaults to 256 MB.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | integer | Cache limit in bytes |

**Returns:** nil

---

#### `set-wired-limit!`

```clojure
(mx/set-wired-limit! n)
```

Set the wired memory limit in bytes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | integer | Wired memory limit in bytes |

**Returns:** nil

---

#### `clear-cache!`

```clojure
(mx/clear-cache!)
```

Clear the Metal memory cache, releasing cached buffers.

**Returns:** nil

---

### Metal Device

#### `metal-is-available?`

```clojure
(mx/metal-is-available?)  ;; => true
```

Check if a Metal GPU is available.

**Returns:** boolean

---

#### `metal-device-info`

```clojure
(mx/metal-device-info)
;; => {:architecture "applegpu_g14s"
;;     :device-name "Apple M2 Pro"
;;     :memory-size 32000000000
;;     :max-buffer-length 16000000000
;;     :max-recommended-working-set-size 22000000000
;;     :resource-limit 499000}
```

Return Metal device information as a Clojure map.

**Returns:** map with device details

---

## Sorting

### `argsort`

```clojure
(mx/argsort a)
(mx/argsort a 0)
```

Return indices that sort the array along the given axis (default: last axis). Like `numpy.argsort`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer (optional) | Axis to sort along |

**Returns:** MLX integer array of indices

---

### `sort-arr`

```clojure
(mx/sort-arr a)
(mx/sort-arr a 0)
```

Sort array along the given axis (default: last axis). Like `numpy.sort`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `axis` | integer (optional) | Axis to sort along |

**Returns:** MLX array (sorted)

---

### `topk`

```clojure
(mx/topk a 5)
```

Return the top-k largest values along the last axis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | MLX array | Input array |
| `k` | integer | Number of top values |

**Returns:** MLX array of top-k values

---

## Device and Constants

### Device

#### `default-device`

```clojure
(mx/default-device)
```

Get the current default compute device.

**Returns:** device object

---

#### `set-default-device!`

```clojure
(mx/set-default-device! mx/gpu)
```

Set the default compute device.

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | device | Device to use (`mx/cpu` or `mx/gpu`) |

**Returns:** nil

---

#### `mx/cpu`

CPU device constant.

#### `mx/gpu`

GPU device constant.

### Constants

| Constant | Value |
|----------|-------|
| `mx/pi` | 3.14159... |
| `mx/e-val` | 2.71828... |
| `mx/inf` | Positive infinity |
| `mx/nan` | Not a number |

---

## Utilities

### `training-step!`

```clojure
(mx/training-step! module optimizer vg-fn input1 input2)
```

One neural network training step: compute loss + gradients, update module parameters. Returns the loss as a JS number.

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | NN module | Neural network module |
| `optim` | optimizer | MLX optimizer |
| `vg-fn` | function | Value-and-grad function |
| `inputs` | any (variadic) | Training inputs |

**Returns:** JS number (loss value)

---

### `jsc-cleanup!`

```clojure
(mx/jsc-cleanup!)
```

Trigger JSC (Bun's JavaScript engine) garbage collection, microtask drain, and weak reference cleanup. Fires N-API destroy callbacks for dead MLX arrays, releasing Metal buffers. Safe to call from synchronous code. Only available when running under Bun.

**Returns:** nil
