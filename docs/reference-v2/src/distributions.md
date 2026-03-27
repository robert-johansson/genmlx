# Distributions

GenMLX provides 30+ built-in distributions, all sharing a single `Distribution` record type with behavior dispatched via open multimethods. New distributions can be added from any namespace using `defdist` or manual `defmethod`, without modifying core code.

**Source files:**
- `src/genmlx/dist.cljs` -- all built-in distribution definitions
- `src/genmlx/dist/core.cljs` -- `Distribution` record, multimethods, `mixture`, `product`, `map->dist`
- `src/genmlx/dist/macros.cljc` -- `defdist` macro

**Usage:**

```clojure
(require '[genmlx.dist :as dist]
         '[genmlx.dist.core :as dc])

;; Inside a gen body:
(trace :x (dist/gaussian 0 1))

;; Standalone:
(dist/gaussian 0 1)  ;; => Distribution record
```

Every distribution constructor returns a `Distribution` record that can be passed to `trace`, used with `p/simulate`, `p/generate`, `p/assess`, or called directly with the core multimethods.

---

## Distribution System

### Distribution record

All distributions share a single record type. Behavior is dispatched via open multimethods keyed by the `:type` keyword.

```clojure
(defrecord Distribution [type params])
```

The `Distribution` record implements the GFI protocols (`IGenerativeFunction`, `IGenerate`, `IAssess`, `IPropose`, `IProject`), so distributions are themselves generative functions. You can call `(p/simulate (dist/gaussian 0 1) [])` directly.

| Field | Type | Description |
|-------|------|-------------|
| `type` | keyword | Dispatch key (e.g. `:gaussian`, `:beta-dist`) |
| `params` | map | Named parameters (e.g. `{:mu <mlx> :sigma <mlx>}`) |

### Core multimethods

| Multimethod | Dispatch | Signature | Description |
|-------------|----------|-----------|-------------|
| `dc/dist-sample` | `:type` | `(dist-sample dist key)` | Draw one sample. Returns MLX scalar. |
| `dc/dist-log-prob` | `:type` | `(dist-log-prob dist value)` | Compute log-probability. Returns MLX scalar. |
| `dc/dist-sample-n` | `:type` | `(dist-sample-n dist key n)` | Draw `n` samples as `[n]`-shaped array. Used by vectorized inference. |
| `dc/dist-reparam` | `:type` | `(dist-reparam dist key)` | Reparameterized sample for gradient estimation. Returns MLX value. |
| `dc/dist-support` | `:type` | `(dist-support dist)` | Return support as a sequence of MLX values (for enumerable distributions). |
| `dc/dist-log-prob-support` | `:type` | `(dist-log-prob-support dist)` | Log-probabilities for all support values at once. Returns `[K]`-shaped tensor. |

### Public API wrappers

The `genmlx.dist` namespace provides convenience wrappers that delegate to the multimethods:

```clojure
(dist/sample d)           ;; or (dist/sample d key)
(dist/log-prob d value)
(dist/sample-reparam d key)
(dist/support d)
```

### `defdist`

```clojure
(defdist dist-name
  "Optional docstring."
  [params ...]
  (sample [key] body)
  (log-prob [v] body)
  ;; Optional:
  (reparam [key] body)
  (support [] body))
```

Define a new distribution type with a constructor function and multimethod implementations. Only `sample` and `log-prob` are required. Parameters are automatically wrapped with `mx/ensure-array` in the constructor.

**Example -- custom Laplace:**

```clojure
(defdist my-laplace [loc scale]
  (sample [key]
    (let [u (mx/subtract (rng/uniform key []) (mx/scalar 0.5))]
      (mx/add loc (mx/multiply scale
        (mx/multiply (mx/sign u)
                     (mx/log (mx/abs u)))))))
  (log-prob [v]
    (let [z (mx/abs (mx/divide (mx/subtract v loc) scale))]
      (mx/subtract (mx/negative z)
                   (mx/log (mx/multiply (mx/scalar 2) scale))))))
```

### Batch sampling

For vectorized inference, distributions should provide a `dc/dist-sample-n*` implementation. If not provided, a sequential fallback loops `n` times.

```clojure
(defmethod dc/dist-sample-n* :my-dist [d key n]
  (let [{:keys [loc scale]} (:params d)]
    (mx/add loc (mx/multiply scale (rng/normal key [n])))))
```

### `gamma-sample-n`

```clojure
(dist/gamma-sample-n shape-val rate key n)
```

Vectorized Marsaglia-Tsang gamma sampling. `shape-val` is a JS number, `rate` is an MLX scalar. Exposed for reuse by distributions built on Gamma (Beta, Dirichlet, Inverse-Gamma, Chi-squared).

---

## Continuous Distributions

### `gaussian`

```clojure
(dist/gaussian mu sigma)
```

\\( p(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) \\)

Gaussian (normal) distribution. Reparameterizable. Alias: `dist/normal`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | MLX scalar | Mean |
| `sigma` | MLX scalar | Standard deviation (must be positive) |

**Support:** \\( x \in (-\infty, +\infty) \\)

**Example:**
```clojure
(let [slope (trace :slope (dist/gaussian 0 10))
      noise (trace :noise (dist/gaussian 0 1))]
  (mx/add (mx/multiply slope x) noise))
```

### `normal`

Alias for `gaussian`.

```clojure
(dist/normal 0 1)  ;; identical to (dist/gaussian 0 1)
```

### `uniform`

```clojure
(dist/uniform lo hi)
```

\\( p(x \mid a, b) = \frac{1}{b - a} \quad \text{for } x \in [a, b] \\)

Continuous uniform distribution. Reparameterizable.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lo` | MLX scalar | Lower bound |
| `hi` | MLX scalar | Upper bound (must be greater than `lo`) |

**Support:** \\( x \in [a, b] \\)

**Example:**
```clojure
(trace :threshold (dist/uniform 0 1))
```

### `beta-dist`

```clojure
(dist/beta-dist alpha beta-param)
```

\\( p(x \mid \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)} \\)

Beta distribution. Sampled via Johnk's algorithm. Batch sampling uses the Gamma ratio method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | MLX scalar | Shape parameter \\(\alpha > 0\\) |
| `beta-param` | MLX scalar | Shape parameter \\(\beta > 0\\) |

**Support:** \\( x \in (0, 1) \\)

**Example:**
```clojure
(let [p (trace :p (dist/beta-dist 2 5))]
  (trace :outcome (dist/bernoulli p)))
```

### `gamma-dist`

```clojure
(dist/gamma-dist shape rate)
```

\\( p(x \mid k, \lambda) = \frac{\lambda^k}{\Gamma(k)} x^{k-1} e^{-\lambda x} \\)

Gamma distribution (shape-rate parameterization). Uses the Marsaglia-Tsang method with Ahrens-Dieter boost for \\(\alpha < 1\\).

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | MLX scalar | Shape parameter \\(k > 0\\) |
| `rate` | MLX scalar | Rate parameter \\(\lambda > 0\\) |

**Support:** \\( x \in (0, +\infty) \\)

**Example:**
```clojure
(trace :precision (dist/gamma-dist 2 1))
```

### `exponential`

```clojure
(dist/exponential rate)
```

\\( p(x \mid \lambda) = \lambda e^{-\lambda x} \\)

Exponential distribution. Reparameterizable (inverse CDF method).

| Parameter | Type | Description |
|-----------|------|-------------|
| `rate` | MLX scalar | Rate parameter \\(\lambda > 0\\) |

**Support:** \\( x \in [0, +\infty) \\)

**Example:**
```clojure
(trace :wait-time (dist/exponential 0.5))
```

### `log-normal`

```clojure
(dist/log-normal mu sigma)
```

\\( p(x \mid \mu, \sigma) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\!\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right) \\)

Log-normal distribution. Reparameterizable (exponential transform of Gaussian).

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | MLX scalar | Log-mean |
| `sigma` | MLX scalar | Log-standard-deviation (must be positive) |

**Support:** \\( x \in (0, +\infty) \\)

**Example:**
```clojure
(trace :income (dist/log-normal 10 1))
```

### `cauchy`

```clojure
(dist/cauchy loc scale)
```

\\( p(x \mid x_0, \gamma) = \frac{1}{\pi \gamma \left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]} \\)

Cauchy distribution. Reparameterizable (inverse CDF via tangent). Heavy tails -- no finite moments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `loc` | MLX scalar | Location \\(x_0\\) |
| `scale` | MLX scalar | Scale \\(\gamma > 0\\) |

**Support:** \\( x \in (-\infty, +\infty) \\)

**Example:**
```clojure
(trace :outlier (dist/cauchy 0 1))
```

### `student-t`

```clojure
(dist/student-t df loc scale)
```

\\( p(x \mid \nu, \mu, \sigma) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sigma\sqrt{\nu\pi}\;\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right)^{-(\nu+1)/2} \\)

Student's t-distribution with location and scale. Sampled via chi-squared ratio method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | MLX scalar | Degrees of freedom \\(\nu > 0\\) |
| `loc` | MLX scalar | Location \\(\mu\\) |
| `scale` | MLX scalar | Scale \\(\sigma > 0\\) |

**Support:** \\( x \in (-\infty, +\infty) \\)

**Example:**
```clojure
(trace :robust-obs (dist/student-t 4 0 1))
```

### `laplace`

```clojure
(dist/laplace loc scale)
```

\\( p(x \mid \mu, b) = \frac{1}{2b} \exp\!\left(-\frac{|x - \mu|}{b}\right) \\)

Laplace distribution. Reparameterizable (inverse CDF method).

| Parameter | Type | Description |
|-----------|------|-------------|
| `loc` | MLX scalar | Location \\(\mu\\) |
| `scale` | MLX scalar | Scale \\(b > 0\\) |

**Support:** \\( x \in (-\infty, +\infty) \\)

**Example:**
```clojure
(trace :sparse-weight (dist/laplace 0 0.1))
```

### `inv-gamma`

```clojure
(dist/inv-gamma shape scale)
```

\\( p(x \mid \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha-1} \exp\!\left(-\frac{\beta}{x}\right) \\)

Inverse-Gamma distribution. Sampled as the reciprocal of a Gamma variate.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | MLX scalar | Shape \\(\alpha > 0\\) |
| `scale` | MLX scalar | Scale \\(\beta > 0\\) |

**Support:** \\( x \in (0, +\infty) \\)

**Example:**
```clojure
;; Conjugate prior for Gaussian variance
(let [var (trace :var (dist/inv-gamma 3 1))]
  (trace :obs (dist/gaussian 0 (mx/sqrt var))))
```

### `truncated-normal`

```clojure
(dist/truncated-normal mu sigma lo hi)
```

\\( p(x \mid \mu, \sigma, a, b) = \frac{\phi\!\left(\frac{x-\mu}{\sigma}\right)}{\sigma\left[\Phi\!\left(\frac{b-\mu}{\sigma}\right) - \Phi\!\left(\frac{a-\mu}{\sigma}\right)\right]} \quad \text{for } x \in [a, b] \\)

Gaussian truncated to interval \\([a, b]\\). Reparameterizable. Uses `rng/truncated-normal` for sampling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | MLX scalar | Mean of the untruncated Gaussian |
| `sigma` | MLX scalar | Standard deviation (must be positive) |
| `lo` | MLX scalar | Lower truncation bound |
| `hi` | MLX scalar | Upper truncation bound (must be greater than `lo`) |

**Support:** \\( x \in [a, b] \\)

**Example:**
```clojure
;; Positive-only Gaussian
(trace :positive (dist/truncated-normal 5 2 0 100))
```

### `von-mises`

```clojure
(dist/von-mises mu kappa)
```

\\( p(\theta \mid \mu, \kappa) = \frac{e^{\kappa \cos(\theta - \mu)}}{2\pi I_0(\kappa)} \\)

Von Mises distribution on the circle \\([-\pi, \pi)\\). Sampled via Best's rejection algorithm. \\(I_0\\) is the modified Bessel function of the first kind.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | MLX scalar | Mean direction |
| `kappa` | MLX scalar | Concentration \\(\kappa > 0\\) (higher = more concentrated) |

**Support:** \\( \theta \in [-\pi, \pi) \\)

**Example:**
```clojure
(trace :heading (dist/von-mises 0 5))
```

### `wrapped-cauchy`

```clojure
(dist/wrapped-cauchy mu rho)
```

\\( p(\theta \mid \mu, \rho) = \frac{1 - \rho^2}{2\pi\left(1 - 2\rho\cos(\theta - \mu) + \rho^2\right)} \\)

Wrapped Cauchy distribution on \\([-\pi, \pi)\\). Closed-form density with inverse CDF sampling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | MLX scalar | Mean direction |
| `rho` | MLX scalar | Concentration \\(\rho \in (0, 1)\\) |

**Support:** \\( \theta \in [-\pi, \pi) \\)

**Example:**
```clojure
(trace :wind-dir (dist/wrapped-cauchy 0 0.8))
```

### `wrapped-normal`

```clojure
(dist/wrapped-normal mu sigma)
```

\\( p(\theta \mid \mu, \sigma) = \sum_{k=-\infty}^{\infty} \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(\theta + 2\pi k - \mu)^2}{2\sigma^2}\right) \\)

Gaussian wrapped onto the circle \\([-\pi, \pi)\\). Log-probability computed via truncated series sum (\\(k = -3\\) to \\(3\\)).

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | MLX scalar | Mean direction |
| `sigma` | MLX scalar | Standard deviation (must be positive) |

**Support:** \\( \theta \in [-\pi, \pi) \\)

**Example:**
```clojure
(trace :angle (dist/wrapped-normal 0 0.5))
```

---

## Discrete Distributions

### `bernoulli`

```clojure
(dist/bernoulli p)
```

\\( P(x = 1) = p, \quad P(x = 0) = 1 - p \\)

Bernoulli distribution. Enumerable (support: \\(\\{0, 1\\}\\)). Alias: `dist/flip`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | MLX scalar | Probability of success \\(p \in [0, 1]\\) |

**Support:** \\( x \in \\{0, 1\\} \\)

**Example:**
```clojure
(let [biased (trace :coin (dist/bernoulli 0.7))]
  (if (pos? (mx/item biased)) :heads :tails))
```

### `flip`

Alias for `bernoulli`.

```clojure
(dist/flip 0.5)  ;; identical to (dist/bernoulli 0.5)
```

### `categorical`

```clojure
(dist/categorical logits)
```

\\( P(x = k) = \frac{e^{\ell_k}}{\sum_j e^{\ell_j}} \\)

Categorical distribution over \\(K\\) categories, parameterized by unnormalized log-probabilities (logits). Sampling uses the Gumbel-max trick. Enumerable.

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | MLX array `[K]` | Unnormalized log-probabilities |

**Support:** \\( x \in \\{0, 1, \ldots, K-1\\} \\) (MLX int32)

**Example:**
```clojure
(trace :category (dist/categorical (mx/array [0.0 1.0 2.0])))
```

### `categorical-weights`

```clojure
(dist/categorical-weights weights)
```

Categorical distribution from unnormalized weights (not log-space). Handles zero weights safely -- no `log(0)`, no NaN gradients. Zero weights get a large negative logit with zero gradient.

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | MLX array `[K]` | Non-negative weights (may contain zeros) |

**Example:**
```clojure
(dist/categorical-weights (mx/array [1.0 0.0 3.0]))
```

### `weighted`

```clojure
(dist/weighted [w1 w2 ...])
```

Categorical distribution from a Clojure vector of weights. Each weight may be a number or an MLX array. Handles scalar promotion, stacking, and log transform automatically. Use this to write model code at the distribution layer without dropping to raw MLX operations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | Clojure vector | Numbers or MLX arrays |

**Example:**
```clojure
;; Clean distribution-layer code:
(trace :eval (dist/weighted [1.0 1.5 w]))

;; Instead of noisy MLX-layer code:
(trace :eval (dist/categorical
  (mx/log (mx/stack #js [(mx/scalar 1.0) (mx/scalar 1.5) w]))))
```

### `poisson`

```clojure
(dist/poisson rate)
```

\\( P(x = k \mid \lambda) = \frac{\lambda^k e^{-\lambda}}{k!} \\)

Poisson distribution. Sampled via Knuth's algorithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rate` | MLX scalar | Rate \\(\lambda > 0\\) |

**Support:** \\( x \in \\{0, 1, 2, \ldots\\} \\)

**Example:**
```clojure
(trace :count (dist/poisson 3.5))
```

### `geometric`

```clojure
(dist/geometric p)
```

\\( P(x = k \mid p) = (1 - p)^k \, p \\)

Number of failures before the first success. Sampled via inverse CDF. Enumerable (support capped at the 0.999 quantile).

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | MLX scalar | Success probability \\(p \in (0, 1]\\) |

**Support:** \\( x \in \\{0, 1, 2, \ldots\\} \\)

**Example:**
```clojure
(trace :attempts (dist/geometric 0.3))
```

### `binomial`

```clojure
(dist/binomial n p)
```

\\( P(x = k \mid n, p) = \binom{n}{k} p^k (1-p)^{n-k} \\)

Number of successes in \\(n\\) independent Bernoulli trials. Enumerable.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | MLX scalar | Number of trials |
| `p` | MLX scalar | Success probability \\(p \in [0, 1]\\) |

**Support:** \\( x \in \\{0, 1, \ldots, n\\} \\)

**Example:**
```clojure
(trace :successes (dist/binomial 10 0.4))
```

### `neg-binomial`

```clojure
(dist/neg-binomial r p)
```

\\( P(x = k \mid r, p) = \binom{k + r - 1}{k} p^r (1-p)^k \\)

Negative binomial (Polya) distribution: number of failures before \\(r\\) successes. Sampled via the Gamma-Poisson mixture method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `r` | MLX scalar | Number of successes \\(r > 0\\) |
| `p` | MLX scalar | Success probability \\(p \in [0, 1]\\) |

**Support:** \\( x \in \\{0, 1, 2, \ldots\\} \\)

**Example:**
```clojure
(trace :failures (dist/neg-binomial 5 0.6))
```

### `discrete-uniform`

```clojure
(dist/discrete-uniform lo hi)
```

\\( P(x = k) = \frac{1}{b - a + 1} \quad \text{for } k \in \\{a, a+1, \ldots, b\\} \\)

Discrete uniform over integers. Enumerable.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lo` | MLX scalar | Lower bound (integer) |
| `hi` | MLX scalar | Upper bound (integer, must be greater than `lo`) |

**Support:** \\( x \in \\{a, a+1, \ldots, b\\} \\) (MLX int32)

**Example:**
```clojure
(trace :die (dist/discrete-uniform 1 6))
```

---

## Multivariate Distributions

### `multivariate-normal`

```clojure
(dist/multivariate-normal mean-vec cov-matrix)
```

\\( p(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right) \\)

Multivariate normal distribution. Cholesky decomposition and \\(L^{-1}\\) are computed once at construction time. Reparameterizable.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mean-vec` | MLX array `[k]` | Mean vector |
| `cov-matrix` | MLX array `[k, k]` | Positive definite covariance matrix |

**Support:** \\( \mathbf{x} \in \mathbb{R}^k \\)

**Example:**
```clojure
(let [mu (mx/array [0 0])
      cov (mx/array [[1 0.5] [0.5 2]])]
  (trace :latent (dist/multivariate-normal mu cov)))
```

### `dirichlet`

```clojure
(dist/dirichlet alpha)
```

\\( p(\mathbf{x} \mid \boldsymbol{\alpha}) = \frac{\Gamma\!\left(\sum_i \alpha_i\right)}{\prod_i \Gamma(\alpha_i)} \prod_{i=1}^{k} x_i^{\alpha_i - 1} \\)

Dirichlet distribution. Sampled via independent Gamma variates followed by normalization. Batch sampling uses `gamma-sample-n`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | MLX array `[k]` | Concentration parameters \\(\alpha_i > 0\\) |

**Support:** \\( \mathbf{x} \in \Delta^{k-1} \\) (the probability simplex: \\(x_i > 0\\), \\(\sum x_i = 1\\))

**Example:**
```clojure
(let [probs (trace :probs (dist/dirichlet (mx/array [1 1 1])))]
  (trace :choice (dist/categorical (mx/log probs))))
```

### `wishart`

```clojure
(dist/wishart df scale-matrix)
```

\\( p(X \mid \nu, V) = \frac{|X|^{(\nu-k-1)/2} \exp\!\left(-\frac{1}{2}\text{tr}(V^{-1}X)\right)}{2^{\nu k/2} |V|^{\nu/2} \Gamma_k(\nu/2)} \\)

Wishart distribution over positive definite matrices. Sampled via Bartlett decomposition.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | number | Degrees of freedom \\(\nu \geq k\\) |
| `scale-matrix` | MLX array `[k, k]` | Positive definite scale matrix \\(V\\) |

**Support:** \\( X \in \mathbb{S}_{++}^k \\) (\\(k \times k\\) positive definite matrices)

**Example:**
```clojure
(let [V (mx/array [[1 0] [0 1]])]
  (trace :precision (dist/wishart 5 V)))
```

### `inv-wishart`

```clojure
(dist/inv-wishart df scale-matrix)
```

\\( p(X \mid \nu, \Psi) = \frac{|\Psi|^{\nu/2} |X|^{-(\nu+k+1)/2} \exp\!\left(-\frac{1}{2}\text{tr}(\Psi X^{-1})\right)}{2^{\nu k/2} \Gamma_k(\nu/2)} \\)

Inverse Wishart distribution. Sampled by drawing \\(W \sim \text{Wishart}(\nu, \Psi^{-1})\\) and returning \\(W^{-1}\\).

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | number | Degrees of freedom \\(\nu > k + 1\\) |
| `scale-matrix` | MLX array `[k, k]` | Positive definite scale matrix \\(\Psi\\) |

**Support:** \\( X \in \mathbb{S}_{++}^k \\) (\\(k \times k\\) positive definite matrices)

**Example:**
```clojure
;; Conjugate prior for MVN covariance
(let [Psi (mx/array [[1 0] [0 1]])
      cov (trace :cov (dist/inv-wishart 5 Psi))]
  (trace :obs (dist/multivariate-normal mu cov)))
```

---

## Special Distributions

### `delta`

```clojure
(dist/delta v)
```

\\( P(x = v) = 1, \quad \log p(v) = 0 \\)

Point mass (Dirac delta) at value \\(v\\). Score is 0 when the value matches, \\(-\infty\\) otherwise. Enumerable (support is `[v]`). Used for deterministic values in traces.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | MLX value | The point mass location |

**Example:**
```clojure
(trace :fixed (dist/delta (mx/scalar 42)))
```

### `broadcasted-normal`

```clojure
(dist/broadcasted-normal mu sigma)
```

\\( p(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\sigma}) = \prod_i \frac{1}{\sigma_i\sqrt{2\pi}} \exp\!\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right) \\)

Independent element-wise normal distribution. `mu` and `sigma` can be MLX arrays of any matching shape. Log-probability is the sum over all elements. Reparameterizable.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | MLX array | Mean (any shape) |
| `sigma` | MLX array | Standard deviation (same shape as `mu`, positive) |

**Example:**
```clojure
(let [mu (mx/array [1 2 3])
      sigma (mx/array [0.1 0.2 0.3])]
  (trace :obs (dist/broadcasted-normal mu sigma)))
```

### `gaussian-vec`

```clojure
(dist/gaussian-vec mu sigma)
```

\\( p(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\sigma}) = \prod_i \frac{1}{\sigma_i\sqrt{2\pi}} \exp\!\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right) \\)

Vector of independent Gaussians with log-prob summed over the last axis. For a `[D]`-shaped value, returns a scalar log-prob. For an `[N, D]`-shaped value, returns `[N]`-shaped log-probs. Ideal for vectorized inference with one trace site per latent vector. Reparameterizable.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | MLX array `[D]` | Per-element means |
| `sigma` | MLX array (scalar or `[D]`) | Standard deviation(s) |

**Example:**
```clojure
(let [mu (mx/zeros [10])
      sigma (mx/scalar 1.0)]
  (trace :latent (dist/gaussian-vec mu sigma)))
```

### `beta-uniform-mixture`

```clojure
(dist/beta-uniform-mixture theta alpha beta-param)
```

\\( p(x \mid \theta, \alpha, \beta) = \theta \cdot \text{Beta}(x; \alpha, \beta) + (1 - \theta) \cdot \text{Uniform}(x; 0, 1) \\)

Mixture of \\(\text{Beta}(\alpha, \beta)\\) with probability \\(\theta\\) and \\(\text{Uniform}(0, 1)\\) with probability \\(1 - \theta\\). Built using `dc/mixture`. Common prior for bounded parameters.

| Parameter | Type | Description |
|-----------|------|-------------|
| `theta` | number | Mixing weight for Beta component \\(\theta \in (0, 1)\\) |
| `alpha` | MLX scalar | Beta shape \\(\alpha > 0\\) |
| `beta-param` | MLX scalar | Beta shape \\(\beta > 0\\) |

**Support:** \\( x \in (0, 1) \\)

**Example:**
```clojure
(trace :rate (dist/beta-uniform-mixture 0.9 2 5))
```

### `piecewise-uniform`

```clojure
(dist/piecewise-uniform bounds probs)
```

\\( p(x) = \frac{p_i}{w_i \sum_j p_j} \quad \text{for } x \in [b_i, b_{i+1}) \\)

Piecewise uniform distribution over bins defined by sorted boundary points. Bin selection uses categorical sampling; uniform sampling within the selected bin.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bounds` | MLX array `[N+1]` | Sorted boundary points |
| `probs` | MLX array `[N]` | Unnormalized bin probabilities |

**Example:**
```clojure
(let [bounds (mx/array [0 1 3 10])
      probs  (mx/array [0.5 0.3 0.2])]
  (trace :x (dist/piecewise-uniform bounds probs)))
```

---

## Constructors and Combinators

### `iid`

```clojure
(dist/iid base-dist t)
```

IID (independent and identically distributed) constructor. Wraps any base distribution to sample \\(T\\) independent values as a single trace site.

- **Scalar mode:** sample returns `[T]`-shaped tensor, log-prob returns scalar (sum).
- **Batch mode:** `sample-n(N)` returns `[N, T]`-shaped tensor, log-prob returns `[N]`.

When base-dist parameters are already batched (e.g., means `[N, T]` from batched latent variables), `iid` detects this and handles correctly.

| Parameter | Type | Description |
|-----------|------|-------------|
| `base-dist` | Distribution | Any distribution record |
| `t` | integer | Number of independent samples |

**Example:**
```clojure
;; Sample 5 independent Gaussian values as one trace site
(trace :observations (dist/iid (dist/gaussian mu sigma) 5))
```

### `iid-gaussian`

```clojure
(dist/iid-gaussian mu sigma t)
```

Optimized IID Gaussian. Generates all \\(T\\) samples in a single MLX noise draw instead of \\(T\\) separate draws. `mu` and `sigma` can be scalar (shared across elements) or `[T]`-shaped (per-element parameters).

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | MLX scalar or `[T]` | Mean(s) |
| `sigma` | MLX scalar or `[T]` | Standard deviation(s) (must be positive) |
| `t` | integer | Number of independent samples |

**Example:**
```clojure
;; 100 i.i.d. standard normal samples in one op
(trace :noise (dist/iid-gaussian 0 1 100))

;; Per-element means
(trace :obs (dist/iid-gaussian (mx/array [1 2 3]) 0.5 3))
```

### `dc/mixture`

```clojure
(dc/mixture components log-weights)
```

Create a mixture distribution from component distributions and unnormalized log mixing weights. Sampling draws from all components, then selects by categorical index (stays in the MLX graph -- vectorizable and differentiable).

| Parameter | Type | Description |
|-----------|------|-------------|
| `components` | vector of Distributions | Component distributions |
| `log-weights` | MLX array `[K]` | Unnormalized log mixing weights |

**Example:**
```clojure
(let [mix (dc/mixture [(dist/gaussian -2 1) (dist/gaussian 2 1)]
                       (mx/array [0.0 0.0]))]  ;; equal weights
  (trace :x mix))
```

### `dc/product`

```clojure
(dc/product components)
```

Create a product (joint independent) distribution from a vector or map of component distributions. Sampling is independent; log-prob is the sum of component log-probs.

**Vector form** -- returns a vector of MLX values:

```clojure
(dc/product [(dist/gaussian 0 1) (dist/uniform 0 1)])
```

**Map form** -- returns a map of named MLX values:

```clojure
(dc/product {:loc (dist/gaussian 0 10)
             :scale (dist/exponential 1)})
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `components` | vector or map | Component distributions |

### `dc/map->dist`

```clojure
(dc/map->dist {:type :my-dist
               :sample (fn [key] ...)
               :log-prob (fn [value] ...)})
```

Create a `Distribution` from a plain map with function-valued keys. Registers multimethod implementations automatically. This is an alternative to `defdist` for programmatic distribution construction.

| Key | Required | Description |
|-----|----------|-------------|
| `:type` | no | Keyword identifier (auto-generated if omitted) |
| `:sample` | yes | `(fn [key] -> MLX-value)` |
| `:log-prob` | yes | `(fn [value] -> MLX-scalar)` |
| `:reparam` | no | `(fn [key] -> MLX-value)` reparameterized sample |
| `:support` | no | `(fn [] -> seq)` enumerable support |
| `:sample-n` | no | `(fn [key n] -> MLX-array)` batch sampling |

**Example:**
```clojure
(def my-dist
  (dc/map->dist
    {:type :spike-and-slab
     :sample (fn [key]
               (let [[k1 k2] (rng/split key)
                     spike? (< (mx/item (rng/uniform k1 [])) 0.5)]
                 (if spike?
                   (mx/scalar 0)
                   (mx/add (mx/scalar 0) (mx/multiply (mx/scalar 5) (rng/normal k2 []))))))
     :log-prob (fn [v]
                 (mx/logsumexp
                   (mx/array [(+ (js/Math.log 0.5)
                                 (mx/item (dc/dist-log-prob (dist/delta (mx/scalar 0)) v)))
                              (+ (js/Math.log 0.5)
                                 (mx/item (dc/dist-log-prob (dist/gaussian 0 5) v)))])))}))
```
