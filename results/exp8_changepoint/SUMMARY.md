# Experiment 8: Changepoint Detection

**Date:** 2026-03-05
**Model:** T=100, p_change=0.05, mu ~ N(0,5), sigma=1
**True changepoints:** [10 43 46 62 90 93] (6 total)
**Exact log P(y):** -178.4776

## Key Expressiveness Feature

```clojure
(let [cp (trace :cp (dist/bernoulli p-change))
      _ (mx/eval! cp)
      is-cp (> (mx/item cp) 0.5)
      new-mu (if is-cp
               (trace :mu_new (dist/gaussian 0 5))
               prev-mu)]
  (trace :y (dist/gaussian new-mu 1.0)))
```

The `if` on a sampled Bernoulli creates **data-dependent random structure**.

## Results

| Method | Particles | log-ML error | ESS | Time (ms) |
|--------|-----------|-------------|-----|----------|
| smc | 100 | 51.56 +/- 17.06 | 100.0 | 55993 |
| smc | 250 | 7.93 +/- 8.85 | 250.0 | 131209 |
| smc | 500 | 5.33 +/- 1.94 | 500.0 | 172645 |
| is | 1000 | 263.92 +/- 16.39 | 1.0 | 95148 |

## Interpretation

SMC via Unfold combinator exploits sequential structure. IS with prior proposals suffers weight degeneracy. The model demonstrates GenMLX's expressiveness: data-dependent structure is natural.
