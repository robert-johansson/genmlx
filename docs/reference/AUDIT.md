# API Reference Audit — 2026-03-27

## Coverage Summary

~200 functions documented out of ~400+ public functions. ~50% coverage.
No stale/removed functions found — everything documented still exists.

## docs/index.html Issues

- [ ] Line count: says "~10,800" but actual is ~25,250
- [ ] Quick Start section: install instructions don't match README.md
  - Missing: `git submodule update --init --recursive node-mlx`
  - Missing: `cd node-mlx && npm install && npm run build && cd ..`
  - Missing: `bun install` (currently says `npm install`)
- [ ] Feature text: says "Purely Functional" — should say "Functional Core" (already fixed, verify)

## Existing Reference Pages — Missing Functions

### gfi.html (protocols.cljs)
- [ ] Add `IBatchedSplice` protocol
- [ ] Add `IUpdateWithDiffs` protocol
- [ ] Add `IHasArgumentGrads` protocol

### distributions.html (dist.cljs, dist/core.cljs)
- [ ] Add `half-normal` distribution
- [ ] Add `logistic` distribution
- [ ] Add `pareto` distribution
- [ ] Add `gumbel` distribution
- [ ] Add `von-mises` distribution (if present)
- [ ] Add `truncated-normal` distribution
- [ ] Add `beta-uniform-mixture` distribution
- [ ] Add `wishart` and `inv-wishart` distributions
- [ ] Add `iid` and `iid-gaussian` constructors
- [ ] Add `flip` (alias for bernoulli)
- [ ] Add `weighted` (alias for categorical)
- [ ] Add `dc/map->dist` function
- [ ] Add `dc/mixture` function
- [ ] Add `dc/product` function

### choicemap.html (choicemap.cljs)
- [ ] Add `from-flat-map` function

### trace.html (trace.cljs)
- [ ] Document trace metadata (::splice-scores, ::element-scores)

### combinators.html (combinators.cljs, vmap.cljs)
- [ ] Add `unfold-empty-trace` function
- [ ] Add `unfold-extend` function
- [ ] Add `vectorized-switch` function
- [ ] Add `mask-combinator` function
- [ ] Add `contramap-gf` function
- [ ] Add `map-retval` function
- [ ] Add `dimap` function
- [ ] Add `vmap/vmap-gf` function
- [ ] Add `vmap/repeat-gf` function

### inference.html (inference/*.cljs)
- [ ] Add `mcmc/mala` function
- [ ] Add `mcmc/hmc` function (with adaptive options)
- [ ] Add `mcmc/nuts` function
- [ ] Add `mcmc/gibbs` function
- [ ] Add `mcmc/elliptical-slice` function
- [ ] Add `mcmc/map-optimize` function
- [ ] Add `mcmc/compiled-mh` function
- [ ] Add `mcmc/involutive-mh` function
- [ ] Add `smc/csmc` function
- [ ] Add `smc/vsmc` function
- [ ] Add `smc/smc-unfold` function
- [ ] Add `vi/compiled-vi` function
- [ ] Add `vi/vi-from-model` function
- [ ] Add `vi/compiled-vi-from-model` function
- [ ] Add `vi/programmable-vi` function
- [ ] Add VI objectives: `elbo-objective`, `iwelbo-objective`, `vimco-objective`
- [ ] Add `smcp3/smcp3` function
- [ ] Add `importance/tidy-importance-sampling` function

### mlx.html (mlx.cljs)
- [ ] Add `astype` function
- [ ] Add `argsort` function
- [ ] Add `sort-arr` function
- [ ] Add `searchsorted` function
- [ ] Add `topk` function
- [ ] Add `logcumsumexp` function
- [ ] Add `meshgrid` function
- [ ] Add `tile` function
- [ ] Add `repeat-arr` function
- [ ] Add `split-arr` function
- [ ] Add `take-along-axis` function
- [ ] Add `mat-get` function
- [ ] Add `einsum` function
- [ ] Add `logdet` function
- [ ] Add `det` function
- [ ] Add `cholesky-inv` function
- [ ] Add `tri-inv` function
- [ ] Add `eigh` function
- [ ] Add `eigvalsh` function
- [ ] Add `arccos` function
- [ ] Add `erf` / `erfinv` functions
- [ ] Add `lgamma` / `digamma` functions
- [ ] Add `bessel-i0e` / `bessel-i1e` functions
- [ ] Add `nan-to-num` function
- [ ] Add `eq?` / `neq?` / `gt?` / `lt?` / `and*` / `or*` model helpers
- [ ] Add memory management section: `memory-report`, `get-num-resources`, `get-resource-limit`, `metal-device-info`
- [ ] Add `compile-fn` function
- [ ] Add `compile-clear-cache!` function
- [ ] Add `tidy-run` / `tidy-scalar` / `tidy-materialize` functions
- [ ] Add `with-resource-guard` function
- [ ] Add `force-gc!` function
- [ ] Add `auto-cleanup!` function

## Missing Reference Pages (need new HTML files)

### selection.html (selection.cljs)
- [ ] Create page
- [ ] `ISelection` protocol: `selected?`, `get-subselection`
- [ ] `all`, `none` singletons
- [ ] `select` function
- [ ] `from-set` function
- [ ] `hierarchical` function
- [ ] `complement-sel` function

### kernel.html (inference/kernel.cljs)
- [ ] Create page
- [ ] `mh-kernel`, `update-kernel` constructors
- [ ] `chain`, `repeat-kernel`, `cycle-kernels`, `mix-kernels`, `seed` composition
- [ ] `with-reversal`, `symmetric-kernel`, `reversal`, `reversed`
- [ ] `run-kernel`, `collect-samples`
- [ ] `random-walk`, `prior`, `gibbs`, `proposal` DSL constructors

### learning.html (learning.cljs)
- [ ] Create page
- [ ] `make-param-store`, `get-param`, `set-param`, `update-params`
- [ ] `params->array`, `array->params`, `param-names`
- [ ] `sgd-step`, `adam-init`, `adam-step`
- [ ] `train` generic loop
- [ ] `simulate-with-params`, `generate-with-params`
- [ ] `make-param-loss-fn`
- [ ] `wake-phase-loss`, `sleep-phase-loss`, `wake-sleep`

### gradients.html (gradients.cljs)
- [ ] Create page
- [ ] `choice-gradients`
- [ ] `score-gradient`

### diagnostics.html (inference/diagnostics.cljs)
- [ ] Create page
- [ ] `ess`, `r-hat`
- [ ] `sample-mean`, `sample-std`, `sample-quantiles`
- [ ] `summarize`

### contracts.html (contracts.cljs)
- [ ] Create page
- [ ] `verify-gfi-contracts`
- [ ] List all 11 contracts with descriptions

### verify.html (verify.cljs)
- [ ] Create page
- [ ] `validate-gen-fn`

### edit.html (edit.cljs)
- [ ] Create page
- [ ] `ConstraintEdit`, `SelectionEdit`, `ProposalEdit` records
- [ ] `constraint-edit`, `selection-edit`, `proposal-edit` constructors
- [ ] `edit-dispatch` function

### nn.html (nn.cljs)
- [ ] Create page
- [ ] `sequential`, `linear`, activation functions
- [ ] `nn->gen-fn`
- [ ] MLX module training utilities

### serialize.html (serialize.cljs)
- [ ] Create page
- [ ] `save-trace`, `load-trace`
- [ ] `save-choices`, `load-choices`
- [ ] All serialization functions

### adev.html (inference/adev.cljs)
- [ ] Create page
- [ ] `adev-execute`, `adev-surrogate`, `adev-gradient`
- [ ] `adev-optimize`, `vadev-gradient`
- [ ] `compiled-adev-optimize`

### fit.html (fit.cljs, method_selection.cljs)
- [ ] Create page
- [ ] `fit` function
- [ ] `select-method` function
- [ ] Method selection decision tree

## Lower Priority (internal modules)

### dynamic.html (dynamic.cljs)
- [ ] `make-gen-fn`, `with-key`, `auto-key`
- [ ] `vsimulate`, `vgenerate`, `vupdate`, `vregenerate`

### handler.html (handler.cljs)
- [ ] Document transitions for advanced users
- [ ] `simulate-transition`, `generate-transition`, etc.

### schema.html (schema.cljs)
- [ ] Schema structure documentation
- [ ] How to read `:trace-sites`, `:static?`, etc.

### compiled.html (compiled.cljs, compiled_gen.cljs)
- [ ] Compiled path documentation for advanced users

## Priority Order

1. **CRITICAL**: Fix docs/index.html (line count, install instructions)
2. **HIGH**: Create selection.html, kernel.html, learning.html, diagnostics.html
3. **HIGH**: Complete distributions.html, inference.html, mlx.html
4. **MEDIUM**: Create gradients.html, contracts.html, verify.html, edit.html
5. **MEDIUM**: Create nn.html, serialize.html, adev.html, fit.html
6. **LOW**: Complete dynamic.html, handler.html, schema.html, compiled.html
