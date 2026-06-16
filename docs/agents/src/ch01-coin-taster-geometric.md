# Probabilistic Programming in Five Minutes

> **Ports** agentmodels.org 1-introduction.md (taster snippets), 2-webppl.md (language primer).

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

coin() = does a ghost turn left or right at a junction (flip). geometric(p) = how many cells Pac-Man travels before a random ghost blocks it (recursive sampler). categorical = which of 4 directions Pac-Man picks. condition = 'given Pac-Man reached the pellet, what path did it take?' (the two-heads conditioning example reframed as a surviving trajectory).

## What this chapter builds on

- **genmlx code:** exists — test/genmlx/agentmodels_ch01_intro_test.cljs (geometric P(n=k)=0.5^(k+1), E[n]=1) and test/genmlx/agentmodels_ch02_webppl_test.cljs (ERP, multivariateGaussian, twoHeads/moreThanTwoHeads conditioning P(first=H|>=2)=0.75, forward positionDist). Examples: examples/agentmodels/ch01_intro.cljs + ch02_webppl.cljs. dist/* primitives + p/generate/condition. No new code.

## Live figures (planned)

- ch01-geometric-bars.png — posterior bar chart over trajectory length (presentation/dist->bars rendered)
- ch01-conditioning.png — reuse template figures/conditioning.png concept, redrawn as surviving-vs-all maze paths
- ch01-position-gaussian.png — 2D Gaussian scatter of a noisy ghost position estimate

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
