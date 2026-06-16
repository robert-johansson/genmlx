# Cognitive Biases and Bounded Rationality (Why Softmax Isn't Enough)

> **Ports** agentmodels.org 5-biases-intro.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Expository: a Pac-Man player who systematically avoids power pellets near ghosts, or keeps returning to a low-value corridor, shows SYSTEMATIC (not random) deviation from optimal play — something softmax noise alone cannot explain. Motivates extending the agent model with biases (time inconsistency, myopia) so inverse planning recovers true preferences.

## What this chapter builds on

- **genmlx code:** exists — no-code narrative chapter (mirrors agentmodels 5-biases-intro which has no models). Sets up the biased-planner machinery in src/genmlx/agents/biased_planners.cljs covered in following chapters. Includes the 'two uses of decision models' table (authored).

## Live figures (planned)

- ch08-systematic-vs-noise.png — two maze paths: a noisy wobble vs a systematic dominated detour (static, authored from real rollouts)
- ch08-two-uses-table.png — Table 1 (practical problem-solving vs learning preferences) authored figure

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
