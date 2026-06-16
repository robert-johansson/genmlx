# Joint Inference II: Was That Detour Noise, Bias, or Taste?

> **Ports** agentmodels.org 5e-joint-inference.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

A Pac-Man that passes near a power pellet but eats a regular pellet first could be (a) noisy, (b) a Naive hyperbolic discounter tempted by the immediate pellet, or (c) genuinely preferring regular pellets. Joint inference over all three from repeated plays: observing the same 'mistake' three times collapses the noise explanation. Expanding the preference space (DonutN != DonutS, positive time cost) lets alternative tastes compete with discounting. Uses the full [immediate, delayed] restaurant-twin geometry.

## What this chapter builds on

- **genmlx code:** exists — full joint model (discount in {0,1}, sophisticatedOrNaive, vegMinusDonut, donutTempting summary stats), conditioning on Naive path once vs three times, two-donut and positive-timeCost competing explanations. Tests: test/genmlx/agentmodels_5e_joint_test.cljs ([imm,del] 6×8 grid 48+4 twins=52 states, 2-step stay delivers delta(d)*imm + delta(d+1)*del disU formula, joint k+utility inference) + agentmodels_biased_inverse_test.cljs (naive EU(a0)=8/3 tie, soph=2.5, pi_naive(a0)=0.5 vs pi_soph(a0)=0, prior decomposition via p/assess on {:bias} alone, posterior concentrates on :sophisticated). Examples restaurant_joint_inference.cljs, biased_inverse.cljs.

## Live figures (planned)

- ch13-three-explanations.png — the ambiguous detour on the maze with three candidate causes
- ch13-once-vs-thrice.gif — posterior bars for sophisticatedOrNaive/discount/alpha after 1 vs 3 identical observations
- ch13-competing-tastes.png — prior vs posterior bars over vegMinusDonut, donutNGreaterDonutS, timeCost

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
