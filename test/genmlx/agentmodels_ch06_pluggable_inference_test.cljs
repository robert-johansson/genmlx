;; @tier slow
;; agentmodels Ch 6 ("Efficient inference") port — suite wrapper.
;;
;; Ch 6 is AUTHORED NATIVELY as the genmlx.agents pluggable-inference chapter:
;; one forward gridworld agent reasoned about by four inference backends that
;; must AGREE — (6a) value iteration == recursive expected-utility, (6b) exact
;; enumeration / importance sampling / Metropolis-Hastings over the same joint
;; goal-inference GF (cross-checked by the independent posterior-sequence oracle),
;; and (6c) gradient/amortized utility recovery through the differentiable planner.
;;
;; The runnable example examples/agentmodels/ch06_pluggable_inference.cljs defines
;; the model + backends as public fns and self-checks via (-main): it prints each
;; backend's posterior and asserts agreement to tolerance (deterministic seeds),
;; exiting non-zero on any failed check. The example only auto-runs when invoked
;; as a script, so this wrapper requires it (loading the model with zero GPU work
;; beyond agent construction) and then calls (-main) explicitly — making a Ch-6
;; regression a hard test failure under test/run.sh (exit code is the reliable
;; signal).
;;
;; Run: bun run --bun nbb test/genmlx/agentmodels_ch06_pluggable_inference_test.cljs

(ns genmlx.agentmodels-ch06-pluggable-inference-test
  (:require [agentmodels.ch06-pluggable-inference :as ch06]))

(ch06/-main)

(println "\n[agentmodels-ch06-pluggable-inference-test] Ch 6 example self-check completed (PASS).")
