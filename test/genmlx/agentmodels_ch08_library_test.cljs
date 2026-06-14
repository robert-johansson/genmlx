;; @tier slow
;; agentmodels.org Ch 8 (WebPPL agents library quick-start) port — suite wrapper.
;;
;; The runnable example examples/agentmodels/ch08_library_guide.cljs self-checks on
;; load: a guided tour of the genmlx.agents foundation — make-line-mdp +
;; make-mdp-agent (optimal/soft), the hiking gridworld, a Naive hyperbolic biased
;; planner, a line POMDP with belief filtering (belief snaps to truth at the
;; signpost; QMDP reaches the true goal), and custom RANDOM + EPSILON-GREEDY policy
;; GFs — asserting each headline behavior and exiting non-zero on any failure.
;; Requiring it here makes a Ch-8 regression a hard test failure under test/run.sh.
;;
;; Run: bun run --bun nbb test/genmlx/agentmodels_ch08_library_test.cljs

(ns genmlx.agentmodels-ch08-library-test
  (:require [agentmodels.ch08-library-guide]))

(println "\n[agentmodels-ch08-library-test] Ch 8 example self-check completed (PASS).")
