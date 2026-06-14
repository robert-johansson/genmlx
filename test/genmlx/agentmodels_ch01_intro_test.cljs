;; @tier slow
;; agentmodels.org Ch 1 (Introduction) port — suite wrapper.
;;
;; The runnable example examples/agentmodels/ch01_intro.cljs self-checks on load:
;; it prints each marginal and asserts it against the analytic reference number
;; (coin taster; geometric two ways, P(n=k)=0.5^(k+1), E[n]=1), exiting non-zero
;; on any failed check. Requiring it here makes a Ch-1 regression a hard test
;; failure under test/run.sh (exit code is the runner's reliable signal).
;;
;; Run: bun run --bun nbb test/genmlx/agentmodels_ch01_intro_test.cljs

(ns genmlx.agentmodels-ch01-intro-test
  (:require [agentmodels.ch01-intro]))

(println "\n[agentmodels-ch01-intro-test] Ch 1 example self-check completed (PASS).")
