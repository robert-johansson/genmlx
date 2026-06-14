;; @tier slow
;; agentmodels.org Ch 2 (WebPPL primer) port — suite wrapper.
;;
;; The runnable example examples/agentmodels/ch02_webppl.cljs self-checks on load:
;; ERP draws, multivariateGaussian, binomial-from-3-flips (two ways agree), the
;; twoHeads/moreThanTwoHeads conditioning (P(first=H|total>=2)=0.75, evidence 0.5,
;; posterior {2:0.75, 3:0.25}), and a forward positionDist — each asserted against
;; the analytic reference number, exiting non-zero on any failed check. Requiring
;; it here makes a Ch-2 regression a hard test failure under test/run.sh.
;;
;; Run: bun run --bun nbb test/genmlx/agentmodels_ch02_webppl_test.cljs

(ns genmlx.agentmodels-ch02-webppl-test
  (:require [agentmodels.ch02-webppl]))

(println "\n[agentmodels-ch02-webppl-test] Ch 2 example self-check completed (PASS).")
