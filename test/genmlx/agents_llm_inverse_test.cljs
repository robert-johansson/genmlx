;; @tier slow
;; agents × llm flagship (ROADMAP Phase 3, item 4, Direction 2: LLM reasons OVER an
;; agent model) — suite wrapper.
;;
;; The runnable example examples/agents_llm_inverse.cljs does inverse planning over an
;; agent's hidden goal by FUSING two generative functions: the agent GF's action
;; likelihood P(action|goal) and the LLM GF's reasoner score P_LLM(yes|goal), in one
;; host-enumerated Bayesian inference (no inference engine over the LLM → no KV-cache
;; hazard). It asserts the EXACT agent-model half (observing g*'s optimal action makes
;; g* the agent-model MAP) and the validity of the LLM reasoner half (probabilities in
;; [0,1], posteriors normalized), exiting non-zero on failure — or SKIPS cleanly if the
;; qwen3-0.6b model is absent. The self-check is async, so this wrapper calls
;; run-or-skip explicitly and returns its promise (nbb awaits the top-level promise).
;;
;; Run: bun run --bun nbb test/genmlx/agents_llm_inverse_test.cljs

(ns genmlx.agents-llm-inverse-test
  (:require [agents-llm-inverse :as flag]))

(flag/run-or-skip)
