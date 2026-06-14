;; @tier slow
;; agents × llm flagship (ROADMAP Phase 3, item 4, Direction 1: LLM-as-policy) — suite wrapper.
;;
;; The runnable example examples/agents_llm_policy.cljs builds an agent whose POLICY
;; is an LLM generative function: the per-action policy logits ARE the LLM GF's
;; per-action p/assess log-probs, and the action is drawn by a categorical policy GF.
;; It self-checks the GFI composition (valid :action site; p/assess == logsoftmax of
;; the LLM logits; p/generate over the 4 actions has probs summing to 1; reproducibility;
;; state-dependence) and exits non-zero on any failure — or SKIPS cleanly if the
;; qwen3-0.6b model is absent. The example's self-check is async (model load +
;; tokenizer encode are promises), so this wrapper calls run-or-skip explicitly and
;; returns its promise (nbb awaits the top-level promise); a plain require would not
;; await it.
;;
;; Run: bun run --bun nbb test/genmlx/agents_llm_policy_test.cljs

(ns genmlx.agents-llm-policy-test
  (:require [agents-llm-policy :as flag]))

(flag/run-or-skip)
