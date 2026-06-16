# Beyond agentmodels: LLM-as-Policy, LLM-over-Agent, and Remote Environments

> **Ports** agentmodels.org extension beyond agentmodels (Phase 3 external environment + LLM agents); thematically continues 4 / 6.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

LLM-as-policy = a Pac-Man whose action logits ARE an LLM's p/assess log-probs over up/down/left/right given a state description. LLM-over-agent = inverse planning over a hidden goal that FUSES the agent-GF likelihood P(action|goal) with an LLM reasoner P_LLM(yes|goal). Remote = a Gym-style episodic transport running a Pac-Man MDP/POMDP env as an HTTP service (genmlx.world.net), with async client rollouts crossing the wire.

## What this chapter builds on

- **genmlx code:** partial — LLM-as-policy and LLM-over-agent exist and are tested but SKIP when qwen3-0.6b is absent (no always-on regression): test/genmlx/agents_llm_policy_test.cljs (per-action LLM logits, p/assess==logsoftmax, reproducible, state-dependent) + test/genmlx/agents_llm_inverse_test.cljs (fused agent+LLM posterior, g* optimal-action MAP). Remote is explicitly PROVISIONAL/unfrozen: src/genmlx/agents/remote.cljs (gym-transport, mdp-env-handler, pomdp-env-handler, remote-mdp-rollout, remote-pomdp-rollout) marked maturity:partial in CONTRACTS.md. Mark this chapter clearly as the unfrozen frontier; the cs188 layout→genmlx bridge (~20 lines) for a live remote Pac-Man env is must-write.

## Live figures (planned)

- ch17-llm-policy.png — LLM per-action logits → softmax action bars for a maze state
- ch17-llm-inverse.png — fused agent+LLM posterior over hidden goal bars
- ch17-remote-rollout.gif — Pac-Man driven over the HTTP gym transport (live remote env)

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
