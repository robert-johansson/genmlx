;; @tier slow
;; FIRST EXTERNAL ENVIRONMENT (ROADMAP Phase 3, item 5) — suite wrapper.
;;
;; The runnable example examples/external_env.cljs reifies the agent/environment
;; boundary as real network I/O over the Bun world membrane (genmlx.world.net):
;; actuator = an outbound action POSTed to a localhost environment server, sensor =
;; the observation read back. It self-checks three sections — (1) MDP across the wire
;; is BIT-IDENTICAL to in-process simulate-mdp at the deterministic regime (membrane
;; faithfulness); (2) a POMDP whose latent is hidden behind the membrane, where the
;; belief filters on sensor observations and matches simulate-pomdp exactly; (3) an
;; external 'noisy oracle' API modelled (never GF'd) with importance sampling
;; recovering the latent — exiting non-zero on any failed check, or SKIPPING cleanly
;; if the Bun network membrane is unavailable (not under `bun`).
;;
;; @tier slow is load-bearing: the slow tier runs SERIAL, so this stands up Bun.serve
;; listeners + does Metal value-iteration without parallel-GPU contention. The
;; self-check is async, so this wrapper calls run-or-skip explicitly and returns its
;; promise (nbb awaits the top-level promise; with-server guarantees server teardown).
;;
;; Run: bun run --bun nbb test/genmlx/external_env_test.cljs

(ns genmlx.external-env-test
  (:require [external-env :as flag]))

(flag/run-or-skip)
