(ns agentmodels.ch17-capstone
  "ch17 capstone — the agents x LLM + remote-environment frontier (PROVISIONAL;
   genmlx.agents.remote is explicitly OUTSIDE the frozen v1.0 surface).

   This file is the MODEL-FREE half: it bridges a Pac-Man ASCII maze (the cs188
   `.lay` glyph convention, via genmlx.agents.pacman) into a remote Gym-style MDP
   environment served over the genmlx.world.net membrane, then drives the optimal
   agent against it ACROSS THE WIRE. At the deterministic regime (alpha = ##Inf,
   noise = 0) the remote trajectory is BIT-IDENTICAL to the in-process rollout —
   the proof the membrane is faithful and the agent code is unchanged whether the
   environment is in-process or external.

   The LLM-as-policy half lives in examples/agents_llm_policy.cljs (the per-action
   policy logits ARE an LLM generative function's p/assess log-probs); it is gated
   on a local qwen3-0.6b model and skips cleanly when the model is absent.

   Run: bun run --bun nbb examples/agentmodels/ch17_capstone.cljs"
  (:require [genmlx.agents.pacman :as pac]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.remote :as remote]
            [genmlx.world.net :as net]
            [agentmodels.harness :as chk]
            [promesa.core :as pr]))

(defn pacman-remote-handler
  "The cs188-layout -> genmlx-remote bridge: compile a Pac-Man ASCII maze into a
   build-mdp bundle (pac/pacman-mdp) and hand it to remote/mdp-env-handler — the
   single (fn [route payload] -> response) that genmlx.world.net/serve! hosts. The
   world's true state lives behind the membrane; the agent only learns the next
   cell by receiving it across the wire."
  [maze & [opts]]
  (remote/mdp-env-handler (pac/pacman-mdp {:ascii maze}) opts))

(defn -main []
  (let [maze    pac/corridor
        mdp     (pac/pacman-mdp {:ascii maze})
        ag      (agent/make-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 12})
        start   (:start-idx mdp)
        H       12
        in-proc (agent/simulate-mdp ag start H)]
    (println "\n== ch17 capstone — a Pac-Man maze as a REMOTE environment ==")
    (net/with-server (pacman-remote-handler maze {:start start})
      (fn [url]
        (pr/let [rem (remote/remote-mdp-rollout ag (remote/gym-transport url) H)]
          (println "  in-process states :" (:states in-proc))
          (println "  across-wire states:" (:states rem))
          (chk/check-true "remote Pac-Man rollout is bit-identical to in-process (states)"
                          (= (:states in-proc) (:states rem)))
          (chk/check-true "remote actions match in-process exactly"
                          (= (:actions in-proc) (:actions rem)))
          (chk/check-true "Pac-Man reached the goal across the wire"
                          (= (last (:states in-proc)) (last (:states rem))))
          (chk/check-true "each step was a real network round-trip"
                          (pos? (count (:actions rem))))
          (chk/report!))))))

(def ^:private main?
  (boolean (some #(re-find #"ch17_capstone\.cljs" (str %)) (array-seq js/process.argv))))
(when main? (-main))
