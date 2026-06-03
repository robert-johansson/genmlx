;; Headless tests for agentmodels Chapter 7 — Multi-agent models (native genmlx.agents).
;; Run: bun run --bun nbb test/genmlx/agentmodels_multi_agent_test.cljs
;;
;; Ground truth is ANALYTIC (derived from the model math), not from examples/memo.
;; Sections also demonstrate the core genmlx.agents principle: the SAME agent GF runs
;; under exact OR sampled OR mixed inference — the backend is a pluggable seam.

(ns genmlx.agentmodels-multi-agent-test
  (:require [agentmodels.multi-agent :as ma]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol) (do (vswap! passed inc) (println " PASS" msg "  =" actual))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))

(defn- argmax-idx [xs] (first (apply max-key second (map-indexed vector xs))))

;; ===========================================================================
(println "\n== Section 1: Schelling — focal-point amplification (exact) ==")
(let [{:keys [alice bob]} (ma/schelling-agents)
      p (fn [probs] (ma/p-popular probs))]
  (println "  depth :  bob P(popular)   alice P(popular)")
  (doseq [d (range 5)]
    (println (str "    " d "   :    "
                  (.toFixed (p (bob d)) 4) "        "
                  (if (>= d 1) (.toFixed (p (alice d)) 4) "   -"))))
  (assert-close "bob(0) = prior 0.55"        0.55   (p (bob 0))   1e-6)
  (assert-close "alice(1) = 0.5990"          0.5990 (p (alice 1)) 1e-3)
  (assert-close "bob(1) = 0.6461"            0.6461 (p (bob 1))   1e-3)
  (assert-close "alice(2) = 0.6905"          0.6905 (p (alice 2)) 1e-3)
  (assert-close "bob(2) = 0.7317"            0.7317 (p (bob 2))   1e-3)
  (assert-true  "alice P(popular) strictly increasing in depth"
                (apply < (mapv #(p (alice %)) (range 1 5))))
  (assert-true  "alice(4) P(popular) >= 0.83 (focal-point convergence)"
                (>= (p (alice 4)) 0.83)))

;; ===========================================================================
(println "\n== Section 2: Schelling — inference is pluggable (exact ≈ importance, same GF) ==")
(let [exact-a1 (ma/p-popular ((:alice (ma/schelling-agents)) 1))
      ;; reason about the SAME coordination GF by importance sampling instead:
      samp-a1  (ma/p-popular ((:alice (ma/schelling-agents (ma/importance-marginal 4000))) 1))]
  (println "  alice(1) P(popular):  exact =" (.toFixed exact-a1 4) "   importance =" (.toFixed samp-a1 4))
  (assert-close "importance sampling reproduces exact alice(1) (same agent, swapped backend)"
                exact-a1 samp-a1 0.05))

;; ===========================================================================
(println "\n== Section 3: RSA sprouted-seeds — scalar implicature (exact) ==")
;; utterances [all=0 some=1 none=2], states [0 1 2 3]
(let [{:keys [L0 S1 L1]} (ma/make-rsa {:denotation ma/sprouted-denotation
                                       :state-prior ma/sprouted-prior
                                       :alpha 2.0})
      l0-some (ma/table-row L0 1)
      s1-s3   (ma/table-row S1 3)         ; speaker policy for state 3 = [all some none]
      l1-some (ma/table-row L1 1)
      l1-all  (ma/table-row L1 0)
      l1-none (ma/table-row L1 2)]
  (println "  L0(some)  =" (mapv #(.toFixed % 4) l0-some))
  (println "  S1(.|s=3) =" (mapv #(.toFixed % 4) s1-s3) " [all some none]")
  (println "  L1(some)  =" (mapv #(.toFixed % 4) l1-some))
  ;; L0: literal listener for "some" is uniform over {1,2,3}, zero on 0
  (assert-close "L0(some)->0 = 0"     0.0       (nth l0-some 0) 1e-6)
  (assert-close "L0(some)->1 = 1/3"   (/ 1 3.0) (nth l0-some 1) 1e-4)
  (assert-close "L0(some)->3 = 1/3"   (/ 1 3.0) (nth l0-some 3) 1e-4)
  ;; S1: a speaker who knows state=3 mostly says "all" (0.9), rarely "some" (0.1)
  (assert-close "S1(all | s=3) = 0.9"  0.9 (nth s1-s3 0) 1e-4)
  (assert-close "S1(some| s=3) = 0.1"  0.1 (nth s1-s3 1) 1e-4)
  ;; L1: the scalar implicature — "some" ⇒ not none (P0=0) AND not all (P3 ≪ P1=P2)
  (assert-close "L1(some)->0 = 0   (not none)"        0.0           (nth l1-some 0) 1e-6)
  (assert-close "L1(some)->1 = 10/21"                 (/ 10.0 21.0) (nth l1-some 1) 1e-4)
  (assert-close "L1(some)->2 = 10/21"                 (/ 10.0 21.0) (nth l1-some 2) 1e-4)
  (assert-close "L1(some)->3 = 1/21 (not all)"        (/ 1.0 21.0)  (nth l1-some 3) 1e-4)
  (assert-true  "implicature: P(3|some) ≪ P(1|some)" (< (nth l1-some 3) (* 0.2 (nth l1-some 1))))
  (assert-close "L1(all)->3  = 1"  1.0 (nth l1-all 3)  1e-6)
  (assert-close "L1(none)->0 = 1"  1.0 (nth l1-none 0) 1e-6))

;; ===========================================================================
(println "\n== Section 4: RSA re-parameterized denotation — referential implicature ==")
;; Same tower, different matrix. Utterances [u0=0 uboth=1], referents [r0 r1].
(let [{:keys [L1]} (ma/make-rsa {:denotation ma/reference-denotation
                                 :state-prior ma/reference-prior
                                 :alpha 1.0})
      l1-uboth (ma/table-row L1 1)
      l1-u0    (ma/table-row L1 0)]
  (println "  L1(uboth) =" (mapv #(.toFixed % 4) l1-uboth) " [r0 r1]")
  (assert-close "L1(uboth)->r0 = 0.25"  0.25 (nth l1-uboth 0) 1e-4)
  (assert-close "L1(uboth)->r1 = 0.75"  0.75 (nth l1-uboth 1) 1e-4)
  (assert-true  "implicature: ambiguous 'uboth' resolves toward r1"
                (> (nth l1-uboth 1) (nth l1-uboth 0)))
  (assert-close "L1(u0)->r0 = 1.0"      1.0  (nth l1-u0 0)    1e-6))

;; ===========================================================================
(println "\n== Section 5: RSA mixed inference — sampled L0 → exact S1/L1, implicature survives ==")
(let [{:keys [L1]} (ma/make-rsa {:denotation ma/sprouted-denotation
                                 :state-prior ma/sprouted-prior
                                 :alpha 2.0
                                 :infer (ma/importance-marginal 3000)})
      l1-some (ma/table-row L1 1)]
  (println "  L1(some) [sampled-L0 pipeline] =" (mapv #(.toFixed % 4) l1-some))
  (assert-close "P(0|some) = 0 exactly (rejection never keeps state 0)" 0.0 (nth l1-some 0) 1e-9)
  (assert-true  "P(3|some) ≪ P(1|some) survives mixed inference"
                (< (nth l1-some 3) (* 0.3 (nth l1-some 1))))
  (assert-true  "P(1|some) ≈ P(2|some) ≈ 0.476 (within MC error)"
                (and (< 0.40 (nth l1-some 1) 0.56) (< 0.40 (nth l1-some 2) 0.56))))

;; ===========================================================================
(println "\n== Section 6: Tic-tac-toe — forced win (planning and non-planning) ==")
(let [agent (ma/make-game-agent {:alpha ##Inf})]
  (assert-equal "planning best move completes the row (cell 2)"
                2 (ma/best-move agent ma/forced-win-board :x))
  ;; the one-step agent also finds an IMMEDIATE win (its argmax cell, not list position)
  (assert-equal "one-step agent also takes the immediate win (cell 2)"
                2 (first (apply max-key second (ma/non-planning-move-q ma/forced-win-board :x)))))

;; ===========================================================================
(println "\n== Section 7: Tic-tac-toe — forced block (planning vs non-planning) ==")
(let [agent (ma/make-game-agent {:alpha ##Inf})
      pq    ((:move-q agent) ma/forced-block-board :x)
      npq   (ma/non-planning-move-q ma/forced-block-board :x)
      block-q (second (first (filter #(= 5 (first %)) pq)))
      other-q (apply max (map second (remove #(= 5 (first %)) pq)))]
  (println "  planning move-q     :" (mapv (fn [[m q]] [m (.toFixed q 1)]) pq))
  (println "  non-planning move-q :" (mapv (fn [[m q]] [m (.toFixed q 1)]) npq))
  (assert-equal "PLANNING blocks at cell 5 (forces a draw instead of a loss)"
                5 (ma/best-move agent ma/forced-block-board :x))
  (assert-true  "planning: block value (draw=0) strictly beats every non-block move (loss=-10)"
                (> block-q other-q))
  (assert-true  "one-step agent is INDIFFERENT (every immediate move scores 0)"
                (every? #(< (Math/abs (second %)) 1e-9) npq)))

;; ===========================================================================
(println "\n== Section 8: Tic-tac-toe — factor(α·EU) as a softmax-action GF ==")
(let [agent (ma/make-game-agent {:alpha ##Inf})
      moves (mapv (fn [_] ((:act agent) ma/forced-win-board :x)) (range 5))]
  (println "  softmax-action (α=##Inf) :act moves:" moves)
  (assert-true  "softmax-action policy deterministically takes the winning move (cell 2)"
                (every? #(= 2 %) moves)))

;; ===========================================================================
(println "\n== Section 9: determinism ==")
(let [agent (ma/make-game-agent {:alpha ##Inf})]
  (assert-equal "best-move is repeatable"
                (ma/best-move agent ma/forced-block-board :x)
                (ma/best-move agent ma/forced-block-board :x)))

;; ===========================================================================
(println (str "\n==== " @passed " passed, " @failed " failed ===="))
(when (pos? @failed) (js/process.exit 1))
