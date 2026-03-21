(ns vizzini
  "Battle of Wits — recursive game theory via exact enumeration.

   Vizzini and Westley play the poisoned cup game from The Princess Bride.
   Two cups (0 and 1). Westley poisons one cup; Vizzini chooses which to drink.

   At each depth of reasoning:
     - Vizzini models Westley's choice using the previous depth's Vizzini probs
     - Westley poisons the cup Vizzini is most likely to avoid
     - Vizzini picks the cup that maximizes survival probability

   The reasoning is a pure function: table → table'. Each depth takes the
   previous depth's probability table and produces the next — a perfect fit
   for Clojure's `iterate`. No mutation, no cache atoms.

   The result oscillates: [0,1], [1,0], [0,1], [1,0]...
   With increasing depth, neither player gains an advantage — the Nash
   equilibrium is uniform [0.5, 0.5]."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; One step of recursive reasoning (pure function: table → table)
;; ---------------------------------------------------------------------------

(defn reasoning-step
  "One depth of Battle of Wits reasoning. Pure function: prev-table → next-table.

   Given Vizzini's previous choice distribution (prev-table):
   1. Westley models Vizzini, poisons optimally (categorical-argmax)
   2. Vizzini picks the cup maximizing survival (1 - poison prob)
   Returns the new Vizzini choice distribution as [P(cup0), P(cup1)]."
  [prev-table]
  (let [;; Westley models Vizzini's choice, poisons to maximize kill
        westley-model (gen []
                        (let [vc (trace :vc (dist/categorical (mx/log prev-table)))
                              p  (trace :p  (exact/categorical-argmax prev-table))]
                          p))
        wr (exact/exact-joint westley-model [] nil)
        westley-poison-probs (mx/exp (exact/marginal (:log-probs wr) (:axes wr) :p))
        _ (mx/eval! westley-poison-probs)
        ;; Vizzini picks the cup that maximizes survival (1 - poison prob)
        eu-survive (mx/subtract (mx/scalar 1.0) westley-poison-probs)
        vizzini-model (gen [] (trace :cup (exact/categorical-argmax eu-survive)))
        result (exact/exact-posterior vizzini-model [] nil)
        table (mx/array #js [(get-in result [:marginals :cup 0])
                              (get-in result [:marginals :cup 1])])
        _ (mx/eval! table)]
    table))

;; ---------------------------------------------------------------------------
;; Depth iteration via `iterate`
;; ---------------------------------------------------------------------------

(defn battle-of-wits
  "Run Battle of Wits for n depths. Returns a vector of probability tables,
   one per depth (0 through n-1).

   The seed [1,0] (Vizzini naively picks cup 0) is never returned — it feeds
   into depth 0's reasoning step. The sequence is:
     seed → depth 0 → depth 1 → ...

   Pure functional — no mutation, just Clojure's `iterate`."
  [n-depths]
  (let [seed (mx/array #js [1 0])  ;; naive prior: Vizzini picks cup 0
        _ (mx/eval! seed)]
    (->> seed
         (iterate reasoning-step)
         (drop 1)                  ;; skip seed
         (take n-depths)
         vec)))

;; ---------------------------------------------------------------------------
;; Run and display
;; ---------------------------------------------------------------------------

(println "Battle of Wits (The Princess Bride)")
(println "====================================")
(println)
(println "Vizzini and Westley play the poisoned cup game.")
(println "Vizzini recursively models Westley's strategy.")
(println)

(let [tables (battle-of-wits 4)]
  (doseq [[d v] (map-indexed vector tables)]
    (let [p0 (mx/item (mx/slice v 0 1))
          p1 (mx/item (mx/slice v 1 2))]
      (println (str "  Depth " d ": P(cup0) = " (.toFixed p0 4)
                    "  P(cup1) = " (.toFixed p1 4))))))

(println)
(println "Pattern: oscillates [0,1] -> [1,0] -> [0,1] -> [1,0]")
(println "Nash equilibrium: converges to [0.5, 0.5] (neither can gain advantage)")

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------

(println)
(println "Verification:")

(let [tables (battle-of-wits 4)
      results (mapv (fn [v]
                      [(mx/item (mx/slice v 0 1))
                       (mx/item (mx/slice v 1 2))])
                    tables)
      expected [[0 1] [1 0] [0 1] [1 0]]
      pass? (atom true)]
  (doseq [[d [p0 p1]] (map-indexed vector results)]
    (let [[e0 e1] (nth expected d)
          ok (and (< (abs (- p0 e0)) 1e-5)
                  (< (abs (- p1 e1)) 1e-5))]
      (when-not ok (reset! pass? false))
      (println (str "  Depth " d ": " (if ok "PASS" "FAIL")
                    " (expected [" e0 ", " e1 "]"
                    "  got [" (.toFixed p0 4) ", " (.toFixed p1 4) "])"))))
  (println)
  (if @pass?
    (println "All checks passed.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
