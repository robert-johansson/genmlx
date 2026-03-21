(ns seven-segment
  "Emergent 7-segment display conventions via neural perception.

   A speaker assigns display patterns to characters. A listener sees
   the pattern through a neural perceptual model (ResNet trained on
   EMNIST handwriting) and infers the character. RSA-like iteration
   converges on display conventions that maximize discriminability.

   The neural perception comes from memo's pretrained ResNet — the
   128×36 probability matrix P(character | display pattern) is loaded
   from 7seg_perception.json. The RSA reasoning runs in GenMLX as
   pure functional iteration.

   This demonstrates the composition of neural perception (from JAX)
   with probabilistic pragmatic reasoning (in GenMLX).

   Implements memo's demo-7segment.ipynb."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact]
            ["fs" :as fs])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Load neural perception matrix from memo's pretrained ResNet
;; =========================================================================

;; P(character | display pattern): [128 patterns, 36 characters]
;; Characters: 0-9 then A-Z
(def perception-matrix
  (let [json-str (.readFileSync fs "examples/memo/7seg_perception.json" "utf8")
        data (js/JSON.parse json-str)
        flat (into-array (for [row data, val row] val))]
    (mx/reshape (.astype (mx/array flat) mx/float32) #js [128 36])))

(mx/eval! perception-matrix)
(println "Loaded perception matrix:" (mx/shape perception-matrix))

;; Character labels
(def chars "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

;; Subset of "well-formed" patterns (from memo's notebook)
(def well-formed
  [27 40 42 43 44 45 46 47 53 55 57 59 61 62 63
   79 80 81 84 87 89 90 91 94 95 102 106 109 110
   111 116 117 118 119 121 122 123 124 125 127])

;; Characters to display (digits + some letters)
(def display-chars [0 24 1 18 2 35 9 16 6 11 20 29])
;; 0, O, 1, I, 2, Z, 9, G, 6, B, K, T

;; =========================================================================
;; RSA iteration: speaker ↔ listener with neural perception
;; =========================================================================

(def n-patterns (count well-formed))
(def n-chars 36)

;; Extract perception sub-matrix for well-formed patterns: [n-patterns, 36]
(def channel
  (let [indices (.astype (mx/array (clj->js well-formed)) mx/int32)
        rows (mapv #(mx/take-idx perception-matrix (mx/idx indices %) 0) (range n-patterns))]
    (mx/stack (clj->js (vec rows)) 0)))
(mx/eval! channel)

(defn rsa-step
  "One RSA step. Pure function: state → state'.
   Speaker chooses pattern weighted by listener's accuracy.
   Listener inverts speaker using neural perception + Bayes."
  [beta {:keys [speaker]}]
  (let [;; speaker: [n-chars, n-patterns] — P(pattern | char)
        ;; channel: [n-patterns, n-chars] — P(char perceived | pattern) from ResNet
        ;; P(perceived-char | intended-char) = sum_p speaker(p|c) * channel(p, perceived)
        p-perceived (mx/matmul speaker channel)  ;; [36, 36]
        ;; Listener: P(intended | perceived) ∝ P(perceived | intended) — Bayes
        listener (mx/softmax (mx/log (mx/maximum p-perceived (mx/scalar 1e-30))) 0)
        ;; Accuracy for each (char, pattern):
        ;; P(correct | char=c, pattern=p) = sum_perceived channel(p, perceived) * listener(c | perceived)
        accuracy (mx/transpose (mx/matmul channel (mx/transpose listener)))
        ;; New speaker: softmax over patterns weighted by accuracy
        new-speaker (mx/softmax (mx/multiply (mx/scalar beta)
                      (mx/log (mx/maximum accuracy (mx/scalar 1e-30)))) -1)
        _ (mx/eval! new-speaker)]
    {:speaker new-speaker :listener listener}))

;; =========================================================================
;; Demo
;; =========================================================================

(println "\n=============================================")
(println " 7-Segment Display: Neural Perception + RSA")
(println "=============================================\n")

;; What does the ResNet see for standard digit patterns?
(println "-- Neural perception of standard digits --\n")
(let [std-patterns [63 6 91 79 102 109 125 7 127 111]]
  (doseq [d (range 10)]
    (let [p-idx (nth std-patterns d)
          row (mx/take-idx perception-matrix (mx/scalar p-idx mx/int32) 0)
          _ (mx/eval! row)
          top3 (->> (range 36)
                    (map (fn [i] [i (mx/item (mx/idx row i))]))
                    (sort-by second >)
                    (take 3))]
      (println (str "  Digit " d ": "
                    (clojure.string/join ", "
                      (map (fn [[i p]] (str (nth chars i) ":" (.toFixed p 3))) top3)))))))

;; RSA iteration
(println "\n-- RSA iteration (beta=2.0, 3 levels) --\n")

(let [;; Initial speaker: based on neural perception (level 0)
      ;; For each char, weight patterns by how likely the ResNet recognizes them
      init-speaker (mx/softmax (mx/multiply (mx/scalar 2.0)
                     (mx/log (mx/maximum (mx/transpose channel) (mx/scalar 1e-30)))) -1)
      _ (mx/eval! init-speaker)
      ;; Iterate
      states (->> {:speaker init-speaker}
                  (iterate (partial rsa-step 2.0))
                  (take 4)
                  vec)]

  (doseq [[t {:keys [speaker]}] (map-indexed vector states)]
    (println (str "Level " t ":"))
    (doseq [ci display-chars]
      (let [row (mx/take-idx speaker (mx/scalar ci mx/int32) 0)
            _ (mx/eval! row)
            best-idx (mx/item (mx/argmin (mx/negative row)))
            best-pattern (nth well-formed best-idx)
            bits (mapv #(bit-and (bit-shift-right best-pattern %) 1) (range 7))]
        (println (str "  " (nth chars ci) " → pattern " best-pattern
                      " [" (clojure.string/join "" bits) "]"))))
    (println))

  ;; Verification
  (let [final (:speaker (last states))
        model (gen []
                (let [c (trace :char (dist/weighted (vec (repeat n-chars 1.0))))
                      p-logits (mx/log (mx/maximum (mx/take-idx final c 0) (mx/scalar 1e-30)))
                      p (trace :pattern (dist/categorical p-logits))]
                  c))
        mi (/ (exact/mutual-info model #{:char} #{:pattern}) (js/Math.log 2))]
    (println (str "I(character; pattern) = " (.toFixed mi 3) " bits"
                  " (max = " (.toFixed (/ (js/Math.log 36) (js/Math.log 2)) 3) " bits)"))
    (assert (> mi 3.0) "MI > 3 bits with neural perception")
    (println "PASS: mutual information > 3 bits")

    ;; Check distinct encodings for display chars
    (let [bests (mapv (fn [ci]
                        (let [row (mx/take-idx final (mx/scalar ci mx/int32) 0)
                              _ (mx/eval! row)]
                          (mx/item (mx/argmin (mx/negative row)))))
                      display-chars)
          n-distinct (count (set bests))]
      (println (str "Distinct encodings for " (count display-chars)
                    " display chars: " n-distinct))
      (assert (>= n-distinct 6) "at least 6 distinct encodings")
      (println "PASS: sufficient distinct encodings"))))

(println "\nAll checks passed.")
