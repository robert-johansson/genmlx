(ns sally-anne
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Sally-Anne False Belief Test — 3-level theory of mind via exact enumeration
;;
;; The classic developmental psychology task:
;;   1. Sally puts a marble in her BASKET and leaves the room
;;   2. Anne moves the marble to her BOX (or doesn't)
;;   3. Sally comes back — where does she look for the marble?
;;
;; Three levels of reasoning (all computed by exact enumeration):
;;   Level 1 (Anne):  Chooses an action (stay/move), producing a new position
;;   Level 2 (Sally): Reasons about Anne's action given what she observed,
;;                    then looks where she thinks the marble is
;;   Level 3 (Child): Runs inference on Sally's model to predict her behavior
;;
;; Encoding: location 0 = BOX, location 1 = BASKET
;;           action  0 = STAY, action  1 = MOVE
;;           obs     0 = NONE (didn't see), 1 = SAW STAY, 2 = SAW MOVE

;; ---------------------------------------------------------------------------
;; Level 1: Anne's action model
;; ---------------------------------------------------------------------------

(defn sa-do-action
  "Apply action to marble location. STAY keeps it; MOVE swaps it."
  [loc action]
  (mx/where (mx/eq? action 0)
            loc
            (mx/subtract (mx/scalar 1) loc)))

(defn sa-anne-model
  "Anne's model: mostly stays (p=0.99), rarely moves (p=0.01).
   Traces: :action, :new-pos, :obs (what Sally observes)."
  [marble-pos-val]
  (gen []
    (let [action (trace :action (dist/weighted [0.99 0.01]))
          new-pos-float (sa-do-action (mx/scalar marble-pos-val) action)
          ;; Deterministic new-pos from action
          new-pos-probs (mx/transpose
                          (mx/stack [(mx/eq? new-pos-float 0)
                                     (mx/eq? new-pos-float 1)]))
          new-pos (trace :new-pos (dist/categorical (mx/log new-pos-probs)))
          ;; Observation model: NONE always possible, SAW STAY/MOVE only if that happened
          is-stay (mx/eq? action 0)
          is-move (mx/eq? action 1)
          obs-wpp (mx/transpose (mx/stack [(mx/add is-stay is-move) is-stay is-move]))
          obs (trace :obs (dist/categorical (mx/log obs-wpp)))]
      new-pos)))

;; ---------------------------------------------------------------------------
;; Level 2: Sally's reasoning model
;; ---------------------------------------------------------------------------

(defn sa-sally-model
  "Sally reasons about Anne via exact enumeration (exact/Exact splice).
   Given her observation, she marginalizes over Anne's action to get
   P(new-pos | obs), then looks where she thinks the marble most likely is."
  [marble-pos-val]
  (gen []
    (let [;; Splice Anne as an exactly-enumerated sub-model
          anne-probs (splice :anne (exact/Exact (sa-anne-model marble-pos-val)))
          ;; anne-probs shape: [2, 2] — P(new-pos, action | obs)
          ;; Marginalize over action (axis 1) to get P(new-pos | obs)
          p-new-pos (mx/sum anne-probs [1])
          where-look (trace :where-look (dist/categorical (mx/log p-new-pos)))]
      where-look)))

;; ---------------------------------------------------------------------------
;; Level 3: Child observer — runs exact inference on Sally
;; ---------------------------------------------------------------------------

(println "Sally-Anne False Belief Test")
(println "============================\n")

(println "Scenario 1: Marble starts in BASKET, Sally sees NOTHING")
(println "  (Anne moved the marble but Sally didn't see)")
(let [r (exact/exact-posterior (sa-sally-model 1) []
          (cm/choicemap :anne (cm/choicemap :obs (mx/scalar 0 mx/int32))))]
  (println (str "  P(Sally looks in BOX)    = "
                (.toFixed (get-in r [:marginals :where-look 0]) 4)))
  (println (str "  P(Sally looks in BASKET) = "
                (.toFixed (get-in r [:marginals :where-look 1]) 4)))
  (println "  -> Sally has a FALSE BELIEF: she looks in the BASKET (where she left it)\n"))

(println "Scenario 2: Marble starts in BASKET, Sally sees the MOVE")
(let [r (exact/exact-posterior (sa-sally-model 1) []
          (cm/choicemap :anne (cm/choicemap :obs (mx/scalar 2 mx/int32))))]
  (println (str "  P(Sally looks in BOX)    = "
                (.toFixed (get-in r [:marginals :where-look 0]) 4)))
  (println (str "  P(Sally looks in BASKET) = "
                (.toFixed (get-in r [:marginals :where-look 1]) 4)))
  (println "  -> Sally saw the move: she looks in the BOX (correct belief)\n"))

(println "Scenario 3: Marble starts in BOX, Sally sees NOTHING")
(let [r (exact/exact-posterior (sa-sally-model 0) []
          (cm/choicemap :anne (cm/choicemap :obs (mx/scalar 0 mx/int32))))]
  (println (str "  P(Sally looks in BOX)    = "
                (.toFixed (get-in r [:marginals :where-look 0]) 4)))
  (println (str "  P(Sally looks in BASKET) = "
                (.toFixed (get-in r [:marginals :where-look 1]) 4)))
  (println "  -> No move happened: Sally correctly looks in the BOX\n"))

(println "Scenario 4: Marble starts in BOX, Sally sees the MOVE")
(let [r (exact/exact-posterior (sa-sally-model 0) []
          (cm/choicemap :anne (cm/choicemap :obs (mx/scalar 2 mx/int32))))]
  (println (str "  P(Sally looks in BOX)    = "
                (.toFixed (get-in r [:marginals :where-look 0]) 4)))
  (println (str "  P(Sally looks in BASKET) = "
                (.toFixed (get-in r [:marginals :where-look 1]) 4)))
  (println "  -> Sally saw the move to BASKET: she looks in the BASKET (correct belief)\n"))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "Key insight: when Sally DOESN'T see Anne move the marble,")
(println "she looks where she LAST SAW it (false belief). When she")
(println "DOES see the move, she updates correctly (true belief).")
(println "The child (level 3) can predict this via nested exact enumeration.")
