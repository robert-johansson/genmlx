(ns agentmodels.multi-agent
  "Chapter 7 — Multi-agent models, as native genmlx.agents.

   An *agent is a generative function*; the inference backend used to reason about
   it — exact enumeration, importance/Monte-Carlo, or a MIX — is a pluggable seam,
   orthogonal to the model. That separation is the point: every model here takes an
   `infer` argument and defaults it, and the test demonstrates exact ≈ sampled on
   the very same agent GF.

   Three agents built from one small set of primitives:

   - Schelling coordination : two nested-reasoning GFs that condition on meeting
                              (native bernoulli-observe), memoized over a depth
                              ladder with exact/with-cache. Focal-point amplification.
   - RSA (sprouted seeds)   : the speaker is a softmax-action agent — factor(α·EU),
                              EU = the literal listener's log-score; the literal
                              listener is a GF run through the (pluggable) inference
                              seam. A mixed pipeline: inferred L0 → softmax S1 → Bayes L1.
   - Tic-tac-toe game tree  : a softmax-action agent over a with-cache game-tree
                              expected utility = expectation over the opponent's own
                              softmax policy (soft minimax; hard minimax at α=##Inf).

   Source: agentmodels.org Chapter 7 (canonical). Ground truth is ANALYTIC (derived
   from the model math), not copied from examples/memo (a separate memo-conversion
   corpus). No engine change — composes genmlx.inference.exact, genmlx.agents.helpers,
   and exact/with-cache only."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.inference.exact :as exact]
            [genmlx.agents.helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ===========================================================================
;; Inference seam — an agent is a GF; HOW we reason about it is pluggable.
;;
;; A "marginal inferer" reasons about a discrete gen model GF and returns the [k]
;; posterior marginal of one address under observations `obs`. `exact-marginal`
;; is the default (exact enumeration); `importance-marginal` is a sampling backend
;; that reasons about the SAME model GF. Swapping one for the other changes nothing
;; about the agent definition — that orthogonality is the whole idea.
;; ===========================================================================

(defn exact-marginal
  "Default reasoning backend: exact enumeration. Returns the [k] posterior marginal
   of `addr` in `model` under observation choicemap `obs`, as an MLX vector."
  [model obs addr k]
  ;; Reads :marginals off exact-posterior's result map directly (no public addr-marginal
  ;; accessor exists yet); the [addr v] lookup is the documented shape of that map.
  (let [r (exact/exact-posterior model [] obs)]
    (mx/array (clj->js (mapv #(get-in (:marginals r) [addr %]) (range k))))))

(defn importance-marginal
  "A SAMPLING reasoning backend: importance sampling with the model's own priors as
   the proposal. `p/generate` constrains `obs` and returns weight = log p(obs | sampled
   latents) — exactly the importance log-weight — so weighting samples by exp(weight)
   yields the posterior. Reasons about the SAME agent GF as `exact-marginal`,
   demonstrating that the inference backend is orthogonal to the model. Returns a
   [k] probability vector for `addr`. (n samples; uses live entropy, so callers
   should assert with a tolerance, not for bit-equality.)"
  [n]
  (fn [model obs addr k]
    (let [tally (reduce
                  (fn [acc _]
                    (let [{:keys [trace weight]} (p/generate (dyn/auto-key model) [] obs)
                          w (Math/exp (mx/item weight))
                          v (int (mx/item (cm/get-value (cm/get-submap (:choices trace) addr))))]
                      (update acc v + w)))
                  (vec (repeat k 0.0))
                  (range n))
          z (reduce + tally)]
      (mx/array (clj->js (mapv #(/ % (max z 1e-300)) tally))))))

;; ===========================================================================
;; Model 1 — Schelling coordination (focal-point amplification)
;;
;; agentmodels.org Ch 7: alice(d) samples a location ~ prior, samples bob(d-1)'s
;; location, and conditions on meeting; bob(d) coordinates with alice(d); base case
;; bob(0) = prior (the non-strategic agent). There is NO utility/softmax here — pure
;; coordination by conditioning. The repeated multiplication by the prior at each
;; rung amplifies the 0.55 focal point toward ~1.0.
;; ===========================================================================

(def location-prior
  "P(popular-bar)=0.55, P(unpopular-bar)=0.45. Index 0 = popular (the focal point)."
  (mx/array #js [0.55 0.45]))

(defn coordination-model
  "One agent reasoning about the other: sample my location ~ prior and the other's
   ~ `other-probs`, then CONDITION on meeting. Conditioning is the native GenMLX
   bernoulli-observe idiom (cf. exact/observe-constraint): a Bernoulli whose success
   probability is 1 iff we picked the same bar, observed to be 1. Works identically
   under exact enumeration and importance sampling — the inference is pluggable."
  [other-probs]
  (gen []
    (let [other (trace :other (dist/categorical (mx/log other-probs)))
          me    (trace :me    (dist/categorical (mx/log location-prior)))
          match (.astype (mx/equal me other) mx/float32)]
      (trace :meet (dist/bernoulli match))           ; observed = 1  ⇒  condition(me == other)
      me)))

(def ^:private MEET (cm/choicemap :meet (mx/scalar 1)))

(defn coordinate
  "The coordination posterior over my location (the [2] marginal of :me), given the
   other agent's location distribution. `infer` is the pluggable reasoning backend."
  [other-probs infer]
  (infer (coordination-model other-probs) MEET :me 2))

(defn schelling-agents
  "Build the mutually-recursive alice/bob coordination agents over a depth ladder.
   Returns {:alice :bob}, each a memoized (fn [depth] -> [2] location probs):
     alice(d) coordinates with bob(d-1);  bob(d>0) coordinates with alice(d);
     bob(0) = prior (base case). `infer` defaults to exact enumeration; pass
     (importance-marginal n) to reason by sampling instead — same agents, same
     numbers (within MC error). Call alice for d>=1, bob for d>=0."
  ([] (schelling-agents exact-marginal))
  ([infer]
   ;; Atoms (not letfn) because alice and bob are mutually recursive AND each is wrapped
   ;; in exact/with-cache: we need each to close over the OTHER's memoized fn, which only
   ;; exists after both are built. The atoms are write-once during construction and never
   ;; mutated thereafter — a construction-scoped knot-tie, not stateful recursion.
   (let [alice (atom nil)
         bob   (atom nil)]
     (reset! alice (exact/with-cache (fn [d] (coordinate (@bob (dec d)) infer))))
     (reset! bob   (exact/with-cache (fn [d] (if (zero? d)
                                               location-prior
                                               (coordinate (@alice d) infer)))))
     {:alice @alice :bob @bob})))

(defn p-popular
  "P(popular-bar) from a [2] location-probability vector, as a JS number."
  [probs]
  (mx/item (mx/slice probs 0 1)))

;; ===========================================================================
;; Model 2 — RSA: the speaker is a softmax-action agent (sprouted-seeds + reference)
;;
;; A denotation is a [n-utts × n-states] truth matrix (rows = utterance). The tower:
;;   L0(s|u) ∝ prior(s)·denotation(u,s)              literal listener (a GF, via `infer`)
;;   S1(u|s) ∝ exp(α · log L0(s|u))                  pragmatic speaker = softmax-action
;;   L1(s|u) ∝ prior(s)·S1(u|s)                      pragmatic listener = Bayes(S1)
;; This is a MIXED-inference pipeline: L0 goes through the pluggable inference seam
;; (swap exact↔importance freely); S1/S2/L1 are the speaker's softmax policy and its
;; Bayesian inversion — pure tensor transforms of the L0 table. The "sprouted-seeds
;; variant = re-parameterized denotation matrix": same code, different denotation.
;; ===========================================================================

(defn literal-listener-model
  "L0 as a GF: a state ~ prior, observe that the utterance's denotation (`truth-row`,
   a [n-states] 0/1 vector gathered at the sampled state) holds — native bernoulli-
   observe = condition(meaning(state)). Runs under exact OR importance unchanged."
  [state-prior truth-row]
  (gen []
    (let [s (trace :s (dist/categorical (mx/log state-prior)))]
      (trace :holds (dist/bernoulli (mx/take-idx truth-row s)))   ; observed = 1 ⇒ condition(denotes)
      s)))

(def ^:private HOLDS (cm/choicemap :holds (mx/scalar 1)))

(defn literal-listener
  "The literal-listener table L0: [n-utts × n-states], row u = P_L0(state | utterance u).
   Each row is produced by the pluggable inference seam over the listener GF."
  [{:keys [denotation state-prior infer] :or {infer exact-marginal}}]
  (let [[nu ns] (mx/shape denotation)
        rows (mapv (fn [u] (infer (literal-listener-model state-prior (mx/idx denotation u))
                                  HOLDS :s ns))
                   (range nu))]
    (mx/stack rows)))

(defn speaker-table
  "The pragmatic speaker S1: [n-states × n-utts], row s = P_S1(utterance | state s).
   S1(u|s) = softmax_u(α · log L0(s|u)) — Boltzmann/softmax-action over utterances
   with utility = the literal listener's log-score. (This is agentmodels' factor(α·EU)
   realized as a policy; see genmlx.agents.helpers/softmax-action.)"
  [L0 alpha]
  (mx/softmax (mx/multiply (mx/scalar alpha) (mx/log (mx/transpose L0))) 1))

(defn pragmatic-listener-table
  "The pragmatic listener L1: [n-utts × n-states], row u = P_L1(state | utterance u).
   Bayesian inversion of the speaker: L1(s|u) ∝ prior(s)·S1(u|s)."
  [L0 state-prior alpha]
  (let [S1     (speaker-table L0 alpha)                         ; [ns × nu]
        [ns _] (mx/shape S1)
        joint (mx/multiply S1 (mx/reshape state-prior #js [ns 1]))  ; [ns × nu]
        col-z (mx/sum joint [0] true)                           ; [1 × nu] = P(u)
        L1-su (mx/divide joint col-z)]                          ; [ns × nu] = P(s | u)
    (mx/transpose L1-su)))                                      ; [nu × ns]

(defn make-rsa
  "Build an RSA tower over a denotation matrix. Returns {:L0 :S1 :L1}, the literal
   listener, speaker, and pragmatic listener tables. `infer` (default exact) is the
   pluggable backend for the literal listener — pass (importance-marginal n) to make
   L0 sampled while S1/L1 stay exact (a genuine mixed-inference pipeline).

   `alpha` is the speaker rationality / optimality parameter (factor(alpha·EU)); it is
   domain-specific. The canonical Ch 7 sprouted-seeds tower uses 2 (the default here);
   classic reference games often use 1 (pass :alpha 1.0)."
  [{:keys [denotation state-prior alpha infer]
    :or   {alpha 2.0 infer exact-marginal}}]
  (let [L0 (literal-listener {:denotation denotation :state-prior state-prior :infer infer})]
    {:L0 L0
     :S1 (speaker-table L0 alpha)
     :L1 (pragmatic-listener-table L0 state-prior alpha)}))

(defn table-row
  "Row `i` of a probability table as a vector of JS numbers."
  [table i]
  (vec (mx/->clj (mx/idx table i))))

;; -- Two denotations, same tower (the re-parameterized-denotation point) --

(def sprouted-denotation
  "Sprouted-seeds scalar implicature (agentmodels.org Ch 7). Utterances [all some none],
   states [0 1 2 3] = number of sprouted seeds. all=(s==3), some=(s>0), none=(s==0)."
  (mx/array #js [#js [0 0 0 1]      ; all  : s == 3
                 #js [0 1 1 1]      ; some : s  > 0
                 #js [1 0 0 0]]     ; none : s == 0
            mx/float32))

(def sprouted-prior (mx/array #js [0.25 0.25 0.25 0.25]))

(def reference-denotation
  "A minimal referential-implicature game. Utterances [u0 uboth], referents [r0 r1].
   u0 is true of r0 only; uboth is true of both. Different denotation, same tower —
   the pragmatic listener resolves the ambiguous 'uboth' toward r1."
  (mx/array #js [#js [1 0]          ; u0    : r0 only
                 #js [1 1]]         ; uboth : both
            mx/float32))

(def reference-prior (mx/array #js [0.5 0.5]))

;; ===========================================================================
;; Model 3 — Tic-tac-toe game tree (a softmax-action agent over a with-cache EU)
;;
;; Board = a flat 9-vector of #{:x :o :_} (index = 3*row + col). Everything about the
;; game is pure host data/combinatorics (the host-geometry side of the split); MLX
;; appears only in the final softmax-action policy distribution. The agent's value is
;;   val(board, mover, scorer) = utility(board, scorer)                   if terminal
;;                             = Σ_m π_mover(m) · val(child_m, other(mover), scorer)
;;   π_mover = softmax(α · [val(child_m, other(mover), mover)])    (argmax at α=##Inf)
;; i.e. the expectation of the scorer's value under the mover's own softmax policy —
;; soft minimax, collapsing to exact (adversarial) minimax in the α=##Inf limit.
;; ===========================================================================

(def ^:private lines
  [[0 1 2] [3 4 5] [6 7 8]    ; rows
   [0 3 6] [1 4 7] [2 5 8]    ; cols
   [0 4 8] [2 4 6]])          ; diagonals

(defn other-player [p] (if (= p :x) :o :x))

(defn won? [board p]
  (boolean (some (fn [[a b c]] (and (= p (board a)) (= p (board b)) (= p (board c)))) lines)))

(defn board-full? [board] (not-any? #{:_} board))

(defn terminal? [board] (or (won? board :x) (won? board :o) (board-full? board)))

(defn utility
  "Utility of a board to `player`: win 10, loss -10, draw / non-terminal 0
   (per the chapter: isDraw = neither has three-in-a-row)."
  [board player]
  (cond (won? board player)               10.0
        (won? board (other-player player)) -10.0
        :else                              0.0))

(defn legal-moves [board] (filterv #(= :_ (board %)) (range 9)))

(defn place [board i player] (assoc board i player))

;; Host-geometry side of the split: the value recursion and policy weights are pure
;; host arithmetic over the (tiny, combinatorial) game tree. MLX is reserved for the
;; final softmax-action policy distribution in make-game-agent — deliberately NOT
;; pushed down here, since the tree walk is irregular host control flow.

(defn- softmax-weights [alpha qs]
  (let [m  (apply max qs)
        es (mapv #(Math/exp (* alpha (- % m))) qs)
        z  (reduce + es)]
    (mapv #(/ % z) es)))

(defn- argmax-weights [qs]                ; α=##Inf: uniform over argmax ties
  (let [m  (apply max qs)
        is (mapv #(if (== % m) 1.0 0.0) qs)
        z  (reduce + is)]
    (mapv #(/ % z) is)))

(defn- policy-weights [alpha qs]
  (if (= alpha ##Inf) (argmax-weights qs) (softmax-weights alpha qs)))

(defn make-game-agent
  "Build the planning tic-tac-toe agent. Returns {:alpha :val :move-q :policy :act}.
   :policy is the softmax-action GF (retval = index into legal moves); :act simulates
   it host-side and returns the chosen cell. :move-q gives the per-move EU table.
   :val is the memoized (with-cache) host recursion val(board, mover, scorer).
   α defaults to ##Inf (exact/adversarial minimax — deterministic); finite α gives a
   soft-rational opponent. The full game tree from a near-terminal board is tiny;
   with-cache (the transposition table) keeps deeper boards tractable."
  [{:keys [alpha] :or {alpha ##Inf}}]
  ;; The atom (not letfn) lets the with-cache-wrapped val recurse into its OWN memoized
  ;; self: val is defined in terms of @val so deeper boards hit the transposition table.
  ;; Write-once at construction, then read-only — a knot-tie, not stateful recursion.
  (let [val (atom nil)]
    (reset! val
      (exact/with-cache
        (fn [board mover scorer]
          (if (terminal? board)
            (utility board scorer)
            (let [ms  (legal-moves board)
                  ;; the mover picks per a softmax over ITS OWN continuation value:
                  qs  (mapv (fn [m] (@val (place board m mover) (other-player mover) mover)) ms)
                  pol (policy-weights alpha qs)
                  ;; the scorer's value = expectation, under the mover's policy, of the
                  ;; scorer's own continuation value:
                  vs  (mapv (fn [m] (@val (place board m mover) (other-player mover) scorer)) ms)]
              (reduce + (map * pol vs)))))))
    (let [policy (fn [player]                    ; factor(α·EU) as a softmax-action GF; retval = INDEX into legal moves
                   (gen [board]
                     (let [eus (mapv (fn [m] (@val (place board m player) (other-player player) player))
                                     (legal-moves board))]
                       ;; raw traced index (NO in-body mx/item) — stays vectorization-safe,
                       ;; matching agent.cljs / biased_planners.cljs. The legal cell is
                       ;; resolved host-side in :act below.
                       (trace :move (h/softmax-action alpha (mx/array (clj->js eus)))))))]
      {:alpha alpha
       :val @val
       :move-q (fn [board player]               ; [[move eu]...] EU to `player` of each legal move
                 (mapv (fn [m] [m (@val (place board m player) (other-player player) player)])
                       (legal-moves board)))
       :policy policy
       :act (fn [board player]                  ; host-side: simulate the policy, resolve the chosen legal cell
              (let [ms (legal-moves board)
                    i  (Math/round (mx/item (:retval (p/simulate (dyn/auto-key (policy player)) [board]))))]
                (nth ms i)))})))

(defn best-move
  "The agent's argmax legal move for `player` at `board` (min cell index breaks ties —
   deterministic). Exact minimax at α=##Inf."
  [agent board player]
  (let [qs   ((:move-q agent) board player)
        best (apply max (map second qs))]
    (->> qs (filter #(>= (second %) (- best 1e-9))) (map first) (apply min))))

(defn non-planning-move-q
  "The one-step (no-lookahead) agent of the chapter: EU(move) = utility of the board
   immediately after the move, for `player`. Used to contrast with planning."
  [board player]
  (mapv (fn [m] [m (utility (place board m player) player)]) (legal-moves board)))

;; -- Reference boards used by the headless test (x to move; '_' = empty) --

(def forced-win-board
  "x has two-in-a-row on the top row; (0,2)=cell 2 completes it for an immediate win."
  [:x :x :_
   :o :o :_
   :_ :_ :_])

(def forced-block-board
  "Near-endgame, x to move (3 empty cells). o threatens to complete the middle row at
   cell 5. Blocking at cell 5 forces a DRAW (value 0); any other move lets o win
   (value -10). So PLANNING (lookahead) blocks at cell 5, while the one-step agent is
   indifferent (every immediate move is non-terminal → scores 0). The clean
   planning-vs-non-planning discriminator: undiscounted minimax can only tell block
   from non-block when the block actually salvages the game."
  [:x :x :o
   :o :o :_
   :x :_ :_])
