(ns genmlx.tabular
  "Vectorized deterministic primitives over enumerated state/hypothesis spaces.

   Two families (theory-neutral infrastructure, requested by arc3-solver —
   lab note interpreted-vs-compiled-gpu.md):

   1. Frontier-mask BFS over enumerated transition tables (genmlx-9ufc):
      given int32 T[s,a] -> s', reachability/shortest-path as iterated tensor
      ops — one argsort + searchsorted membership test per BFS level instead
      of per-node host traversal. Exact int32/bool semantics end-to-end (no
      float comparisons on state ids), parent/action arrays for path
      extraction, and batched multi-source runs fused into a single flat
      global-id space (row b's state s has global id b*S+s), so B independent
      searches cost one sort per level, not B.

      Design note: the natural formulation is a scatter (frontier[T] -> next
      mask), but the membrane exposes no scatter; successor membership is
      instead computed by sorting the active edge destinations (inactive
      edges routed to a sentinel that sorts last) and binary-searching every
      state id into that sorted list (searchsorted). The first hit per
      destination also yields a deterministic parent/action choice.

   2. Batched exact-match scoring (genmlx-8aqn): [N & dims] hypothesis grids
      vs an observed grid — per-hypothesis all-cells-equal masks, match
      counts, and fractions as [N] arrays, int32-exact, with an optional
      don't-care mask. Pure lazy graph ops; callers materialize.

   The GFI never appears here: these are Layer-8 supporting primitives, pure
   graph construction except the BFS level loop, which is an inference-style
   hot loop (per-level materialize! + one host sync on frontier emptiness)."
  (:refer-clojure :exclude [])
  (:require [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; Exact integer index construction
;; ---------------------------------------------------------------------------

(defn- iarange
  "Exact int32 [0..n). mx/arange builds float32, whose 24-bit mantissa
   silently corrupts ids above 2^24; cumsum over int32 ones stays exact for
   any id below 2^31."
  [n]
  (mx/subtract (mx/cumsum (mx/ones [n] mx/int32)) (mx/scalar 1 mx/int32)))

;; ---------------------------------------------------------------------------
;; Transition-table BFS (genmlx-9ufc)
;; ---------------------------------------------------------------------------

(defn ->table
  "Coerce a transition table to an int32 [S A] MLX array. Accepts nested data
   (rows = states, cols = actions — the exact int32 path) or an MLX array
   (astype'd; float32 inputs are only id-exact below 2^24, so pass int32 or
   nested data for large spaces). A 1-D [S] table is treated as [S 1]."
  [t]
  (let [arr (if (mx/array? t) (mx/astype t mx/int32) (mx/array t mx/int32))
        sh  (mx/shape arr)]
    (if (= 1 (count sh))
      (mx/reshape arr [(first sh) 1])
      arr)))

(defn- source-mask
  "Normalize sources to [mask-[B S] batched?]. A single id or a flat seq of
   ids is ONE (multi-source) run; a seq of id-seqs or a [B S] mask is B
   independent runs. An MLX [S] mask is one run."
  [sources S]
  (cond
    (mx/array? sources)
    (let [sh (mx/shape sources)]
      (case (count sh)
        1 [(mx/reshape (mx/astype sources mx/int32) [1 S]) false]
        2 [(mx/astype sources mx/int32) true]
        (throw (ex-info "bfs sources mask must be [S] or [B S]" {:shape sh}))))

    (number? sources)
    (source-mask [sources] S)

    (and (sequential? sources) (every? number? sources))
    (let [flat (make-array S)]
      (dotimes [i S] (aset flat i 0))
      (doseq [id sources] (aset flat id 1))
      [(mx/array (vec flat) [1 S] mx/int32) false])

    (sequential? sources)
    (let [B    (count sources)
          flat (make-array (* B S))]
      (dotimes [i (* B S)] (aset flat i 0))
      (doseq [[b ids] (map-indexed vector sources)
              id ids]
        (aset flat (+ (* b S) id) 1))
      [(mx/array (vec flat) [B S] mx/int32) true])

    :else
    (throw (ex-info "bfs sources: id, seq of ids, seq of id-seqs, or mask"
                    {:sources sources}))))

(defn bfs
  "Frontier-mask BFS over a deterministic transition table.

   table   — [S A] int32 (or nested data): table[s,a] = successor state id,
             each in [0,S). Self-loops are fine.
   sources — a state id or flat seq of ids (one multi-source run), a seq of
             id-seqs or an MLX [B S] 0/1 mask (B independent runs in one
             fused pass), or an MLX [S] mask.
   opts    — {:max-depth d} cap on BFS levels (default S, the diameter bound).

   Returns int32 MLX arrays shaped [S] (or [B S] when batched):
     :dist      shortest distance from the source set (-1 = unreachable)
     :parent    predecessor state on a shortest path (-1 = source/unreachable)
     :action    action index taken FROM :parent to reach the state (-1 = none)
     :visited   0/1 reachability mask
   plus host ints/flags:
     :levels    BFS levels that discovered at least one new state
     :complete? true when the frontier exhausted (false = :max-depth cut off)

   Per level: one argsort over the B*S*A active-edge destinations, one
   searchsorted membership probe per state, O(1) gathers — and a single host
   sync on frontier emptiness. All state/parent/action values stay int32.
   Use extract-path (host boundary) to walk concrete paths."
  ([table sources] (bfs table sources {}))
  ([table sources {:keys [max-depth]}]
   (let [t         (->table table)
         [S A]     (mx/shape t)
         [mask batched?] (source-mask sources S)
         B         (first (mx/shape mask))
         E         (* S A)
         BS        (* B S)
         max-depth (or max-depth S)
         zero-i    (mx/scalar 0 mx/int32)
         one-i     (mx/scalar 1 mx/int32)
         sent-i    (mx/scalar BS mx/int32)
         ;; Static edge structure in the flat global-id space. Edge order is
         ;; e = b*S*A + s*A + a, matching the row-major [B S A] layout.
         t-flat    (mx/reshape t [E])
         src-loc   (mx/repeat-arr (iarange S) A 0)                       ;; [E]
         offsets   (mx/reshape (mx/multiply (iarange B) (mx/scalar S mx/int32))
                               [B 1])
         dest-g    (mx/reshape (mx/add offsets (mx/reshape t-flat [1 E])) [(* B E)])
         src-g     (mx/reshape (mx/add offsets (mx/reshape src-loc [1 E])) [(* B E)])
         act-e     (mx/tile (iarange A) [BS])                            ;; [B*E]
         queries   (iarange BS)
         neg1      (mx/astype (mx/full [BS] -1) mx/int32)
         pad-neg1  (mx/reshape (mx/scalar -1 mx/int32) [1])
         pad-sent  (mx/reshape sent-i [1])
         mask-flat (mx/reshape mask [BS])
         finalize  (fn [visited dist parent action levels complete?]
                     (let [as-2d  #(mx/reshape % [B S])
                           par-2d (as-2d parent)
                           ;; parents were recorded as global ids; localize
                           par-2d (mx/where (mx/greater-equal par-2d zero-i)
                                            (mx/subtract par-2d offsets)
                                            par-2d)
                           out    {:dist      (as-2d dist)
                                   :parent    par-2d
                                   :action    (as-2d action)
                                   :visited   (as-2d visited)
                                   :levels    levels
                                   :complete? complete?}]
                       (if batched?
                         out
                         (-> out
                             (update :dist mx/squeeze [0])
                             (update :parent mx/squeeze [0])
                             (update :action mx/squeeze [0])
                             (update :visited mx/squeeze [0])))))]
     (loop [frontier mask-flat
            visited  mask-flat
            dist     (mx/where mask-flat zero-i neg1)
            parent   neg1
            action   neg1
            level    1]
       (let [;; edge is active iff its source is on the frontier
             active   (mx/take-idx frontier src-g)
             ;; inactive edges go to the sentinel id BS, which sorts last
             cand     (mx/where active dest-g sent-i)
             order    (mx/argsort cand)
             sorted   (mx/take-idx cand order)
             ;; first occurrence of each state id among active destinations
             pos      (mx/searchsorted sorted queries)
             sortedp  (mx/concatenate [sorted pad-sent])
             hit      (mx/astype (mx/equal (mx/take-idx sortedp pos) queries)
                                 mx/int32)
             fresh    (mx/multiply hit (mx/subtract one-i visited))
             ;; the same first-occurrence position names a shortest parent edge
             par-cand (mx/take-idx (mx/concatenate [(mx/take-idx src-g order)
                                                    pad-neg1]) pos)
             act-cand (mx/take-idx (mx/concatenate [(mx/take-idx act-e order)
                                                    pad-neg1]) pos)
             parent'  (mx/where fresh par-cand parent)
             action'  (mx/where fresh act-cand action)
             dist'    (mx/where fresh (mx/scalar level mx/int32) dist)
             visited' (mx/maximum visited fresh)]
         ;; hot loop: break graph accumulation + one host sync for termination
         (mx/materialize! fresh visited' dist' parent' action')
         (cond
           (zero? (mx/item (mx/sum fresh)))
           (finalize visited' dist' parent' action' (dec level) true)

           (>= level max-depth)
           (finalize visited' dist' parent' action' level false)

           :else
           (recur fresh visited' dist' parent' action' (inc level))))))))

(defn bfs->host
  "Pull a bfs result across the host boundary once: :dist/:parent/:action/
   :visited become nested ClojureScript vectors. Do this before repeated
   extract-path calls."
  [result]
  (-> result
      (update :dist mx/->clj)
      (update :parent mx/->clj)
      (update :action mx/->clj)
      (update :visited mx/->clj)))

(defn extract-path
  "Walk the parent/action arrays from a source to `target` (host boundary).
   result — a bfs result (MLX or bfs->host); pass batch row `b` for batched
   runs. Returns {:states [s0 .. target] :actions [a1 .. an]} with s0 a
   source and (count actions) = (count states) - 1, or nil when target is
   unreachable."
  ([result target] (extract-path result nil target))
  ([result b target]
   (let [row  (fn [x] (let [v (if (mx/array? x) (mx/->clj x) x)]
                        (if b (nth v b) v)))
         dist   (row (:dist result))
         parent (row (:parent result))
         action (row (:action result))]
     (when-not (neg? (nth dist target))
       (loop [s target, states (list target), actions '(), steps 0]
         (when (<= steps (count dist)) ;; corrupt-input guard, not a semantic
           (let [p (nth parent s)]
             (if (neg? p)
               {:states (vec states) :actions (vec actions)}
               (recur p (cons p states) (cons (nth action s) actions)
                      (inc steps))))))))))

;; ---------------------------------------------------------------------------
;; Batched exact-match scoring (genmlx-8aqn)
;; ---------------------------------------------------------------------------

(defn- ->grid
  "Coerce hypothesis/observation grids to int32 MLX arrays. Comparison is
   integer-exact: this primitive is for id/label grids, not float payloads."
  [x]
  (if (mx/array? x) (mx/astype x mx/int32) (mx/array x mx/int32)))

(defn- flat-pair
  "Flatten batch [N & dims] and target ([& dims] or [N & dims]) to
   [batch-[N cells] target-[cells]-or-[N cells] n cells]."
  [batch target]
  (let [b  (->grid batch)
        t  (->grid target)
        [n & ds] (mx/shape b)
        cells (reduce * 1 ds)
        tsh   (mx/shape t)
        t'    (if (= (count tsh) (count ds))
                (mx/reshape t [cells])
                (mx/reshape t [n cells]))]
    [(mx/reshape b [n cells]) t' n cells]))

(defn- eq-cells
  "[N cells] int32 0/1 per-cell equality of batch vs (broadcast) target."
  [batch target]
  (let [[b t _ _] (flat-pair batch target)]
    (mx/astype (mx/equal b t) mx/int32)))

(defn match-count
  "Per-hypothesis count of matching cells: batch [N & dims] vs target
   ([& dims], broadcast over N — or [N & dims] for per-row targets).
   With ignore-mask (same dims as one grid; 0 = don't-care), only cells
   where the mask is 1 are counted. Returns [N] int32. Lazy."
  ([batch target]
   (mx/sum (eq-cells batch target) [1]))
  ([batch target ignore-mask]
   (let [[_ _ _ cells] (flat-pair batch target)
         m (mx/reshape (->grid ignore-mask) [1 cells])]
     (mx/sum (mx/multiply (eq-cells batch target) m) [1]))))

(defn match-mask
  "Per-hypothesis exact match: 1 iff every (non-ignored) cell equals the
   target. Same shapes as match-count. Returns [N] int32 0/1. Lazy."
  ([batch target]
   (mx/astype (mx/all (eq-cells batch target) 1) mx/int32))
  ([batch target ignore-mask]
   (let [[_ _ _ cells] (flat-pair batch target)
         m  (mx/reshape (->grid ignore-mask) [1 cells])
         ok (mx/maximum (eq-cells batch target)
                        (mx/subtract (mx/scalar 1 mx/int32) m))]
     (mx/astype (mx/all ok 1) mx/int32))))

(defn match-frac
  "Per-hypothesis fraction of matching cells among compared (non-ignored)
   cells. Returns [N] float32. Lazy."
  ([batch target]
   (let [[_ _ _ cells] (flat-pair batch target)]
     (mx/divide (mx/astype (match-count batch target) mx/float32)
                (mx/scalar cells))))
  ([batch target ignore-mask]
   (let [[_ _ _ cells] (flat-pair batch target)
         m     (mx/reshape (->grid ignore-mask) [1 cells])
         denom (mx/maximum (mx/astype (mx/sum m [1]) mx/float32)
                           (mx/scalar 1))]
     (mx/divide (mx/astype (match-count batch target ignore-mask) mx/float32)
                denom))))

(defn matching-indices
  "Indices of exactly-matching hypotheses as a ClojureScript vector.
   HOST BOUNDARY: evaluates the match mask and reads it back."
  ([batch target]
   (vec (keep-indexed (fn [i v] (when (pos? v) i))
                      (mx/->clj (match-mask batch target)))))
  ([batch target ignore-mask]
   (vec (keep-indexed (fn [i v] (when (pos? v) i))
                      (mx/->clj (match-mask batch target ignore-mask))))))
