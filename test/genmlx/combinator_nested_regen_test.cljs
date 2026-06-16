;; @tier fast core
(ns genmlx.combinator-nested-regen-test
  "genmlx-7opt: mirror of the genmlx-nt0c vmap fix for Map/Unfold/Scan.

   The general retained-only regenerate/update weight is project-based
   (W = Σ_retained Δlp), NOT linear in the element :score. A combinator
   reconstructs each element's old kernel trace with `tr/make-trace` from the
   stored per-element score. When the kernel is ITSELF a combinator, that
   reconstructed inner trace must ALSO carry its own per-element metadata
   (::element-scores / ::step-scores), or the inner combinator falls into its
   metadata-loss branch and mis-weights by the whole inner old score.

   Map is the only reachable buggy OUTER: its kernel can be any combinator
   (Map-of-Map / Map-of-Unfold / Map-of-Scan are constructible). Unfold's
   ([t state]->state) and Scan's ([carry input]->[carry output]) kernel
   signatures cannot take a combinator directly, so the nested-via-direct-kernel
   bug is structurally unreachable for them (the splice-of-combinator case is
   the separate genmlx-v740/v5cs path).

   ORACLE (thesis, non-circular): regenerate weight must equal
   project(new, retained) − project(old, retained), computed via the
   combinator's OWN project (a different code path than regenerate's re-base)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.gfi :as gfi]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- num [x] (mx/eval! x) (mx/item x))
(defn- close? [a b tol] (< (js/Math.abs (- a b)) tol))

;; A 3-site chain a→b→c. Selecting {:a :b} forces the GENERAL retained-only
;; path: :a (selected) feeds :b (selected) so it is not fast-eligible, and :c
;; is a dependent RETAINED site whose density shifts when :b is resampled.
(def chain-kernel
  (dyn/auto-key (gen [m] (let [a (trace :a (dist/gaussian m 1))
                               b (trace :b (dist/gaussian a 1))
                               c (trace :c (dist/gaussian b 1))]
                           c))))

(def sel-ab (sel/select :a :b))
(def retained (sel/select :c))

(defn- delta-project [gf old-tr new-tr]
  (mx/subtract (p/project gf new-tr retained)
               (p/project gf old-tr retained)))

(deftest leaf-kernel-forces-general-path
  ;; ANTI-VACUITY: if the leaf selection were fast-eligible, w would be linear
  ;; in the element score and the test would pass even with the bug present.
  (is (not (#'dyn/regen-fast-eligible? chain-kernel sel-ab))
      "chain kernel + {:a :b} must force the general (non-fast) regenerate path"))

(deftest map-single-level-regenerate-weight
  ;; Sanity: single-level Map general path is already correct (the -(:score)
  ;; re-base + the leaf's project-based weight handle it).
  (let [v  (comb/map-combinator chain-kernel)
        tr (p/simulate (dyn/with-key v (rng/fresh-key 7)) [[0.0 1.0 2.0]])
        {new-tr :trace w :weight} (p/regenerate v tr sel-ab)]
    (is (js/isFinite (num w)) "single-level Map weight finite")
    (is (close? (num (delta-project v tr new-tr)) (num w) 1e-3)
        "single-level Map weight = Δproject(retained)")))

(deftest map-of-map-regenerate-weight
  ;; THE BUG: outer Map reconstructs the inner Map old-trace WITHOUT its
  ;; ::element-scores, so the inner Map hits its metadata-loss branch and
  ;; mis-weights by the whole inner old score.
  (let [inner (comb/map-combinator chain-kernel)
        outer (comb/map-combinator inner)
        tr (p/simulate (dyn/with-key outer (rng/fresh-key 11))
                       [[[0.0 1.0 2.0] [3.0 4.0 5.0]]])
        {new-tr :trace w :weight} (p/regenerate outer tr sel-ab)]
    (is (js/isFinite (num w)) "Map-of-Map weight finite")
    (is (close? (num (delta-project outer tr new-tr)) (num w) 1e-3)
        "Map-of-Map weight = Δproject(retained)")))

(deftest map-of-map-update-weight
  ;; Same reconstruction is used in update; an empty-constraint update must
  ;; have weight 0 (nothing changes) even nested.
  (let [inner (comb/map-combinator chain-kernel)
        outer (comb/map-combinator inner)
        tr (p/simulate (dyn/with-key outer (rng/fresh-key 13))
                       [[[0.0 1.0 2.0] [3.0 4.0 5.0]]])
        {w :weight} (p/update outer tr cm/EMPTY)]
    (is (close? (num w) 0.0 1e-3)
        "Map-of-Map empty update weight = 0")))

;; ── Hetero-nesting: Map whose inner kernel is an Unfold / Scan ─────────────
;; Confirms the fix captures ::step-scores too (not only ::element-scores).

(def step-kernel
  ;; Unfold/Scan step with an internal a→b chain so {:a :b} again forces the
  ;; general retained-only path inside each step.
  (dyn/auto-key (gen [t prev] (let [a (trace :a (dist/gaussian prev 1))
                                    b (trace :b (dist/gaussian a 1))]
                                (trace :c (dist/gaussian b 1))
                                b))))

(deftest map-of-unfold-regenerate-weight
  (let [inner (comb/unfold-combinator step-kernel)
        outer (comb/map-combinator inner)
        ;; outer args: two parallel vectors [n_i] and [init-state_i]
        tr (p/simulate (dyn/with-key outer (rng/fresh-key 17))
                       [[2 2] [(mx/scalar 0.0) (mx/scalar 3.0)]])
        {new-tr :trace w :weight} (p/regenerate outer tr sel-ab)]
    (is (js/isFinite (num w)) "Map-of-Unfold weight finite")
    (is (close? (num (delta-project outer tr new-tr)) (num w) 1e-3)
        "Map-of-Unfold weight = Δproject(retained)")))

(def scan-kernel
  ;; Scan step [carry input] -> [new-carry output] with internal a→b chain.
  (dyn/auto-key (gen [carry input]
                     (let [a (trace :a (dist/gaussian carry 1))
                           b (trace :b (dist/gaussian a 1))]
                       (trace :c (dist/gaussian (mx/add b input) 1))
                       [b a]))))

(deftest map-of-scan-regenerate-weight
  (let [inner (comb/scan-combinator scan-kernel)
        outer (comb/map-combinator inner)
        ;; outer args: parallel vectors of init-carry and inputs-seq, one per element
        tr (p/simulate (dyn/with-key outer (rng/fresh-key 19))
                       [[(mx/scalar 0.0) (mx/scalar 1.0)]
                        [[(mx/scalar 0.5) (mx/scalar 1.5)]
                         [(mx/scalar 2.5) (mx/scalar 3.5)]]])
        {new-tr :trace w :weight} (p/regenerate outer tr sel-ab)]
    (is (js/isFinite (num w)) "Map-of-Scan weight finite")
    (is (close? (num (delta-project outer tr new-tr)) (num w) 1e-3)
        "Map-of-Scan weight = Δproject(retained)")))

;; ── Handler-forced leaf: rule out the compiled regenerate path masking the
;;    suspected handler-fallback bug. strip-compiled forces the chain kernel's
;;    handler (interpreter) path, so the inner Map cannot take cops/get-compiled
;;    -regenerate and MUST use the make-trace reconstruction fallback. ─────────
(def chain-handler (dyn/auto-key (gfi/strip-compiled chain-kernel)))

(deftest handler-forced-leaf-is-general-path
  (is (not (#'dyn/regen-fast-eligible? chain-handler sel-ab))
      "handler-forced chain + {:a :b} must still force the general path"))

(deftest map-of-map-handler-path-regenerate-weight
  (let [inner (comb/map-combinator chain-handler)
        outer (comb/map-combinator inner)
        tr (p/simulate (dyn/with-key outer (rng/fresh-key 23))
                       [[[0.0 1.0 2.0] [3.0 4.0 5.0]]])
        {new-tr :trace w :weight} (p/regenerate outer tr sel-ab)]
    (is (js/isFinite (num w)) "Map-of-Map (handler path) weight finite")
    (is (close? (num (delta-project outer tr new-tr)) (num w) 1e-3)
        "Map-of-Map (handler path) weight = Δproject(retained)")))

(deftest map-of-map-metadata-loss-regenerate-weight
  ;; Splice/serialize loss: strip the OUTER trace metadata so even
  ;; ::element-scores is gone. The -(:score trace) re-base must recover exactly.
  (let [inner (comb/map-combinator chain-handler)
        outer (comb/map-combinator inner)
        tr0 (p/simulate (dyn/with-key outer (rng/fresh-key 29))
                        [[[0.0 1.0 2.0] [3.0 4.0 5.0]]])
        tr (with-meta tr0 (dissoc (meta tr0) :genmlx.combinators/element-scores))
        {new-tr :trace w :weight} (p/regenerate outer tr sel-ab)]
    (is (js/isFinite (num w)) "Map-of-Map (meta-loss) weight finite")
    (is (close? (num (delta-project outer tr0 new-tr)) (num w) 1e-3)
        "Map-of-Map (meta-loss) weight = Δproject(retained)")))

(cljs.test/run-tests)
