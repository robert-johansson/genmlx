;; @tier fast
(ns genmlx.mix-regen-general-test
  "genmlx-175y pin: fast vs forced-general regenerate on a spliced Mix.

   The uizc wrapper shape (a gen wrapper splicing a Mix over two gaussian
   kernels, with a downstream site consuming the Mix retval) under the
   selection [:mx0 :component-idx] exposed two defects:

     1. KEY THREADING: Mix derived NO entropy from its threaded splice key —
        the index resample self-seeded via js/Math.random and the flip-branch
        p/simulate via the auto-key sentinel — so two same-key regenerate
        calls flipped components INDEPENDENTLY and the fast/general law's
        same-key premise was unimplementable on Mix.
     2. SOUNDNESS (general path): a component flip freshly draws the WHOLE
        new component at the SAME inner addresses both kernels share, so
        regen-retained-selection (address presence in both traces) wrongly
        counted [:mx0 :z] retained and the two project passes added a
        spurious lp(z_new; new comp) - lp(z_old; old comp) — a term the
        thesis says cancels for fresh sites. The FAST path was thesis-exact
        all along.

   This file pins, per seed, on both flip and stay moves:
     (a) fast weight == the hand-computed thesis weight
         lp(x1; z_new) - lp(x1; z_old)  (identically 0 on a stay);
     (b) forced-general weight == the same hand value — THE assertion that
         kills the spurious z-term (fresh-path provenance tag, genmlx-175y);
     (c) fast and general with the SAME key produce the IDENTICAL new trace
         (pins splice-key-derived Mix entropy);
   plus same-key determinism of Mix simulate/generate through the wrapper."
  (:require [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private fails (atom 0))
(def ^:private passes (atom 0))

(defn- assert-true [desc x]
  (if x (swap! passes inc)
        (do (swap! fails inc) (println "  FAIL" desc))))

;; The uizc corpus shape (pef_corpus.cljs), built with the gen macro. Kernels
;; are auto-keyed like pef/source->model kernels (and mixture kernels in
;; practice): Mix's own sampling entropy now derives from its splice key, but
;; the general path's project passes replay the CURRENT component via
;; p/project, whose DynamicGF entry demands a key even though projection
;; never samples (pre-existing; out of genmlx-175y scope).
(def kern1 (dyn/auto-key (gen [] (trace :z (dist/gaussian -2 0.5)))))
(def kern2 (dyn/auto-key (gen [] (trace :z (dist/gaussian 2 0.5)))))
(def mixed (comb/mix-combinator [kern1 kern2] (mx/array [-0.7 -0.7])))
(def model (gen [] (let [v0 (splice :mx0 mixed)
                         v1 (trace :x1 (dist/gaussian v0 1.0))]
                     v1)))

(def ^:private idx-sel (sel/from-paths [[:mx0 :component-idx]]))

(defn- ch [trace path] (mx/item (cm/get-choice (:choices trace) path)))

(defn- log-normal-pdf [x mu sd]
  (- (* -0.5 (js/Math.log (* 2 js/Math.PI sd sd)))
     (/ (* (- x mu) (- x mu)) (* 2 sd sd))))

(println "\n-- same-key determinism: Mix entropy derives from the splice key --")
;; Before genmlx-175y, Mix self-seeded and two same-key simulates diverged
;; at the Mix sites. (Self-seeding was the bug: nothing could legitimately
;; pin the old behavior, since it was irreproducible by construction.)
(let [k (rng/fresh-key 424)
      t1 (p/simulate (dyn/with-key model k) [])
      t2 (p/simulate (dyn/with-key model k) [])]
  (assert-true "simulate: same key => same :component-idx"
               (= (ch t1 [:mx0 :component-idx]) (ch t2 [:mx0 :component-idx])))
  (assert-true "simulate: same key => same :z"
               (= (ch t1 [:mx0 :z]) (ch t2 [:mx0 :z])))
  (assert-true "simulate: same key => same :x1"
               (= (ch t1 [:x1]) (ch t2 [:x1]))))
(let [k (rng/fresh-key 425)
      g1 (:trace (p/generate (dyn/with-key model k) [] cm/EMPTY))
      g2 (:trace (p/generate (dyn/with-key model k) [] cm/EMPTY))]
  (assert-true "generate: same key => same :component-idx"
               (= (ch g1 [:mx0 :component-idx]) (ch g2 [:mx0 :component-idx])))
  (assert-true "generate: same key => same :z"
               (= (ch g1 [:mx0 :z]) (ch g2 [:mx0 :z]))))

(println "\n-- fast == general == hand-computed on [:mx0 :component-idx] --")
;; Iterate seeds until both a STAY and a FLIP are observed (bounded at 20).
;; For every seed assert (a) fast == hand, (b) general == hand, (c) fast and
;; general new traces are identical under the same key.
(let [tol 1e-3
      run-seed
      (fn [seed]
        (let [t (p/simulate (dyn/with-key model (rng/fresh-key (* seed 7))) [])
              mk #(dyn/with-key model (rng/fresh-key seed))
              fast (p/regenerate (mk) t idx-sel)
              general (binding [dyn/*force-general-regen* true]
                        (p/regenerate (mk) t idx-sel))
              tf (:trace fast)
              tg (:trace general)
              old-idx (ch t [:mx0 :component-idx])
              new-idx (ch tf [:mx0 :component-idx])
              z-old (ch t [:mx0 :z])
              z-new (ch tf [:mx0 :z])
              x1 (ch t [:x1])
              ;; thesis weight: :x1 is the ONLY retained-and-context-moved
              ;; site — fresh z (flip) and the selected index cancel exactly
              hand (- (log-normal-pdf x1 z-new 1.0)
                      (log-normal-pdf x1 z-old 1.0))
              wf (mx/item (:weight fast))
              wg (mx/item (:weight general))]
          {:seed seed
           :flip? (not= old-idx new-idx)
           :fast-hand? (< (js/Math.abs (- wf hand)) tol)
           :general-hand? (< (js/Math.abs (- wg hand)) tol)
           :x1-retained? (= x1 (ch tf [:x1]))
           :same-trace? (and (= new-idx (ch tg [:mx0 :component-idx]))
                             (< (js/Math.abs (- z-new (ch tg [:mx0 :z]))) 1e-6)
                             (< (js/Math.abs (- (ch tf [:x1]) (ch tg [:x1]))) 1e-6))
           :stay-zero? (or (not= old-idx new-idx)
                           (< (js/Math.abs wf) tol))
           :wf wf :wg wg :hand hand}))
      results (mapv run-seed (range 1 21))
      flips (filterv :flip? results)
      stays (filterv (complement :flip?) results)
      bad (fn [k] (mapv #(select-keys % [:seed :flip? :wf :wg :hand])
                        (remove k results)))]
  (println "   " (count flips) "flips /" (count stays) "stays over 20 seeds")
  (assert-true "both a stay and a flip observed within 20 seeds"
               (and (seq flips) (seq stays)))
  (assert-true (str "(a) fast == hand-computed lp(x1;z_new)-lp(x1;z_old), all seeds "
                    (pr-str (bad :fast-hand?)))
               (every? :fast-hand? results))
  (assert-true (str "(b) forced-general == the same hand value (no spurious z-term), all seeds "
                    (pr-str (bad :general-hand?)))
               (every? :general-hand? results))
  (assert-true "(c) fast and general with the same key produce the identical new trace"
               (every? :same-trace? results))
  (assert-true ":x1 is retained (never resampled) on this selection"
               (every? :x1-retained? results))
  (assert-true "stay moves have weight ~0"
               (every? :stay-zero? results)))

;; --------------------------------------------------------------------------
;; genmlx-gxrq pins (Mix follow-ups from 175y)
;; --------------------------------------------------------------------------
;; (2) IProject on KEYLESS components: projection never samples, so a Mix over
;; unkeyed kernels must project without the old 'No PRNG key' throw (the Mix
;; now keys the replayed component from its splice key).
(let [k1  (gen [] (trace :z (dist/gaussian -2 0.5)))
      k2  (gen [] (trace :z (dist/gaussian 2 0.5)))
      m   (comb/mix-combinator [k1 k2] (mx/array [-0.7 -0.7]))
      tr* (p/simulate (dyn/with-key m (rng/fresh-key 77)) [])]
  (assert-true "gxrq(2): project over keyless components does not throw"
               (number? (mx/item (p/project m tr* (sel/from-paths [[:component-idx]]))))))

;; (3) PINNED SEMANTICS: the splice-derived k-comp OVERRIDES a component's
;; construction-time fixed key (the unfold-extend convention) — a fixed-key
;; component does NOT repeat its draws across simulates of an unkeyed Mix;
;; determinism is obtained by keying the MIX (covered above).
(let [fixed (vary-meta (gen [] (trace :z (dist/gaussian 0 1)))
                       assoc :genmlx.dynamic/key (rng/fresh-key 123))
      m     (comb/mix-combinator [fixed] (mx/array [0.0]))
      zs    (mapv (fn [_] (mx/item (cm/get-choice (:choices (p/simulate m []))
                                                  [:z])))
                  (range 5))]
  (assert-true "gxrq(3): splice-derived key overrides the component's fixed key"
               (> (count (distinct zs)) 1)))

(println (str "\n== mix-regen-general: " @passes " passed, " @fails " failed =="))
(when (pos? @fails) (set! (.-exitCode js/process) 1))
