;; @tier medium
(ns genmlx.control-cybernetic-test
  "Laws for genmlx.control.cybernetic (genmlx-cyst): the comparator/TOTE
   controlled-steppable, its recorded stop sites, and the replay-gf inversion.
   Ports the MCT project's validation stack as laws: shape/transparency,
   replay-inversion identity, a miniature grid recovery (their E1), and the
   product-form identifiability ridge + do()-collapse (their E4 pilot — the
   house theorem at the control axis). Synthetic hazards only, no LLM."
  (:require [genmlx.control.cybernetic :as cyb]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]))

(def ^:private fails (atom 0))
(defn assert-true [msg pred] (if pred (println (str "  [PASS] " msg))
                                 (do (swap! fails inc) (println (str "  [FAIL] " msg)))))
(defn assert-close [msg a b tol]
  (let [ok (< (js/Math.abs (- a b)) tol)]
    (if ok (println (str "  [PASS] " msg))
        (do (swap! fails inc)
            (println (str "  [FAIL] " msg " — expected " a " got " b " (tol " tol ")"))))))

;; ---------------------------------------------------------------------------
;; helpers
;; ---------------------------------------------------------------------------

(defn counter-base
  "The simplest possible substrate: count to n-max."
  [n-max]
  {:init (fn [] {:n 0})
   :step (fn [s] (update s :n inc))
   :done? (fn [s] (>= (:n s) n-max))
   :best (fn [s] (:n s))})

(defn drive
  "Run a steppable to completion, returning its :best."
  [st]
  (loop [s ((:init st))]
    (if ((:done? st) s) ((:best st) s) (recur ((:step st) s)))))

(def ^:private LN js/Math.log)
(defn- clip01 [p] (-> p (max 1e-6) (min (- 1.0 1e-6))))
(defn- site-ll [site] (if (= 1 (:bit site))
                        (LN (:hazard site))
                        (LN (- 1.0 (:hazard site)))))

;; ---------------------------------------------------------------------------
;; 1. shape / transparency laws
;; ---------------------------------------------------------------------------
(println "\n== shape laws: the wrapper is transparent until the controller fires ==")

(let [out (drive (cyb/tote-controlled-steppable
                  (counter-base 20)
                  {:monitor (constantly 0.0) :tau 0.5
                   :gate-hazard (constantly 1.0)
                   :key (rng/fresh-key 1)}))]
  (assert-true "never-arming monitor: base runs to its own done? (result = 20)"
               (= 20 (:result out)))
  (assert-true "never-arming monitor: exit :censored" (= :censored (:exit out)))
  (assert-true "never-arming monitor: zero stop sites" (empty? (:sites out))))

(let [out (drive (cyb/tote-controlled-steppable
                  (counter-base 20)
                  {:monitor (constantly 1.0) :tau 0.5
                   :gate-hazard (constantly 1.0)
                   :key (rng/fresh-key 2)}))]
  (assert-true "gate-hazard 1.0: stops at the FIRST proposal (t = 0, no base advance)"
               (and (= :goal (:exit out)) (= 0 (:t out)) (= 0 (:result out))))
  (assert-true "gate-hazard 1.0: exactly one :gate site with bit 1"
               (and (= 1 (count (:sites out)))
                    (= :gate (:kind (first (:sites out))))
                    (= 1 (:bit (first (:sites out)))))))

(let [out (drive (cyb/tote-controlled-steppable
                  (counter-base 20)
                  {:monitor (constantly 1.0) :tau 0.5 :min-engage 3
                   :gate-hazard (constantly 1.0)
                   :key (rng/fresh-key 3)}))]
  (assert-true "min-engage 3: first gate decision only after 3 base advances"
               (and (= :goal (:exit out)) (= 3 (:t out)) (= 3 (:result out)))))

;; ---------------------------------------------------------------------------
;; 2. hysteresis + min-gap arming discipline
;; ---------------------------------------------------------------------------
(println "\n== arming discipline: hysteresis and min-gap ==")

;; gate-hazard ~0 => every decision is a veto; hysteresis means the comparator
;; never re-arms (m never falls back below tau) => exactly ONE gate site ever.
(let [out (drive (cyb/tote-controlled-steppable
                  (counter-base 10)
                  {:monitor (constantly 1.0) :tau 0.5
                   :gate-hazard (constantly 0.0)
                   :hysteresis? true
                   :key (rng/fresh-key 4)}))]
  (assert-true "hysteresis: one veto, then never re-arms (m never re-crosses tau)"
               (and (= :censored (:exit out)) (= 1 (count (:sites out)))
                    (= 0 (:bit (first (:sites out)))))))

;; without hysteresis, min-gap 2 paces the tests: sites at t = 0,2,4,6,8.
(let [out (drive (cyb/tote-controlled-steppable
                  (counter-base 10)
                  {:monitor (constantly 1.0) :tau 0.5
                   :gate-hazard (constantly 0.0)
                   :hysteresis? false :min-gap 2
                   :key (rng/fresh-key 5)}))]
  (assert-true "no hysteresis + min-gap 2: gate tests at t = 0,2,4,6,8 (all vetoes)"
               (and (= :censored (:exit out))
                    (= [0 2 4 6 8] (mapv :t (:sites out)))
                    (every? #(zero? (:bit %)) (:sites out)))))

(assert-true "livelock guard: :hysteresis? false + :min-gap 0 is rejected"
             (try (cyb/tote-controlled-steppable
                   (counter-base 5)
                   {:monitor (constantly 1.0) :tau 0.5
                    :gate-hazard (constantly 0.5)
                    :hysteresis? false :min-gap 0})
                  false
                  (catch :default _ true)))

;; ---------------------------------------------------------------------------
;; 3. the code register: decay clock, decay exit, veto refresh
;; ---------------------------------------------------------------------------
(println "\n== the code register: decay exit + veto refresh ==")

;; lambda 0, eps 1: at the first decay boundary (t=2) r -> 0, hazard -> ~1 => exit.
(let [out (drive (cyb/tote-controlled-steppable
                  (counter-base 20)
                  {:monitor (constantly 0.0) :tau 0.5
                   :gate-hazard (constantly 1.0)
                   :register {:lambda 0.0 :every 2 :eps 1.0}
                   :key (rng/fresh-key 6)}))]
  (assert-true "decay exit: lambda 0 / eps 1 dissolves at the first boundary (t = 2)"
               (and (= :decay (:exit out)) (= 2 (:t out))
                    (= [:decay] (mapv :kind (:sites out)))
                    (= 1 (:bit (first (:sites out)))))))

;; veto refresh: r decays to 0.25 by t=4, the veto at t=6 refreshes it to 1.0.
(let [st (cyb/tote-controlled-steppable
          (counter-base 20)
          {:monitor (fn [s] (if (= 6 (:n s)) 1.0 0.0)) :tau 0.5
           :gate-hazard (constantly 0.0)
           :register {:lambda 0.5 :every 2 :eps 0.0}
           :key (rng/fresh-key 7)})
      upto-veto (loop [s ((:init st))]
                  (if (or ((:done? st) s)
                          (some #(= :gate (:kind %)) (:sites s)))
                    s (recur ((:step st) s))))]
  (assert-close "the veto refreshed the (0.25-decayed) register back to 1.0"
                1.0 (:r upto-veto) 1e-9)
  (assert-true "the register had decayed first (decay sites at t=2,4 precede the veto)"
               (= [:decay :decay :gate] (mapv :kind (:sites upto-veto)))))

;; refresh-on-veto? false: the veto leaves r decayed.
(let [st (cyb/tote-controlled-steppable
          (counter-base 20)
          {:monitor (fn [s] (if (= 6 (:n s)) 1.0 0.0)) :tau 0.5
           :gate-hazard (constantly 0.0)
           :register {:lambda 0.5 :every 2 :eps 0.0 :refresh-on-veto? false}
           :key (rng/fresh-key 8)})
      upto-veto (loop [s ((:init st))]
                  (if (or ((:done? st) s)
                          (some #(= :gate (:kind %)) (:sites s)))
                    s (recur ((:step st) s))))]
  (assert-close ":refresh-on-veto? false — the veto leaves r at 0.25"
                0.25 (:r upto-veto) 1e-9))

;; ---------------------------------------------------------------------------
;; 4. scheduled sites (the do()-operator)
;; ---------------------------------------------------------------------------
(println "\n== scheduled instructed trials (do()-operator) ==")

(let [out (drive (cyb/tote-controlled-steppable
                  (counter-base 20)
                  {:monitor (constantly 0.0) :tau 0.5
                   :gate-hazard (constantly 1.0)
                   :schedule [{:t 3 :hazard (constantly 1.0)}]
                   :key (rng/fresh-key 9)}))]
  (assert-true "scheduled trial fires at t = 3, bypassing the (never-armed) comparator"
               (and (= :scheduled (:exit out)) (= 3 (:t out))
                    (= [:scheduled] (mapv :kind (:sites out))))))

(let [out (drive (cyb/tote-controlled-steppable
                  (counter-base 20)
                  {:monitor (constantly 0.0) :tau 0.5
                   :gate-hazard (constantly 1.0)
                   :schedule [{:t 3 :hazard (constantly 0.0)}]
                   :key (rng/fresh-key 10)}))]
  (assert-true "failed scheduled trial: recorded (bit 0), queue consumed, run continues"
               (and (= :censored (:exit out))
                    (= 1 (count (:sites out)))
                    (= 0 (:bit (first (:sites out)))))))

;; ---------------------------------------------------------------------------
;; 5. replay-inversion identity: p/generate weight == hand-computed site ll
;; ---------------------------------------------------------------------------
(println "\n== replay-inversion identity (the ADR-0002 'traced stop' law) ==")

(let [st (cyb/tote-controlled-steppable
          (counter-base 200)
          {:monitor (constantly 1.0) :tau 0.5
           :hysteresis? false :min-gap 2 :min-engage 2
           :gate-hazard (constantly 0.2)
           :register {:lambda 0.9 :every 3 :eps 0.2}
           :schedule [{:t 4 :hazard (constantly 0.3)}]
           :key (rng/fresh-key 42)})
      episode (drive st)
      sites (:sites episode)
      hand (reduce + 0.0 (map site-ll sites))
      replayed (cyb/episode-log-lik sites cyb/recorded-hazard nil)]
  (assert-true "the episode produced a mixed site record (gate + decay + scheduled)"
               (and (seq sites) (>= (count (set (map :kind sites))) 2)))
  (assert-close (str "p/generate replay weight == hand-computed site log-lik ("
                     (count sites) " sites)")
                hand replayed 1e-3)
  (assert-close "empty site record scores 0.0"
                0.0 (cyb/episode-log-lik [] cyb/recorded-hazard nil) 1e-12))

;; replay-gf is a full generative citizen: p/simulate draws fresh trajectories.
(let [sites [{:kind :gate :t 0 :hazard 0.5 :bit 1}
             {:kind :gate :t 1 :hazard 0.5 :bit 0}]
      tr (p/simulate (cyb/replay-gf sites cyb/recorded-hazard) [nil])]
  (assert-true "p/simulate on replay-gf runs and returns a stop record"
               (contains? (:retval tr) :stopped?)))

;; ---------------------------------------------------------------------------
;; 6. grid recovery of a planted gate hazard (miniature of mct E1)
;; ---------------------------------------------------------------------------
(println "\n== 1D grid recovery of a planted hazard ==")

(defn gen-episodes
  "n episodes of the given controller opts over a counter base; returns site
   records (one vector per episode)."
  [n opts]
  (mapv (fn [i]
          (:sites (drive (cyb/tote-controlled-steppable
                          (counter-base 500)
                          (assoc opts :key (rng/fresh-key (+ 1000 i)))))))
        (range n)))

(let [p* 0.3
      episodes (gen-episodes 40 {:monitor (constantly 1.0) :tau 0.5
                                 :hysteresis? false :min-gap 1
                                 :gate-hazard (constantly p*)})
      grid (mapv #(/ % 100) (range 5 100 5))
      lls (mapv (fn [pp]
                  (let [ll (reduce + 0.0
                                   (map #(cyb/episode-log-lik % (fn [_ v] v) pp)
                                        episodes))]
                    (mx/sweep-dead-arrays!)
                    ll))
                grid)
      best (nth grid (first (apply max-key second (map-indexed vector lls))))]
  (assert-true (str "pooled replay posterior mode recovers p* = 0.3 (got " best ")")
               (<= (js/Math.abs (- best p*)) 0.1)))

;; ---------------------------------------------------------------------------
;; 7. THE PRODUCT-FORM RIDGE (the house theorem at the control axis):
;;    gate hazard (1-a)*b constrains only the product on passive episodes;
;;    scheduled sites with hazard b separate the pair (the do()-collapse).
;; ---------------------------------------------------------------------------
(println "\n== the product-form identifiability ridge + do()-collapse ==")

(defn- suff-stats [episodes]
  (reduce (fn [acc site]
            (update-in acc [(:kind site) (:bit site)] (fnil inc 0)))
          {} (apply concat episodes)))

(defn- pooled-ll
  "Analytic pooled log-lik from sufficient statistics (bernoulli factorizes);
   spot-checked against the GFI replay below."
  [{gate :gate sched :scheduled} a b]
  (let [pg (clip01 (* (- 1.0 a) b))
        ps (clip01 b)]
    (+ (* (get gate 1 0) (LN pg)) (* (get gate 0 0) (LN (- 1.0 pg)))
       (* (get sched 1 0) (LN ps)) (* (get sched 0 0) (LN (- 1.0 ps))))))

(defn- grid-2d
  "Posterior over the (a,b) grid; returns ridge corr in (log(1-a), log b),
   the b-marginal CI90 width, and posterior means."
  [stats]
  (let [axis (mapv #(/ % 100) (range 5 100 5))
        pts (vec (for [a axis b axis] [a b]))
        lls (mapv (fn [[a b]] (pooled-ll stats a b)) pts)
        mxll (apply max lls)
        ws (mapv #(js/Math.exp (- % mxll)) lls)
        z (reduce + ws)
        post (mapv #(/ % z) ws)
        e (fn [f] (reduce + (map (fn [[a b] w] (* w (f a b))) pts post)))
        xf (fn [a _] (LN (- 1.0 a))) yf (fn [_ b] (LN b))
        xm (e xf) ym (e yf)
        cov (e (fn [a b] (* (- (xf a b) xm) (- (yf a b) ym))))
        vx (e (fn [a b] (let [d (- (xf a b) xm)] (* d d))))
        vy (e (fn [a b] (let [d (- (yf a b) ym)] (* d d))))
        corr (if (and (pos? vx) (pos? vy)) (/ cov (js/Math.sqrt (* vx vy))) 0.0)
        bmarg (reduce (fn [acc [[_ b] w]] (update acc b (fnil + 0.0) w))
                      {} (map vector pts post))
        bcdf (reductions + (map #(get bmarg % 0.0) axis))
        blo (nth axis (min (dec (count axis)) (count (take-while #(< % 0.05) bcdf))))
        bhi (nth axis (min (dec (count axis)) (count (take-while #(< % 0.95) bcdf))))]
    {:ridge-corr corr :b-ci-width (- bhi blo)
     :a-mean (e (fn [a _] a)) :b-mean (e (fn [_ b] b))}))

(let [a* 0.45 b* 0.85
      passive (gen-episodes 40 {:monitor (constantly 1.0) :tau 0.5
                                :hysteresis? false :min-gap 1
                                :gate-hazard (constantly (* (- 1.0 a*) b*))})
      instructed (gen-episodes 25 {:monitor (constantly 0.0) :tau 0.5
                                   :gate-hazard (constantly 1.0)
                                   :schedule [{:t 1 :hazard (constantly b*)}]})
      ;; tie the analytic grid loglik to the GFI replay at 3 grid points
      hz-fn (fn [site [a b]] (case (:kind site)
                               :gate (* (- 1.0 a) b)
                               :scheduled b
                               :decay (:hazard site)))
      all-eps (into passive instructed)
      stats-all (suff-stats all-eps)
      stats-passive (suff-stats passive)
      _ (doseq [[a b] [[0.45 0.85] [0.2 0.6] [0.7 0.3]]]
          (let [analytic (pooled-ll stats-all a b)
                gfi (reduce + 0.0 (map #(cyb/episode-log-lik % hz-fn [a b]) all-eps))]
            (mx/sweep-dead-arrays!)
            (assert-close (str "analytic grid ll == GFI replay ll at (a=" a ", b=" b ")")
                          gfi analytic 0.05)))
      passive-post (grid-2d stats-passive)
      pooled-post (grid-2d stats-all)]
  (println (str "    passive:  ridge-corr " (.toFixed (:ridge-corr passive-post) 3)
                "  b-CI90 width " (.toFixed (:b-ci-width passive-post) 2)))
  (println (str "    +do():    ridge-corr " (.toFixed (:ridge-corr pooled-post) 3)
                "  b-CI90 width " (.toFixed (:b-ci-width pooled-post) 2)
                "  b-mean " (.toFixed (:b-mean pooled-post) 2)))
  (assert-true "passive episodes: strongly negative ridge (only the product constrained)"
               (< (:ridge-corr passive-post) -0.8))
  (assert-true "passive episodes: b marginal is wide (CI90 width > 0.3)"
               (> (:b-ci-width passive-post) 0.3))
  (assert-true "do()-collapse: scheduled sites TIGHTEN the b marginal"
               (< (:b-ci-width pooled-post) (:b-ci-width passive-post)))
  (assert-true "do()-collapse: the ridge relaxes (corr strictly less negative)"
               (> (:ridge-corr pooled-post) (:ridge-corr passive-post)))
  (assert-close "do()-collapse: pooled b-mean recovers b* = 0.85"
                b* (:b-mean pooled-post) 0.15))

;; ---------------------------------------------------------------------------
;; 8. grep-guard: objective-free by construction (mirror of assert-downstream!)
;; ---------------------------------------------------------------------------
(println "\n== grep-guard: the cybernetic ns is objective-free ==")

(let [src (.readFileSync (js/require "fs") "src/genmlx/control/cybernetic.cljs" "utf8")]
  (assert-true "no genmlx.control.decision-value require"
               (not (re-find #"\[genmlx\.control\.decision-value" src)))
  (assert-true "no genmlx.inference.cost require"
               (not (re-find #"\[genmlx\.inference\.cost" src))))

;; ---------------------------------------------------------------------------
(println (str "\n== control_cybernetic_test: "
              (if (zero? @fails) "ALL PASS" (str @fails " FAIL")) " =="))
(when (pos? @fails) (js/process.exit 1))
