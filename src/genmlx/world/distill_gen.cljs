(ns genmlx.world.distill-gen
  "Scaled task / curriculum GENERATOR for the cljs-coder distillation loop (genmlx-7473).

   THE FUEL the loop was missing. The seed set (genmlx.world.distill-tasks, 12 tasks) is
   too small/narrow: at that scale SFT overfits (the genmlx-o8w9 smoke regressed). ReST-EM
   shows fixed small task sets regress by iteration 2. This namespace GROWS the set ~15x by
   PARAMETERIZING the proven seed families and adding a broad hand-authored catalog — every
   task carrying a REFERENCE solution that is validated against the SAME native-free oracle
   (genmlx.world.distill/evaluate-candidate) that grades the student.

   GROUNDING — the oracle stays grounded, never an LLM judge:
     :program  tasks are graded by EXACT Bayesian model evidence (conjugate marginal
               log p(obs)). The reference is ONE valid covering model; ANY covering conjugate
               model scores the same analytical marginal, so the oracle is independent of the
               reference's exact form.
     :function tasks are graded behaviorally against HAND-WRITTEN ground-truth (args/state ->
               expected). The expected values are independent mathematical facts authored here,
               NEVER derived by running the reference (that would make the oracle circular).
               The reference's only job is to CONFIRM the ground truth is self-consistent — if a
               reference disagrees with its own test cases, distill_gen_test goes red, catching
               an authoring bug in EITHER the reference or the ground truth.

   SPLIT — train / held-out eval is assigned per task (:split), family-balanced and
   leakage-safe: every held-out eval task is a DISTINCT instance/function whose family also
   appears in train (so passing it is real same-distribution generalization, never the rrps
   validation-on-train leak). Eval references and completions never enter the training corpus.

   This namespace is PURE data + string building (no oracle call, no I/O, no model). The
   reference-validation runs in distill_gen_test (CI) and scripts/gen_tasks.cljs (--validate);
   the teacher/filter/SFT scripts consume `all-tasks` / `train-tasks` / `eval-tasks` exactly as
   the seed scripts consume distill-tasks/tasks."
  (:require [clojure.string :as str]
            [genmlx.world.distill-gen-extra :as extra]))

;; ===========================================================================
;; System prompts — byte-identical to the seed set, so the teacher (bake-off
;; tuned on this exact phrasing, 93% kept) generalizes to the scaled prompts.
;; ===========================================================================

(def program-system-prompt
  (str "You are a ClojureScript code generator for the GenMLX probabilistic "
       "programming system. Reply with ONLY a single (fn [trace] ...) form — no "
       "prose, no markdown, no comments. A random choice is written "
       "(trace :name (dist/DISTRIBUTION args...)). You may use the distributions "
       "dist/gaussian, dist/uniform, dist/bernoulli, dist/beta, dist/gamma, "
       "dist/exponential, dist/poisson and the math ops mx/add, mx/subtract, "
       "mx/multiply, mx/divide, mx/scalar, mx/exp, mx/log, mx/sqrt, mx/abs. "
       "Begin your reply with the characters (fn [trace]."))

(def function-system-prompt
  (str "You are a ClojureScript code generator. Reply with ONLY a single "
       "ClojureScript function — no prose, no markdown, no comments. Begin your "
       "reply with ( ."))

;; ===========================================================================
;; 1. Program tasks — conjugate models, EXACT analytical model evidence
;; ===========================================================================

(defn- gaussian-mean-prompt
  "Shared-mean Gaussian scaffold (identical shape to the seed's): full template over
   the real observation addresses (so a faithful completion clears the coverage guard)
   with deliberately loose priors, asking the teacher to ADAPT the three numbers."
  [obs]
  (let [pairs (sort-by (comp name key) obs)
        data  (str/join ", " (map (fn [[k v]] (str (name k) "=" v)) pairs))
        body  (str/join " " (map (fn [[k _]] (str k " (trace " k
                                                 " (dist/gaussian mu 1))")) pairs))]
    (str "Write a GenMLX probabilistic model that explains this data: " data ".\n"
         "It has one shared latent mean :mu and one Gaussian observation site per\n"
         "data point. Start from this example and ADAPT the three numbers (the prior\n"
         "mean, the prior std, and the observation noise) so the data is well\n"
         "explained — a higher marginal likelihood is better:\n\n"
         "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 10))] {" body "}))\n\n"
         "Output ONLY your completed (fn [trace] ...) form, nothing else.")))

(defn- gaussian-mean-task
  "One shared-mean Gaussian instance from a data vector. Reference centres the prior
   near the data mean with a moderate noise — a covering conjugate model that scores by
   EXACT marginal. id distinguishes the instance; data is a vector of observed reals."
  [id data]
  (let [obs   (into {} (map-indexed (fn [i v] [(keyword (str "y" i)) v]) data))
        pairs (sort-by (comp name key) obs)
        m     (/ (reduce + data) (count data))
        pm    (.toFixed (js/Number m) 1)
        body  (str/join " " (map (fn [[k _]] (str k " (trace " k
                                                  " (dist/gaussian mu 1))")) pairs))]
    {:id id :kind :program :family :gaussian-mean :difficulty :medium
     :system-prompt program-system-prompt
     :prompt (gaussian-mean-prompt obs)
     :observations obs
     :reference (str "(fn [trace] (let [mu (trace :mu (dist/gaussian " pm " 5))] {"
                     body "}))")}))

(defn- bernoulli-prompt
  "Beta-Bernoulli coin scaffold, parameterized over a flip sequence (1 = heads)."
  [flips]
  (let [n      (count flips)
        addrs  (mapv #(str ":f" %) (range n))
        seqstr (str/join " " flips)
        body   (str/join " " (map (fn [a] (str a " (trace " a " (dist/bernoulli p))")) addrs))]
    (str "Write a GenMLX model for a possibly-biased coin observed flipping\n"
         seqstr " (" n " flips, sites :f0..:f" (dec n) ", 1 = heads). One latent bias\n"
         ":p with a Beta prior; each flip is Bernoulli(p). Start from this and\n"
         "ADAPT the two Beta-prior numbers so the data is well explained:\n\n"
         "(fn [trace] (let [p (trace :p (dist/beta 1 1))] {" body "}))\n\n"
         "Output ONLY your completed (fn [trace] ...) form, nothing else.")))

(defn- beta-bernoulli-task
  "One beta-Bernoulli instance from a 0/1 flip vector. Reference uses a Beta(2,2)
   prior + Bernoulli flips — conjugate, EXACT marginal."
  [id flips]
  (let [n     (count flips)
        addrs (mapv #(str ":f" %) (range n))
        obs   (into {} (map-indexed (fn [i b] [(keyword (str "f" i)) (double b)]) flips))
        body  (str/join " " (map (fn [a] (str a " (trace " a " (dist/bernoulli p))")) addrs))]
    {:id id :kind :program :family :beta-bernoulli :difficulty :medium
     :system-prompt program-system-prompt
     :prompt (bernoulli-prompt flips)
     :observations obs
     :reference (str "(fn [trace] (let [p (trace :p (dist/beta 2 2))] {" body "}))")}))

(defn- poisson-prompt
  "Gamma-Poisson count scaffold, parameterized over a count vector."
  [counts]
  (let [n      (count counts)
        addrs  (mapv #(str ":c" %) (range n))
        seqstr (str/join " " counts)
        body   (str/join " " (map (fn [a] (str a " (trace " a " (dist/poisson rate))")) addrs))]
    (str "Write a GenMLX model for count data " seqstr " (sites :c0..:c" (dec n) "). One\n"
         "latent rate :rate with a Gamma prior; each count is Poisson(rate).\n"
         "Start from this and ADAPT the two Gamma-prior numbers so the data is\n"
         "well explained:\n\n"
         "(fn [trace] (let [rate (trace :rate (dist/gamma 1 1))] {" body "}))\n\n"
         "Output ONLY your completed (fn [trace] ...) form, nothing else.")))

(defn- gamma-poisson-task
  "One gamma-Poisson instance from a count vector. Reference uses a Gamma(2,1) prior +
   Poisson counts — conjugate, EXACT marginal."
  [id counts]
  (let [n     (count counts)
        addrs (mapv #(str ":c" %) (range n))
        obs   (into {} (map-indexed (fn [i c] [(keyword (str "c" i)) (double c)]) counts))
        body  (str/join " " (map (fn [a] (str a " (trace " a " (dist/poisson rate))")) addrs))]
    {:id id :kind :program :family :gamma-poisson :difficulty :medium
     :system-prompt program-system-prompt
     :prompt (poisson-prompt counts)
     :observations obs
     :reference (str "(fn [trace] (let [rate (trace :rate (dist/gamma 2 1))] {" body "}))")}))

;; --- Parameter banks (deterministic, hand-picked for spread) ----------------

(def ^:private gaussian-datasets
  "Distinct shared-mean Gaussian datasets: clusters at varied means, 3-6 points each."
  [["gm-pos2"   [2.0 2.3 1.7 2.1]]
   ["gm-neg3"   [-3.1 -2.7 -3.4 -2.9]]
   ["gm-zero"   [0.2 -0.1 0.4 -0.3 0.1]]
   ["gm-pos5"   [5.2 4.8 5.5 5.0]]
   ["gm-neg1"   [-1.4 -0.9 -1.7 -1.1 -1.3]]
   ["gm-pos37"  [3.7 4.1 3.4 3.9]]
   ["gm-pos8"   [8.1 7.6 8.4 7.9 8.2 7.8]]
   ["gm-neg6"   [-6.2 -5.8 -6.5 -6.0]]
   ["gm-pos15"  [1.5 1.2 1.8 1.4]]
   ["gm-half"   [0.5 0.7 0.3 0.6 0.4]]
   ["gm-pos10"  [10.3 9.7 10.1 9.9 10.4]]
   ["gm-neg45"  [-4.5 -4.1 -4.8 -4.3]]
   ["gm-pos25"  [2.5 2.9 2.2 2.7 2.4 2.6]]
   ["gm-pos62"  [6.2 5.9 6.5 6.0]]
   ["gm-neg08"  [-0.8 -1.2 -0.5 -0.9]]
   ["gm-pos44"  [4.4 4.0 4.7 4.2 4.5]]
   ["gm-pos33"  [3.3 3.6 3.0 3.4]]
   ["gm-neg25"  [-2.5 -2.1 -2.8 -2.3 -2.6]]
   ["gm-pos71"  [7.1 6.8 7.4 6.9]]
   ["gm-pos18"  [1.8 2.1 1.5 1.9 1.7]]])

(def ^:private bernoulli-datasets
  "Distinct coin-flip sequences (1 = heads), varied length + bias."
  [["bb-3of6"   [1 1 1 0 1 0]]
   ["bb-1of5"   [1 0 0 0 0]]
   ["bb-4of4"   [1 1 1 1]]
   ["bb-0of4"   [0 0 0 0]]
   ["bb-5of8"   [1 0 1 1 0 1 1 0]]
   ["bb-2of6"   [0 1 0 0 1 0]]
   ["bb-6of7"   [1 1 1 0 1 1 1]]
   ["bb-3of5"   [1 0 1 1 0]]
   ["bb-7of8"   [1 1 1 1 0 1 1 1]]
   ["bb-1of4"   [1 0 0 0]]
   ["bb-4of6"   [1 1 0 1 0 1]]
   ["bb-2of8"   [0 0 1 0 0 1 0 0]]])

(def ^:private poisson-datasets
  "Distinct count datasets, varied rate."
  [["gp-mid"    [3 5 4 6 2]]
   ["gp-low"    [1 0 2 1 0]]
   ["gp-high"   [9 11 8 10 12]]
   ["gp-tiny"   [0 1 0 0 1 0]]
   ["gp-mid2"   [4 4 5 3 6 4]]
   ["gp-seven"  [7 6 8 7 5]]
   ["gp-two"    [2 3 1 2 3]]
   ["gp-big"    [15 13 16 14]]
   ["gp-onefive"[1 2 1 1 2 1]]
   ["gp-six"    [6 5 7 6 8 5]]
   ["gp-three"  [3 2 4 3 3]]
   ["gp-ten"    [10 9 11 12 8]]])

(def program-tasks
  (vec (concat
        (map (fn [[id d]] (gaussian-mean-task id d)) gaussian-datasets)
        (map (fn [[id d]] (beta-bernoulli-task id d)) bernoulli-datasets)
        (map (fn [[id d]] (gamma-poisson-task id d)) poisson-datasets))))

;; ===========================================================================
;; 2. Function tasks — state-machine transitions (hand-written ground truth)
;; ===========================================================================

(defn- transition-task
  [{:keys [id family difficulty prompt reference transitions]}]
  {:id id :kind :function :family family :difficulty (or difficulty :easy)
   :system-prompt function-system-prompt
   :prompt prompt :reference reference :transitions transitions})

(def machine-specs
  [{:id "counter-machine" :family :counter
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a "
                 "counter.\nThe state is a map {:count n}. Actions: :inc adds 1 to "
                 ":count, :dec\nsubtracts 1, :reset sets :count to 0. Return the new "
                 "state map.\nExample: (f {:count 5} :inc) => {:count 6}.")
    :reference "(fn [state action] (case action :inc (update state :count inc) :dec (update state :count dec) :reset (assoc state :count 0) state))"
    :transitions [{:state {:count 0}  :action :inc   :expected {:count 1}}
                  {:state {:count 3}  :action :dec   :expected {:count 2}}
                  {:state {:count 7}  :action :reset :expected {:count 0}}
                  {:state {:count -2} :action :inc   :expected {:count -1}}]}

   {:id "counter-by-2" :family :counter
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a step-2 "
                 "counter.\nThe state is {:count n}. Actions: :up adds 2, :down "
                 "subtracts 2, :zero\nsets :count to 0. Return the new state map. "
                 "Example: (f {:count 4} :up) => {:count 6}.")
    :reference "(fn [state action] (case action :up (update state :count + 2) :down (update state :count - 2) :zero (assoc state :count 0) state))"
    :transitions [{:state {:count 4} :action :up   :expected {:count 6}}
                  {:state {:count 4} :action :down :expected {:count 2}}
                  {:state {:count 9} :action :zero :expected {:count 0}}
                  {:state {:count 0} :action :up   :expected {:count 2}}]}

   {:id "bounded-counter" :family :counter :difficulty :medium
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a counter "
                 "clamped to 0..10.\nThe state is {:count n}. :inc adds 1 but never "
                 "above 10; :dec subtracts 1 but\nnever below 0. Return the new state. "
                 "Example: (f {:count 10} :inc) => {:count 10}.")
    :reference "(fn [state action] (case action :inc (update state :count #(min 10 (inc %))) :dec (update state :count #(max 0 (dec %))) state))"
    :transitions [{:state {:count 10} :action :inc :expected {:count 10}}
                  {:state {:count 0}  :action :dec :expected {:count 0}}
                  {:state {:count 5}  :action :inc :expected {:count 6}}
                  {:state {:count 5}  :action :dec :expected {:count 4}}]}

   {:id "traffic-light" :family :cyclic
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a "
                 "traffic light.\nThe state is {:light color} where color is :red, "
                 ":green, or :yellow.\nThe only action is :tick, which advances the "
                 "light: :red -> :green ->\n:yellow -> :red. Return the new state. "
                 "Example: (f {:light :red} :tick) => {:light :green}.")
    :reference "(fn [state action] (if (= action :tick) (update state :light {:red :green :green :yellow :yellow :red}) state))"
    :transitions [{:state {:light :red}    :action :tick :expected {:light :green}}
                  {:state {:light :green}  :action :tick :expected {:light :yellow}}
                  {:state {:light :yellow} :action :tick :expected {:light :red}}]}

   {:id "day-cycle" :family :cyclic
    :prompt (str "Write a ClojureScript function (fn [state action] ...) cycling days.\n"
                 "The state is {:day d} where d is :mon, :tue, or :wed. The action "
                 ":next\nadvances :mon -> :tue -> :wed -> :mon. Return the new state.\n"
                 "Example: (f {:day :mon} :next) => {:day :tue}.")
    :reference "(fn [state action] (if (= action :next) (update state :day {:mon :tue :tue :wed :wed :mon}) state))"
    :transitions [{:state {:day :mon} :action :next :expected {:day :tue}}
                  {:state {:day :tue} :action :next :expected {:day :wed}}
                  {:state {:day :wed} :action :next :expected {:day :mon}}]}

   {:id "four-phase" :family :cyclic :difficulty :medium
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a 4-phase "
                 "cycle.\nThe state is {:phase p} with p in :a :b :c :d. :step advances "
                 ":a->:b->:c->:d->:a;\n:back reverses it. Return the new state. "
                 "Example: (f {:phase :a} :step) => {:phase :b}.")
    :reference "(fn [state action] (case action :step (update state :phase {:a :b :b :c :c :d :d :a}) :back (update state :phase {:a :d :b :a :c :b :d :c}) state))"
    :transitions [{:state {:phase :a} :action :step :expected {:phase :b}}
                  {:state {:phase :d} :action :step :expected {:phase :a}}
                  {:state {:phase :a} :action :back :expected {:phase :d}}
                  {:state {:phase :c} :action :back :expected {:phase :b}}]}

   {:id "toggle-switch" :family :toggle
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a toggle "
                 "switch.\nThe state is {:on bool}. The action :flip inverts :on; any "
                 "other action\nleaves the state unchanged. Return the new state. "
                 "Example:\n(f {:on false} :flip) => {:on true}.")
    :reference "(fn [state action] (if (= action :flip) (update state :on not) state))"
    :transitions [{:state {:on false} :action :flip :expected {:on true}}
                  {:state {:on true}  :action :flip :expected {:on false}}
                  {:state {:on true}  :action :noop :expected {:on true}}]}

   {:id "lamp-switch" :family :toggle
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a lamp.\n"
                 "The state is {:on bool}. :turn-on sets :on true, :turn-off sets it "
                 "false,\n:flip inverts it. Return the new state. "
                 "Example: (f {:on false} :turn-on) => {:on true}.")
    :reference "(fn [state action] (case action :turn-on (assoc state :on true) :turn-off (assoc state :on false) :flip (update state :on not) state))"
    :transitions [{:state {:on false} :action :turn-on  :expected {:on true}}
                  {:state {:on true}  :action :turn-off :expected {:on false}}
                  {:state {:on false} :action :flip     :expected {:on true}}
                  {:state {:on true}  :action :flip     :expected {:on false}}]}

   {:id "accumulator" :family :register :difficulty :medium
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for an "
                 "accumulator.\nThe state is {:acc n}. :double multiplies :acc by 2, "
                 ":negate flips its sign,\n:clear sets it to 0. Return the new state. "
                 "Example: (f {:acc 3} :double) => {:acc 6}.")
    :reference "(fn [state action] (case action :double (update state :acc * 2) :negate (update state :acc -) :clear (assoc state :acc 0) state))"
    :transitions [{:state {:acc 3}  :action :double :expected {:acc 6}}
                  {:state {:acc 5}  :action :negate :expected {:acc -5}}
                  {:state {:acc -4} :action :negate :expected {:acc 4}}
                  {:state {:acc 9}  :action :clear  :expected {:acc 0}}]}

   {:id "turnstile" :family :mode :difficulty :medium
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a "
                 "turnstile.\nThe state is {:mode m} with m :locked or :unlocked. "
                 ":coin unlocks it,\n:push locks it. Return the new state. "
                 "Example: (f {:mode :locked} :coin) => {:mode :unlocked}.")
    :reference "(fn [state action] (case action :coin (assoc state :mode :unlocked) :push (assoc state :mode :locked) state))"
    :transitions [{:state {:mode :locked}   :action :coin :expected {:mode :unlocked}}
                  {:state {:mode :unlocked} :action :push :expected {:mode :locked}}
                  {:state {:mode :locked}   :action :push :expected {:mode :locked}}
                  {:state {:mode :unlocked} :action :coin :expected {:mode :unlocked}}]}

   {:id "position-1d" :family :position :difficulty :medium
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for 1-D "
                 "movement.\nThe state is {:pos n}. :left subtracts 1, :right adds 1, "
                 ":home sets :pos 0.\nReturn the new state. "
                 "Example: (f {:pos 2} :left) => {:pos 1}.")
    :reference "(fn [state action] (case action :left (update state :pos dec) :right (update state :pos inc) :home (assoc state :pos 0) state))"
    :transitions [{:state {:pos 2}  :action :left  :expected {:pos 1}}
                  {:state {:pos 2}  :action :right :expected {:pos 3}}
                  {:state {:pos 5}  :action :home  :expected {:pos 0}}
                  {:state {:pos -1} :action :right :expected {:pos 0}}]}

   {:id "score-board" :family :register :difficulty :medium
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a score "
                 "board.\nThe state is {:score n}. :goal adds 3, :foul subtracts 1, "
                 ":reset sets 0.\nReturn the new state. "
                 "Example: (f {:score 0} :goal) => {:score 3}.")
    :reference "(fn [state action] (case action :goal (update state :score + 3) :foul (update state :score - 1) :reset (assoc state :score 0) state))"
    :transitions [{:state {:score 0} :action :goal  :expected {:score 3}}
                  {:state {:score 3} :action :foul  :expected {:score 2}}
                  {:state {:score 9} :action :reset :expected {:score 0}}
                  {:state {:score 6} :action :goal  :expected {:score 9}}]}])

(def transition-tasks (mapv transition-task machine-specs))

;; ===========================================================================
;; 3. Function tasks — pure functions, HAND-WRITTEN test-case ground truth
;; ===========================================================================

(defn- function-task
  "Build a :function/test-case task from a spec. `:tests` is [[args expected]...] of
   INDEPENDENT ground truth (never derived from the reference). The first `:n-shown`
   tests (default 2) are revealed as examples in the prompt; ALL tests grade the
   student. `:sig` is the signature shown to the model; `:desc` the behavior clause."
  [{:keys [id family difficulty sig desc reference tests n-shown]
    :or   {n-shown 2 difficulty :easy}}]
  (let [shown  (take n-shown tests)
        ex-str (str/join ", "
                         (map (fn [[args r]]
                                (str "(f " (str/join " " (map pr-str args)) ") => " (pr-str r)))
                              shown))]
    {:id id :kind :function :family family :difficulty difficulty
     :system-prompt function-system-prompt
     :prompt (str "Write a ClojureScript function " sig " " desc ". " ex-str ".")
     :reference reference
     :test-cases (mapv (fn [[args r]] {:args args :expected r}) tests)}))

(def function-specs
  "Hand-authored catalog of standard small functions, each with INDEPENDENT ground-truth
   test cases (well-known mathematical/behavioral facts) + an idiomatic reference. The
   reference is validated to satisfy these tests in distill_gen_test (catching authoring
   bugs in either)."
  [;; --- arithmetic / number theory ---
   {:id "factorial" :family :arithmetic :sig "(fn [n] ...)"
    :desc "that returns n! (n factorial) for a non-negative integer n"
    :reference "(fn [n] (reduce * 1 (range 1 (inc n))))"
    :tests [[[0] 1] [[4] 24] [[1] 1] [[5] 120] [[6] 720]]}
   {:id "fib" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)"
    :desc "returning the n-th Fibonacci number (0-indexed: fib 0 = 0, fib 1 = 1)"
    :reference "(fn [n] (loop [a 0 b 1 i n] (if (zero? i) a (recur b (+ a b) (dec i)))))"
    :tests [[[0] 0] [[1] 1] [[2] 1] [[7] 13] [[10] 55]]}
   {:id "gcd" :family :arithmetic :sig "(fn [a b] ...)"
    :desc "returning the greatest common divisor of two positive integers a and b"
    :reference "(fn [a b] (if (zero? b) a (recur b (mod a b))))"
    :tests [[[12 8] 4] [[54 24] 6] [[7 1] 1] [[100 75] 25] [[17 5] 1]]}
   {:id "lcm" :family :arithmetic :difficulty :medium :sig "(fn [a b] ...)"
    :desc "returning the least common multiple of two positive integers a and b"
    :reference "(fn [a b] (letfn [(g [x y] (if (zero? y) x (g y (mod x y))))] (/ (* a b) (g a b))))"
    :tests [[[4 6] 12] [[3 5] 15] [[12 8] 24] [[7 7] 7] [[2 10] 10]]}
   {:id "is-prime" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)"
    :desc "returning true if the integer n (>= 2) is prime, else false"
    :reference "(fn [n] (and (> n 1) (not-any? #(zero? (mod n %)) (range 2 n))))"
    :tests [[[2] true] [[7] true] [[9] false] [[1] false] [[13] true] [[15] false]]}
   {:id "digit-sum" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)"
    :desc "returning the sum of the decimal digits of a non-negative integer n"
    :reference "(fn [n] (loop [x n s 0] (if (zero? x) s (recur (quot x 10) (+ s (mod x 10))))))"
    :tests [[[0] 0] [[123] 6] [[9] 9] [[1000] 1] [[99] 18]]}
   {:id "square" :family :arithmetic :sig "(fn [x] ...)"
    :desc "returning x squared"
    :reference "(fn [x] (* x x))"
    :tests [[[3] 9] [[0] 0] [[-4] 16] [[7] 49] [[10] 100]]}
   {:id "abs-val" :family :arithmetic :sig "(fn [x] ...)"
    :desc "returning the absolute value of x"
    :reference "(fn [x] (if (neg? x) (- x) x))"
    :tests [[[3] 3] [[-5] 5] [[0] 0] [[-12] 12] [[8] 8]]}
   {:id "sum-to-n" :family :arithmetic :sig "(fn [n] ...)"
    :desc "returning the sum 1 + 2 + ... + n for a positive integer n"
    :reference "(fn [n] (reduce + (range 1 (inc n))))"
    :tests [[[1] 1] [[5] 15] [[10] 55] [[3] 6] [[100] 5050]]}
   {:id "power" :family :arithmetic :difficulty :medium :sig "(fn [base exp] ...)"
    :desc "returning base raised to a non-negative integer exp"
    :reference "(fn [base exp] (reduce * 1 (repeat exp base)))"
    :tests [[[2 3] 8] [[5 0] 1] [[3 2] 9] [[10 4] 10000] [[7 1] 7]]}
   {:id "even-odd-label" :family :arithmetic :sig "(fn [n] ...)"
    :desc "returning the string \"even\" if n is even, else \"odd\""
    :reference "(fn [n] (if (even? n) \"even\" \"odd\"))"
    :tests [[[2] "even"] [[3] "odd"] [[0] "even"] [[7] "odd"] [[10] "even"]]}
   {:id "sign-of" :family :arithmetic :sig "(fn [x] ...)"
    :desc "returning -1 if x is negative, 0 if zero, 1 if positive"
    :reference "(fn [x] (cond (neg? x) -1 (pos? x) 1 :else 0))"
    :tests [[[-5] -1] [[0] 0] [[8] 1] [[-1] -1] [[100] 1]]}
   {:id "celsius->fahrenheit" :family :arithmetic :sig "(fn [c] ...)"
    :desc "converting Celsius c to Fahrenheit (F = c * 1.8 + 32)"
    :reference "(fn [c] (+ (* c 1.8) 32))"
    :tests [[[0] 32] [[100] 212] [[-40] -40] [[10] 50] [[20] 68]]}
   {:id "clamp-0-100" :family :arithmetic :sig "(fn [x] ...)"
    :desc "clamping x into the range 0..100 inclusive"
    :reference "(fn [x] (max 0 (min 100 x)))"
    :tests [[[50] 50] [[-10] 0] [[150] 100] [[0] 0] [[100] 100]]}
   {:id "collatz-steps" :family :arithmetic :difficulty :hard :sig "(fn [n] ...)"
    :desc "returning how many Collatz steps reduce a positive integer n to 1 (even -> n/2, odd -> 3n+1)"
    :reference "(fn [n] (loop [x n c 0] (if (= x 1) c (recur (if (even? x) (quot x 2) (inc (* 3 x))) (inc c)))))"
    :tests [[[1] 0] [[2] 1] [[3] 7] [[6] 8] [[16] 4]]}

   ;; --- collections ---
   {:id "sum-list" :family :collection :sig "(fn [coll] ...)"
    :desc "returning the sum of all numbers in the collection coll"
    :reference "(fn [coll] (reduce + 0 coll))"
    :tests [[[[1 2 3 4]] 10] [[[]] 0] [[[5]] 5] [[[-1 1 -2 2]] 0] [[[10 20 30]] 60]]}
   {:id "sum-evens" :family :collection :sig "(fn [coll] ...)"
    :desc "returning the sum of the even numbers in the collection coll"
    :reference "(fn [coll] (reduce + 0 (filter even? coll)))"
    :tests [[[[1 2 3 4]] 6] [[[]] 0] [[[2 4 6]] 12] [[[1 3 5]] 0] [[[10 -2 3]] 8]]}
   {:id "count-positives" :family :collection :sig "(fn [coll] ...)"
    :desc "returning how many numbers in coll are strictly positive"
    :reference "(fn [coll] (count (filter pos? coll)))"
    :tests [[[[1 -2 3 0 5]] 3] [[[]] 0] [[[-1 -2 -3]] 0] [[[4 4 4]] 3] [[[0 0 1]] 1]]}
   {:id "max-of-list" :family :collection :sig "(fn [coll] ...)"
    :desc "returning the largest number in the non-empty collection coll"
    :reference "(fn [coll] (reduce max coll))"
    :tests [[[[3 1 4 1 5]] 5] [[[7]] 7] [[[-3 -1 -2]] -1] [[[2 2 2]] 2] [[[10 9 11 8]] 11]]}
   {:id "min-of-list" :family :collection :sig "(fn [coll] ...)"
    :desc "returning the smallest number in the non-empty collection coll"
    :reference "(fn [coll] (reduce min coll))"
    :tests [[[[3 1 4 1 5]] 1] [[[7]] 7] [[[-3 -1 -2]] -3] [[[2 2 2]] 2] [[[10 9 11 8]] 8]]}
   {:id "my-reverse" :family :collection :sig "(fn [coll] ...)"
    :desc "returning the elements of coll in reverse order, as a vector"
    :reference "(fn [coll] (vec (reverse coll)))"
    :tests [[[[1 2 3]] [3 2 1]] [[[]] []] [[[9]] [9]] [[[1 2 3 4 5]] [5 4 3 2 1]] [[[7 8]] [8 7]]]}
   {:id "list-length" :family :collection :sig "(fn [coll] ...)"
    :desc "returning the number of elements in coll"
    :reference "(fn [coll] (count coll))"
    :tests [[[[1 2 3]] 3] [[[]] 0] [[[9 9]] 2] [[[1 2 3 4 5]] 5] [[["a" "b"]] 2]]}
   {:id "mean-of-list" :family :collection :difficulty :medium :sig "(fn [coll] ...)"
    :desc "returning the arithmetic mean of the non-empty number collection coll"
    :reference "(fn [coll] (/ (reduce + coll) (count coll)))"
    :tests [[[[2 4 6]] 4] [[[5]] 5] [[[1 2 3 4]] 5/2] [[[10 20]] 15] [[[0 0 0]] 0]]}
   {:id "second-largest" :family :collection :difficulty :medium :sig "(fn [coll] ...)"
    :desc "returning the second-largest distinct value in coll (which has >= 2 distinct values)"
    :reference "(fn [coll] (second (sort > (distinct coll))))"
    :tests [[[[3 1 4 1 5]] 4] [[[7 2]] 2] [[[10 9 11 8]] 10] [[[1 2 3]] 2] [[[5 5 4]] 4]]}
   {:id "all-positive?" :family :collection :sig "(fn [coll] ...)"
    :desc "returning true if every number in coll is strictly positive, else false"
    :reference "(fn [coll] (every? pos? coll))"
    :tests [[[[1 2 3]] true] [[[1 -2 3]] false] [[[]] true] [[[0 1]] false] [[[5 5 5]] true]]}
   {:id "running-sum" :family :collection :difficulty :medium :sig "(fn [coll] ...)"
    :desc "returning a vector of cumulative sums of coll (each element is the sum so far)"
    :reference "(fn [coll] (vec (rest (reductions + 0 coll))))"
    :tests [[[[1 2 3]] [1 3 6]] [[[]] []] [[[5]] [5]] [[[1 1 1 1]] [1 2 3 4]] [[[2 -1 3]] [2 1 4]]]}
   {:id "dedupe-list" :family :collection :sig "(fn [coll] ...)"
    :desc "returning a vector of the distinct elements of coll, preserving first-seen order"
    :reference "(fn [coll] (vec (distinct coll)))"
    :tests [[[[1 1 2 3 3]] [1 2 3]] [[[]] []] [[[5 5 5]] [5]] [[[1 2 1 2]] [1 2]] [[[9 8 9 7]] [9 8 7]]]}
   {:id "map-square" :family :collection :sig "(fn [coll] ...)"
    :desc "returning a vector with each element of coll squared"
    :reference "(fn [coll] (mapv #(* % %) coll))"
    :tests [[[[1 2 3]] [1 4 9]] [[[]] []] [[[-2 4]] [4 16]] [[[0 5]] [0 25]] [[[10]] [100]]]}
   {:id "filter-even" :family :collection :sig "(fn [coll] ...)"
    :desc "returning a vector of just the even numbers of coll, in order"
    :reference "(fn [coll] (vec (filter even? coll)))"
    :tests [[[[1 2 3 4]] [2 4]] [[[]] []] [[[1 3 5]] []] [[[2 4 6]] [2 4 6]] [[[7 8 9 10]] [8 10]]]}
   {:id "dot-product" :family :collection :difficulty :medium :sig "(fn [a b] ...)"
    :desc "returning the dot product of two equal-length number vectors a and b"
    :reference "(fn [a b] (reduce + (map * a b)))"
    :tests [[[[1 2 3] [4 5 6]] 32] [[[0 0] [1 1]] 0] [[[2] [3]] 6] [[[1 1 1] [1 2 3]] 6] [[[-1 1] [2 2]] 0]]}

   ;; --- strings ---
   {:id "palindrome?" :family :string :sig "(fn [s] ...)"
    :desc "returning true if the string s reads the same forwards and backwards, else false"
    :reference "(fn [s] (= (vec s) (vec (reverse s))))"
    :tests [[["racecar"] true] [["hello"] false] [[""] true] [["abba"] true] [["abc"] false]]}
   {:id "reverse-str" :family :string :sig "(fn [s] ...)"
    :desc "returning the string s reversed"
    :reference "(fn [s] (apply str (reverse s)))"
    :tests [[["abc"] "cba"] [[""] ""] [["a"] "a"] [["hello"] "olleh"] [["12 3"] "3 21"]]}
   {:id "count-vowels" :family :string :difficulty :medium :sig "(fn [s] ...)"
    :desc "returning how many vowels (a e i o u, lowercase) are in the string s"
    :reference "(fn [s] (count (filter (set \"aeiou\") s)))"
    :tests [[["hello"] 2] [["xyz"] 0] [[""] 0] [["aeiou"] 5] [["banana"] 3]]}
   {:id "str-length" :family :string :sig "(fn [s] ...)"
    :desc "returning the number of characters in the string s"
    :reference "(fn [s] (count s))"
    :tests [[["abc"] 3] [[""] 0] [["hello world"] 11] [["x"] 1] [["12345"] 5]]}
   {:id "fizzbuzz" :family :string :difficulty :medium :sig "(fn [n] ...)"
    :desc (str "that returns \"Fizz\" if n is divisible by 3, \"Buzz\" if divisible by 5, "
               "\"FizzBuzz\" if divisible by both, otherwise the number as a string")
    :reference "(fn [n] (cond (zero? (mod n 15)) \"FizzBuzz\" (zero? (mod n 3)) \"Fizz\" (zero? (mod n 5)) \"Buzz\" :else (str n)))"
    :tests [[[3] "Fizz"] [[5] "Buzz"] [[15] "FizzBuzz"] [[7] "7"] [[9] "Fizz"]]}
   {:id "shout" :family :string :sig "(fn [s] ...)"
    :desc "returning the string s in upper case"
    :reference "(fn [s] (clojure.string/upper-case s))"
    :tests [[["hello"] "HELLO"] [[""] ""] [["AbC"] "ABC"] [["x y"] "X Y"] [["123"] "123"]]}
   {:id "count-char" :family :string :difficulty :medium :sig "(fn [s c] ...)"
    :desc "returning how many times the character c occurs in the string s"
    :reference "(fn [s c] (count (filter #(= % c) s)))"
    :tests [[["banana" \a] 3] [["hello" \l] 2] [["abc" \z] 0] [["" \x] 0] [["aaaa" \a] 4]]}

   ;; --- logic / classification ---
   {:id "leap-year?" :family :logic :difficulty :medium :sig "(fn [y] ...)"
    :desc "returning true if year y is a leap year (divisible by 4 but not 100, unless also by 400)"
    :reference "(fn [y] (and (zero? (mod y 4)) (or (pos? (mod y 100)) (zero? (mod y 400)))))"
    :tests [[[2000] true] [[1900] false] [[2024] true] [[2023] false] [[2400] true]]}
   {:id "grade-letter" :family :logic :difficulty :medium :sig "(fn [score] ...)"
    :desc "returning \"A\" for score >= 90, \"B\" for >= 80, \"C\" for >= 70, else \"F\""
    :reference "(fn [score] (cond (>= score 90) \"A\" (>= score 80) \"B\" (>= score 70) \"C\" :else \"F\"))"
    :tests [[[95] "A"] [[85] "B"] [[72] "C"] [[50] "F"] [[90] "A"]]}
   {:id "max-of-3" :family :logic :sig "(fn [a b c] ...)"
    :desc "returning the largest of three numbers a, b, c"
    :reference "(fn [a b c] (max a b c))"
    :tests [[[1 2 3] 3] [[5 1 2] 5] [[4 9 4] 9] [[-1 -2 -3] -1] [[7 7 7] 7]]}
   {:id "between?" :family :logic :sig "(fn [x lo hi] ...)"
    :desc "returning true if lo <= x <= hi, else false"
    :reference "(fn [x lo hi] (and (<= lo x) (<= x hi)))"
    :tests [[[5 1 10] true] [[0 1 10] false] [[10 1 10] true] [[1 1 10] true] [[11 1 10] false]]}])

(def test-case-tasks
  "Hand-authored core (function-specs) + the workflow-expanded catalog (distill-gen-extra),
   built into :function/test-case tasks. References are oracle-validated downstream; any that
   fail are listed in `excluded-ids` and dropped here."
  (mapv function-task (concat function-specs extra/extra-function-specs)))

;; ===========================================================================
;; 4. Assembly + train/eval split + derived views
;; ===========================================================================

(def excluded-ids
  "Task ids dropped because their reference is not admitted by its own oracle (caught by
   scripts/gen_tasks.cljs --validate / distill_gen_test). Keeping this explicit — rather than
   silently editing the catalog — documents exactly what the grounding guard rejected."
  #{})

(defn- dedupe-by-id
  "Keep the first task per :id (programs/machines/core functions win over later extras),
   dropping any in `excluded-ids`. Guards against an extra-catalog id colliding with a core id."
  [tasks]
  (->> tasks
       (remove #(contains? excluded-ids (:id %)))
       (reduce (fn [{:keys [seen out]} t]
                 (if (contains? seen (:id t))
                   {:seen seen :out out}
                   {:seen (conj seen (:id t)) :out (conj out t)}))
               {:seen #{} :out []})
       :out))

(def ^:private candidate-tasks
  "The full generated set (deduped, excluded-pruned), before split assignment."
  (dedupe-by-id (concat program-tasks transition-tasks test-case-tasks)))

(defn- assign-split
  "Deterministically hold out every `stride`-th task WITHIN each family for eval, so the
   held-out set is family-balanced and every eval family keeps train siblings (real
   same-distribution generalization, never a leaked train instance). Programs are split
   per conjugate family; functions per behavioral family. Original task order is preserved
   via a threaded index (group-by reshuffles)."
  [tasks stride]
  (->> tasks
       (map-indexed (fn [i t] (assoc t ::idx i)))
       (group-by :family)
       (mapcat (fn [[_ fam-tasks]]
                 (map-indexed
                  (fn [i t]
                    (assoc t :split (if (zero? (mod (inc i) stride)) :eval :train)))
                  fam-tasks)))
       (sort-by ::idx)
       (mapv #(dissoc % ::idx))))

(def all-tasks
  "Every generated task, each tagged :split :train | :eval. ~1 in 4 per family is held out."
  (assign-split candidate-tasks 4))

(def train-tasks (vec (filter #(= :train (:split %)) all-tasks)))
(def eval-tasks  (vec (filter #(= :eval (:split %)) all-tasks)))

(def eval-task-ids
  "The held-out eval task ids — the scaled analogue of sft/eval-task-ids. Their references
   and completions must never enter the training corpus."
  (into #{} (map :id) eval-tasks))

(def tasks-by-id
  "Map of :id -> task, for joining verdicts/candidates back to their prompts (and oracle)."
  (into {} (map (juxt :id identity)) all-tasks))

(defn task->prompt-record
  "TEACHER-FACING projection: only {task_id, kind, system_prompt, prompt}. The held-out
   oracle signal (:observations / :transitions / :test-cases) AND the :reference are
   dropped, so neither the test answers nor a canonical solution can leak into a prompt."
  [task]
  {:task_id       (:id task)
   :kind          (name (:kind task))
   :system_prompt (:system-prompt task)
   :prompt        (:prompt task)})

(defn reference-record
  "A teacher-equivalent candidate row carrying the task's REFERENCE solution as the
   completion — for seeding the SFT corpus with one guaranteed-correct, oracle-validated
   exemplar per train task (cold-start insurance alongside teacher diversity)."
  [task]
  {:task_id    (:id task)
   :sample_idx -1
   :raw_text   (:reference task)})

(def summary
  "At-a-glance generated-set shape (counts by kind / family / split / difficulty)."
  {:n-total      (count all-tasks)
   :n-train      (count train-tasks)
   :n-eval       (count eval-tasks)
   :by-kind      (frequencies (map :kind all-tasks))
   :by-family    (into (sorted-map) (frequencies (map :family all-tasks)))
   :by-difficulty (into (sorted-map) (frequencies (map :difficulty all-tasks)))
   :eval-by-family (into (sorted-map) (frequencies (map :family eval-tasks)))})
