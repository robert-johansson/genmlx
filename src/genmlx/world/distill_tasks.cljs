(ns genmlx.world.distill-tasks
  "The seed task set for the cljs-coder distillation loop (genmlx-j0d6).

   The CANONICAL, in-tree source of distillation tasks. Each task carries the
   teacher-facing fields (:id :kind :system-prompt :prompt) AND the held-out oracle
   signal (:observations for :program; :transitions or :test-cases for :function).
   The held-out signal lives HERE, never in the exported teacher prompts, so a
   completion that passes the oracle is real generalization — the teacher never saw
   the tests it is graded on.

   Two kinds, two oracle paths (see genmlx.world.distill):
     :program  — a GenMLX probabilistic program scored by Bayesian model evidence
                 (log p(observations | program)). The four here are conjugate, so the
                 evidence is the EXACT analytical marginal (deterministic).
     :function — an ordinary ClojureScript function checked behaviorally, either as a
                 :transitions state-machine (state x action -> state) or by :test-cases
                 ([{:args [..] :expected v}]).

   Start tiny (12 tasks); the loop grows this set as the teacher's coverage is
   measured (yield-per-prompt) and weak spots are found."
  (:require [clojure.string :as str]))

;; ===========================================================================
;; System prompts
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
;; Program tasks — conjugate models, EXACT analytical model evidence
;; ===========================================================================

(defn- gaussian-mean-prompt
  "A shared-mean Gaussian program prompt: gives a complete template over the real
   observation addresses (so a faithful completion clears the coverage guard) with
   deliberately loose priors, and asks the teacher to ADAPT the three numbers."
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

(def program-tasks
  [{:id "gaussian-mean-near2"
    :kind :program
    :system-prompt program-system-prompt
    :prompt (gaussian-mean-prompt {:y0 2.0 :y1 2.3 :y2 1.7 :y3 2.1})
    :observations {:y0 2.0 :y1 2.3 :y2 1.7 :y3 2.1}}

   {:id "gaussian-mean-negshift"
    :kind :program
    :system-prompt program-system-prompt
    :prompt (gaussian-mean-prompt {:y0 -3.1 :y1 -2.7 :y2 -3.4 :y3 -2.9})
    :observations {:y0 -3.1 :y1 -2.7 :y2 -3.4 :y3 -2.9}}

   {:id "beta-bernoulli-coin"
    :kind :program
    :system-prompt program-system-prompt
    :prompt (str "Write a GenMLX model for a possibly-biased coin observed flipping\n"
                 "1 1 1 0 1 0 (six flips, sites :f0..:f5, 1 = heads). One latent bias\n"
                 ":p with a Beta prior; each flip is Bernoulli(p). Start from this and\n"
                 "ADAPT the two Beta-prior numbers so the data is well explained:\n\n"
                 "(fn [trace] (let [p (trace :p (dist/beta 1 1))]"
                 " {:f0 (trace :f0 (dist/bernoulli p))"
                 " :f1 (trace :f1 (dist/bernoulli p))"
                 " :f2 (trace :f2 (dist/bernoulli p))"
                 " :f3 (trace :f3 (dist/bernoulli p))"
                 " :f4 (trace :f4 (dist/bernoulli p))"
                 " :f5 (trace :f5 (dist/bernoulli p))}))\n\n"
                 "Output ONLY your completed (fn [trace] ...) form, nothing else.")
    :observations {:f0 1.0 :f1 1.0 :f2 1.0 :f3 0.0 :f4 1.0 :f5 0.0}}

   {:id "gamma-poisson-counts"
    :kind :program
    :system-prompt program-system-prompt
    :prompt (str "Write a GenMLX model for count data 3 5 4 6 2 (sites :c0..:c4). One\n"
                 "latent rate :rate with a Gamma prior; each count is Poisson(rate).\n"
                 "Start from this and ADAPT the two Gamma-prior numbers so the data is\n"
                 "well explained:\n\n"
                 "(fn [trace] (let [rate (trace :rate (dist/gamma 1 1))]"
                 " {:c0 (trace :c0 (dist/poisson rate))"
                 " :c1 (trace :c1 (dist/poisson rate))"
                 " :c2 (trace :c2 (dist/poisson rate))"
                 " :c3 (trace :c3 (dist/poisson rate))"
                 " :c4 (trace :c4 (dist/poisson rate))}))\n\n"
                 "Output ONLY your completed (fn [trace] ...) form, nothing else.")
    :observations {:c0 3.0 :c1 5.0 :c2 4.0 :c3 6.0 :c4 2.0}}])

;; ===========================================================================
;; Function tasks — state-machine transitions (held-out)
;; ===========================================================================

(def transition-tasks
  [{:id "counter-machine"
    :kind :function
    :system-prompt function-system-prompt
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a "
                 "counter.\nThe state is a map {:count n}. Actions: :inc adds 1 to "
                 ":count, :dec\nsubtracts 1, :reset sets :count to 0. Return the new "
                 "state map.\nExample: (f {:count 5} :inc) => {:count 6}.")
    :transitions [{:state {:count 0}  :action :inc   :expected {:count 1}}
                  {:state {:count 3}  :action :dec   :expected {:count 2}}
                  {:state {:count 7}  :action :reset :expected {:count 0}}
                  {:state {:count -2} :action :inc   :expected {:count -1}}]}

   {:id "traffic-light"
    :kind :function
    :system-prompt function-system-prompt
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a "
                 "traffic light.\nThe state is {:light color} where color is :red, "
                 ":green, or :yellow.\nThe only action is :tick, which advances the "
                 "light: :red -> :green ->\n:yellow -> :red. Return the new state. "
                 "Example: (f {:light :red} :tick) => {:light :green}.")
    :transitions [{:state {:light :red}    :action :tick :expected {:light :green}}
                  {:state {:light :green}  :action :tick :expected {:light :yellow}}
                  {:state {:light :yellow} :action :tick :expected {:light :red}}]}

   {:id "toggle-switch"
    :kind :function
    :system-prompt function-system-prompt
    :prompt (str "Write a ClojureScript function (fn [state action] ...) for a toggle "
                 "switch.\nThe state is {:on bool}. The action :flip inverts :on; any "
                 "other action\nleaves the state unchanged. Return the new state. "
                 "Example:\n(f {:on false} :flip) => {:on true}.")
    :transitions [{:state {:on false} :action :flip :expected {:on true}}
                  {:state {:on true}  :action :flip :expected {:on false}}
                  {:state {:on true}  :action :noop :expected {:on true}}]}])

;; ===========================================================================
;; Function tasks — pure functions checked by held-out test-cases
;; ===========================================================================

(def test-case-tasks
  [{:id "factorial"
    :kind :function
    :system-prompt function-system-prompt
    :prompt (str "Write a ClojureScript function (fn [n] ...) that returns n! "
                 "(n factorial)\nfor a non-negative integer n. (f 0) => 1, "
                 "(f 4) => 24.")
    :test-cases [{:args [0] :expected 1} {:args [1] :expected 1}
                 {:args [5] :expected 120} {:args [6] :expected 720}]}

   {:id "fizzbuzz"
    :kind :function
    :system-prompt function-system-prompt
    :prompt (str "Write a ClojureScript function (fn [n] ...) that returns \"Fizz\" "
                 "if n is\ndivisible by 3, \"Buzz\" if divisible by 5, \"FizzBuzz\" "
                 "if divisible by\nboth, otherwise the number as a string. "
                 "(f 3) => \"Fizz\", (f 7) => \"7\".")
    :test-cases [{:args [3] :expected "Fizz"} {:args [5] :expected "Buzz"}
                 {:args [15] :expected "FizzBuzz"} {:args [7] :expected "7"}
                 {:args [9] :expected "Fizz"}]}

   {:id "gcd"
    :kind :function
    :system-prompt function-system-prompt
    :prompt (str "Write a ClojureScript function (fn [a b] ...) returning the greatest "
                 "common\ndivisor of two positive integers a and b. (f 12 8) => 4.")
    :test-cases [{:args [12 8] :expected 4} {:args [54 24] :expected 6}
                 {:args [7 1] :expected 1} {:args [100 75] :expected 25}]}

   {:id "palindrome?"
    :kind :function
    :system-prompt function-system-prompt
    :prompt (str "Write a ClojureScript function (fn [s] ...) returning true if the "
                 "string s\nreads the same forwards and backwards, else false. "
                 "(f \"racecar\") => true,\n(f \"hello\") => false.")
    :test-cases [{:args ["racecar"] :expected true} {:args ["hello"] :expected false}
                 {:args [""] :expected true} {:args ["abba"] :expected true}
                 {:args ["abc"] :expected false}]}

   {:id "sum-evens"
    :kind :function
    :system-prompt function-system-prompt
    :prompt (str "Write a ClojureScript function (fn [coll] ...) returning the sum of "
                 "the even\nnumbers in the collection coll. (f [1 2 3 4]) => 6.")
    :test-cases [{:args [[1 2 3 4]] :expected 6} {:args [[]] :expected 0}
                 {:args [[2 4 6]] :expected 12} {:args [[1 3 5]] :expected 0}
                 {:args [[10 -2 3]] :expected 8}]}])

;; ===========================================================================
;; The assembled seed set
;; ===========================================================================

(def tasks
  "The full seed task set (12): 4 conjugate programs + 3 state machines + 5 functions."
  (vec (concat program-tasks transition-tasks test-case-tasks)))

(def tasks-by-id
  "Map of :id -> task, for joining verdicts/candidates back to their prompts."
  (into {} (map (juxt :id identity)) tasks))

(defn task->prompt-record
  "The TEACHER-FACING projection of a task: only the fields the teacher needs to
   generate (id, kind, system + user prompt). The held-out oracle signal
   (:observations / :transitions / :test-cases) is intentionally dropped so it can
   never leak into a teacher prompt. Snake-cased keys for the Python/JSONL side."
  [task]
  {:task_id       (:id task)
   :kind          (name (:kind task))
   :system_prompt (:system-prompt task)
   :prompt        (:prompt task)})
