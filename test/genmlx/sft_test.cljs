(ns genmlx.sft-test
  "Tests for the pure SFT core (genmlx.world.sft, genmlx-o8w9): the leakage-guarded
   train/eval task split, mlx-lm corpus shaping, the Chen-et-al. pass@k estimator, and
   the baseline-vs-SFT report. Pure data + arithmetic — no GPU, no LLM, no I/O.

   Run: bun run --bun nbb test/genmlx/sft_test.cljs"
  (:require [genmlx.world.sft :as sft]
            [genmlx.world.distill-tasks :as t]
            [clojure.set]
            [clojure.string :as str]))

(def ^:private fails (atom 0))
(def ^:private passes (atom 0))

(defn assert-true [desc x]
  (if x (do (swap! passes inc) (println (str "  ok   " desc)))
        (do (swap! fails inc) (println (str "  FAIL " desc)))))

(defn assert-eq [desc expected actual]
  (assert-true (str desc " (expected " (pr-str expected) ", got " (pr-str actual) ")")
               (= expected actual)))

(defn assert-close [desc expected actual tol]
  (assert-true (str desc " (expected " expected ", got " actual ")")
               (< (js/Math.abs (- expected actual)) tol)))

(defn assert-throws [desc thunk]
  (assert-true desc (try (thunk) false (catch :default _ true))))

;; ===========================================================================
(println "\n== 1. train/eval task split ==")

(let [{:keys [train eval]} (sft/split-tasks t/tasks)]
  (assert-eq "12 seed tasks total" 12 (count t/tasks))
  (assert-eq "3 held-out eval tasks" 3 (count eval))
  (assert-eq "9 train tasks" 9 (count train))
  (assert-true "train/eval are disjoint by id"
               (empty? (clojure.set/intersection
                         (set (map :id train)) (set (map :id eval)))))
  (assert-true "eval set is exactly the canonical ids"
               (= sft/eval-task-ids (set (map :id eval))))
  (assert-true "eval covers BOTH oracle kinds (program + function)"
               (= #{:program :function} (set (map :kind eval))))
  (assert-true "each eval task has same-family train siblings"
               (every? (fn [et]
                         (some #(= (:kind et) (:kind %)) train))
                       eval)))

(assert-true "eval-task? true for a held-out id"  (sft/eval-task? "palindrome?"))
(assert-true "eval-task? false for a train id"     (not (sft/eval-task? "factorial")))
(assert-true "train-task? is the complement"       (sft/train-task? "factorial"))
(assert-true "eval-task? accepts a task map"        (sft/eval-task? {:id "traffic-light"}))

;; ===========================================================================
(println "\n== 2. corpus prep / leakage guard ==")

(def ^:private sample-rows
  [{:task-id "factorial" :kind "function" :rank-key 1.0
    :messages [{:role "user" :content "fac"} {:role "assistant" :content "(fn [n] 1)"}]}
   {:task-id "gcd" :kind "function" :rank-key 1.0
    :messages [{:role "user" :content "gcd"} {:role "assistant" :content "(fn [a b] a)"}]}
   {:task-id "gaussian-mean-near2" :kind "program" :rank-key -8.0
    :messages [{:role "user" :content "g"} {:role "assistant" :content "(fn [trace] {})"}]}])

(let [m (sft/row->messages (first sample-rows))]
  (assert-eq "row->messages keeps only :messages" [:messages] (keys m))
  (assert-true "row->messages preserves the conversation"
               (= (:messages (first sample-rows)) (:messages m))))

(assert-true "assert-train-disjoint! passes a clean (train-only) corpus"
             (= sample-rows (sft/assert-train-disjoint! sample-rows)))

(assert-throws "assert-train-disjoint! throws when an eval-task row leaks in"
               #(sft/assert-train-disjoint!
                  (conj sample-rows
                        {:task-id "palindrome?" :kind "function" :rank-key 1.0
                         :messages [{:role "assistant" :content "(fn [s] true)"}]})))

(let [leaky (conj sample-rows
                  {:task-id "traffic-light" :kind "function" :rank-key 1.0
                   :messages [{:role "assistant" :content "x"}]})
      {:keys [train-rows dropped-eval eval-task-ids-present]} (sft/partition-corpus leaky)]
  (assert-eq "partition-corpus keeps the 3 train rows" 3 (count train-rows))
  (assert-eq "partition-corpus drops the 1 eval row" 1 (count dropped-eval))
  (assert-eq "partition-corpus reports which eval id leaked"
             #{"traffic-light"} (set eval-task-ids-present))
  (assert-true "the partitioned train-rows survive the disjointness assertion"
               (= train-rows (sft/assert-train-disjoint! train-rows))))

;; ===========================================================================
(println "\n== 3. valid-split / blend ==")

(let [rows (mapv (fn [i] {:messages [{:role "assistant" :content (str i)}]}) (range 10))
      {:keys [train valid]} (sft/valid-split rows 0.2)]
  (assert-true "valid-split: train non-empty" (seq train))
  (assert-true "valid-split: valid non-empty" (seq valid))
  (assert-eq "valid-split: no rows lost" 10 (+ (count train) (count valid)))
  (assert-true "valid-split: train and valid are disjoint"
               (empty? (clojure.set/intersection (set train) (set valid)))))

(let [{:keys [train valid]} (sft/valid-split [{:messages [:only]}] 0.2)]
  (assert-true "valid-split degenerate (1 row): both files non-empty"
               (and (seq train) (seq valid))))

(let [head [{:messages [:d0]} {:messages [:d1]}]
      vol  (mapv (fn [i] {:messages [(keyword (str "v" i))]}) (range 100))
      out  (sft/blend head vol 5)]
  (assert-eq "blend: head leads, n volume appended" 7 (count out))
  (assert-eq "blend: distilled rows come first" head (vec (take 2 out)))
  (assert-eq "blend: takes the first n volume rows" {:messages [:v0]} (nth out 2)))

;; ===========================================================================
(println "\n== 4. pass@k estimator (Chen et al. 2021) ==")

(assert-close "pass@k: c=0 -> 0"            0.0 (sft/pass-at-k 5 0 1) 1e-9)
(assert-close "pass@k: all correct -> 1"    1.0 (sft/pass-at-k 5 5 1) 1e-9)
(assert-close "pass@1, n=5 c=1 -> 1/5"      0.2 (sft/pass-at-k 5 1 1) 1e-9)
(assert-close "pass@1, n=5 c=2 -> 2/5"      0.4 (sft/pass-at-k 5 2 1) 1e-9)
(assert-close "pass@2, n=5 c=1 -> 0.4"      0.4 (sft/pass-at-k 5 1 2) 1e-9)
(assert-close "pass@2, n=5 c=2 -> 0.7"      0.7 (sft/pass-at-k 5 2 2) 1e-9)
(assert-close "pass@k: n-c<k -> 1 (k=5,c=1)" 1.0 (sft/pass-at-k 5 1 5) 1e-9)
(assert-close "pass@1, n=200 c=100 -> 0.5"  0.5 (sft/pass-at-k 200 100 1) 1e-9)

(let [mk (fn [idx kept?] {:task-id "x" :kind :function :sample-idx idx :kept? kept?})
      verdicts [(mk 0 false) (mk 1 true) (mk 2 false) (mk 3 true) (mk 4 false)]
      r (sft/passk-of verdicts 4)]
  (assert-eq "passk-of: n = sampled count (idx>=1)" 4 (:n r))
  (assert-eq "passk-of: c = kept among sampled" 2 (:c r))
  (assert-close "passk-of: greedy (idx 0) failed -> pass@1-greedy 0" 0.0 (:pass1-greedy r) 1e-9)
  (assert-close "passk-of: pass@4 with c=2/4 -> 1.0" 1.0 (:passk r) 1e-9))

(let [mk (fn [idx kept?] {:task-id "x" :kind :function :sample-idx idx :kept? kept?})
      verdicts [(mk 0 true) (mk 1 false) (mk 2 false)]
      r (sft/passk-of verdicts 2)]
  (assert-close "passk-of: greedy passed -> pass@1-greedy 1" 1.0 (:pass1-greedy r) 1e-9)
  (assert-close "passk-of: pass@2 with c=0/2 -> 0" 0.0 (:passk r) 1e-9))

;; ===========================================================================
(println "\n== 5. eval-report / cold-start guard ==")

(let [mk (fn [tid kind idx kept?] {:task-id tid :kind kind :sample-idx idx :kept? kept?})
      tasks-by-id {"palindrome?" {:id "palindrome?" :kind :function}
                   "gaussian-mean-negshift" {:id "gaussian-mean-negshift" :kind :program}}
      ;; palindrome?: baseline 0 kept of 3 (cold for greedy, but 0 sampled kept -> cold-start),
      ;;              sft 2 kept of 3 + greedy kept  -> a lift
      baseline (concat [(mk "palindrome?" :function 0 false)
                        (mk "palindrome?" :function 1 false)
                        (mk "palindrome?" :function 2 false)]
                       [(mk "gaussian-mean-negshift" :program 0 true)
                        (mk "gaussian-mean-negshift" :program 1 true)
                        (mk "gaussian-mean-negshift" :program 2 false)])
      sft-v    (concat [(mk "palindrome?" :function 0 true)
                        (mk "palindrome?" :function 1 true)
                        (mk "palindrome?" :function 2 false)]
                       [(mk "gaussian-mean-negshift" :program 0 true)
                        (mk "gaussian-mean-negshift" :program 1 true)
                        (mk "gaussian-mean-negshift" :program 2 true)])
      report (sft/eval-report baseline sft-v 2 tasks-by-id)
      pal    (first (filter #(= "palindrome?" (:task-id %)) (:tasks report)))]
  (assert-eq "eval-report: 2 eval tasks" 2 (:n-tasks (:aggregate report)))
  (assert-true "eval-report: palindrome? flagged base-cold (baseline pass@k 0)"
               (:cold-start? pal))
  (assert-true "eval-report: palindrome? NOT sft-cold (SFT lifted it off zero)"
               (not (:sft-cold? pal)))
  (assert-eq "eval-report: 1 base-cold task" 1 (:n-cold-start (:aggregate report)))
  (assert-eq "eval-report: 0 sft-cold tasks (SFT reaches both)" 0 (:n-sft-cold (:aggregate report)))
  (assert-true "eval-report: SFT pass@1 >= baseline pass@1 in aggregate"
               (>= (:sft-pass1 (:aggregate report)) (:baseline-pass1 (:aggregate report))))
  (assert-true "eval-report: SFT shows a non-negative pass@k delta"
               (>= (:delta-passk (:aggregate report)) 0.0))
  (assert-true "eval-report: per-kind breakdown has program + function"
               (= #{"program" "function"} (set (keys (:by-kind report))))))

;; ===========================================================================
(println (str "\n== SUMMARY: " @passes " passed, " @fails " failed =="))
(when (pos? @fails) (set! (.-exitCode js/process) 1))
