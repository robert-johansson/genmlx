;; @tier exclude
(ns genmlx.test-helpers
  "Shared test utilities for the GenMLX test suite.
   Provides numerical comparison, MLX helpers, PRNG management,
   statistical testing, and cljs.test fixtures."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

;; ---------------------------------------------------------------------------
;; Numerical comparison
;; ---------------------------------------------------------------------------

(defn finite?
  "True if x is a finite number (not NaN, not Infinity)."
  [x]
  (and (number? x) (js/isFinite x)))

(defn close?
  "True if |expected - actual| <= tolerance."
  ([expected actual] (close? expected actual 1e-6))
  ([expected actual tol]
   (<= (js/Math.abs (- expected actual)) tol)))

(defn all-close?
  "True if every element pair is within tolerance. Works on seqs."
  ([expected actual] (all-close? expected actual 1e-6))
  ([expected actual tol]
   (and (= (count expected) (count actual))
        (every? true? (map #(close? %1 %2 tol) expected actual)))))

;; ---------------------------------------------------------------------------
;; Host-speed scaling
;; ---------------------------------------------------------------------------

(def time-scale
  "Host-speed multiplier for absolute wall-clock assertions. Ms budgets are
   tuned on Apple Silicon; slower hosts (Thor/CUDA aarch64) export
   TEST_TIME_SCALE=N — the same knob test/run.sh uses to scale tier caps
   (genmlx-9ox0) — and every timing assertion multiplies its budget by this
   (genmlx-y8zt). Default 1 keeps the Apple baselines intact."
  (let [s (js/parseFloat (or (.. js/process -env -TEST_TIME_SCALE) "1"))]
    (if (and (js/isFinite s) (pos? s)) s 1)))

(def par-degree
  "The tier's parallel degree, passed by test/run.sh as TEST_PAR (1 when a file
   runs solo). Under J-way parallelism GPU-bound files share the device and
   wall-clock inflates up to J-fold — run.sh scales its per-file CAPS by J for
   exactly this reason, and absolute-ms assertions must scale the same way or
   contention reads as a perf regression (genmlx-7yam, 8-way Metal battery)."
  (let [p (js/parseFloat (or (.. js/process -env -TEST_PAR) "1"))]
    (if (and (js/isFinite p) (pos? p)) p 1)))

(def wall-scale
  "Combined budget multiplier for ABSOLUTE wall-clock assertions:
   host speed (time-scale) x device contention (par-degree). Relative/ratio
   assertions should stay unscaled — both sides inflate together."
  (* time-scale par-degree))

;; ---------------------------------------------------------------------------
;; MLX helpers
;; ---------------------------------------------------------------------------

(defn realize
  "mx/eval! then mx/item. Returns JS number."
  [x]
  (mx/eval! x)
  (mx/item x))

(defn realize-shape
  "mx/eval! then mx/shape. Returns shape vector."
  [x]
  (mx/eval! x)
  (mx/shape x))

(defn realize-vec
  "mx/eval! then mx/->clj. Returns Clojure vector."
  [x]
  (mx/eval! x)
  (mx/->clj x))

;; ---------------------------------------------------------------------------
;; PRNG
;; ---------------------------------------------------------------------------

(defn deterministic-key
  "Fixed PRNG key for reproducible tests."
  ([] (rng/fresh-key 42))
  ([seed] (rng/fresh-key seed)))

;; ---------------------------------------------------------------------------
;; Statistical helpers
;; ---------------------------------------------------------------------------

(defn sample-mean
  "Mean of a seq of numbers."
  [xs]
  (/ (reduce + xs) (count xs)))

(defn sample-variance
  "Unbiased sample variance."
  [xs]
  (let [mu (sample-mean xs)
        n (count xs)]
    (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs))
       (dec n))))

(defn sample-std-error
  "Standard error of the mean."
  [xs]
  (js/Math.sqrt (/ (sample-variance xs) (count xs))))

(defn z-test-passes?
  "True if sample mean is within z-sigma standard errors of expected.
   Default z=3.5 gives false-positive rate < 0.0005."
  ([expected xs] (z-test-passes? expected xs 3.5))
  ([expected xs z]
   (let [se (sample-std-error xs)]
     (if (zero? se)
       (close? expected (sample-mean xs) 1e-6)
       (<= (js/Math.abs (/ (- (sample-mean xs) expected) se)) z)))))

;; ---------------------------------------------------------------------------
;; cljs.test fixture
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; Analytical log-prob helpers
;; ---------------------------------------------------------------------------

(def LOG-2PI
  "log(2π) — the normalization constant for gaussian log-prob."
  (js/Math.log (* 2 js/Math.PI)))

(defn gaussian-lp
  "Analytically compute log N(x; mu, sigma).
   = -0.5*log(2π) - log(σ) - 0.5*((x-μ)/σ)²"
  [x mu sigma]
  (let [z (/ (- x mu) sigma)]
    (- (* -0.5 LOG-2PI) (js/Math.log sigma) (* 0.5 z z))))

(def mlx-cleanup-fixture
  "Reusable :each fixture for MLX cleanup.
   Usage: (t/use-fixtures :each test-helpers/mlx-cleanup-fixture)"
  {:before (fn [] nil)
   :after (fn [] nil)})

;; ---------------------------------------------------------------------------
;; Model checkpoint resolution (genmlx-5z51)
;;
;; Checkpoint-gated tests used to accept a directory on `config.json` alone —
;; the one file an interrupted `hf download` leaves behind. On Thor the HF hub
;; entry for the 80B is exactly such a stub, so three heavy guards FAILED in
;; `load-model` ("Failed to load tokenizer") while the complete checkpoint sat
;; at the same snapshot hash under ~/code/mlx/models. Each host keeps its
;; checkpoints in a different layout, so the default is a candidate LIST and
;; the witness is what the loader actually reads (health-audit rule 3: accept
;; only what has been affirmatively proven).
;; ---------------------------------------------------------------------------

(defn complete-checkpoint?
  "True iff `dir` holds what `llm/load-model` reads: config.json,
   tokenizer.json and at least one *.safetensors weight file."
  [dir]
  (boolean
   (and (string? dir)
        (.existsSync fs (path/join dir "config.json"))
        (.existsSync fs (path/join dir "tokenizer.json"))
        (some #(.endsWith % ".safetensors") (.readdirSync fs dir)))))

(defn- snapshot-dirs
  "`<repo-dir>/snapshots/<rev>` when `rev` is pinned, else every snapshot."
  [repo-dir rev]
  (let [snaps (path/join repo-dir "snapshots")]
    (cond
      (not (.existsSync fs snaps)) []
      rev [(path/join snaps rev)]
      :else (map #(path/join snaps %) (sort (.readdirSync fs snaps))))))

(defn checkpoint-candidates
  "Directories that may hold the checkpoint `org/name`, in priority order:
   the HF hub cache, ~/code/mlx/models/<name> (HF snapshot layout — Thor),
   and — only when no revision is pinned — the flat ~/.cache/models symlink
   farm, whose links a later download can re-point."
  [{:keys [org name rev]}]
  (let [home (.homedir os)]
    (concat
     (snapshot-dirs (path/join home ".cache" "huggingface" "hub"
                               (str "models--" org "--" name)) rev)
     (snapshot-dirs (path/join home "code" "mlx" "models" name) rev)
     (when-not rev [(path/join home ".cache" "models" name)]))))

(defn resolve-checkpoint
  "The checkpoint directory for `spec` ({:org :name :rev?}): the env var
   `env-var` (may be nil) verbatim when set (an explicit override fails loudly rather than
   being second-guessed), else the first COMPLETE candidate, else nil — so a
   stub or partial download SKIPs instead of failing inside load-model."
  [env-var spec]
  (or (when env-var (not-empty (aget (.-env js/process) env-var)))
      (first (filter complete-checkpoint? (checkpoint-candidates spec)))))
