(ns genmlx.gfi-gen
  "test.check generators for GFI model specifications.
   Produces plain Clojure data maps that gfi-compiler transforms into
   executable DynamicGFs. Generators maintain topological ordering,
   unique addresses, and valid dependency references by construction."
  (:require [clojure.test.check.generators :as gen]))

;; ---------------------------------------------------------------------------
;; Primitives
;; ---------------------------------------------------------------------------

(def ^:private addr-pool [:a :b :c :d :e :f :g :h :i :j])

(def gen-dist-type
  (gen/elements [:gaussian :uniform :exponential :laplace :cauchy :delta :bernoulli]))

(def gen-param
  (gen/double* {:min -5.0 :max 5.0 :NaN? false :infinite? false}))

(def gen-pos-param
  (gen/double* {:min 0.1 :max 5.0 :NaN? false :infinite? false}))

(defn gen-fresh-addr
  "Keyword not yet in `used`. Falls back to :z0..:z99 when pool exhausted."
  [used]
  (let [fresh (vec (remove (set used) addr-pool))]
    (if (seq fresh)
      (gen/elements fresh)
      (gen/fmap #(keyword (str "z" %)) (gen/choose 0 99)))))

;; ---------------------------------------------------------------------------
;; Site arguments — distribution-aware
;; ---------------------------------------------------------------------------

(defn- gen-loc-arg
  "Location param: keyword dep-ref ~70% of the time when deps available."
  [available]
  (if (seq available)
    (gen/frequency [[7 (gen/elements available)]
                    [3 gen-param]])
    gen-param))

(defn- gen-site-args [dist-type available]
  (case dist-type
    :gaussian    (gen/tuple (gen-loc-arg available) gen-pos-param)
    :laplace     (gen/tuple (gen-loc-arg available) gen-pos-param)
    :cauchy      (gen/tuple (gen-loc-arg available) gen-pos-param)
    :uniform     (gen/bind gen-param
                   (fn [low]
                     (gen/fmap (fn [offset] [low (+ low offset)])
                       (gen/double* {:min 0.5 :max 5.0 :NaN? false :infinite? false}))))
    :exponential (gen/tuple gen-pos-param)
    :bernoulli   (gen/tuple (gen/double* {:min 0.01 :max 0.99 :NaN? false :infinite? false}))
    :delta       (if (seq available)
                   (gen/frequency [[5 (gen/tuple gen-param)]
                                   [5 (gen/tuple (gen/elements available))]])
                   (gen/tuple gen-param))))

;; ---------------------------------------------------------------------------
;; Single site
;; ---------------------------------------------------------------------------

(defn- deps-from-args [args]
  (vec (filter keyword? args)))

(defn gen-site
  "Site map with fresh addr, distribution-appropriate args, and correct deps.
   Keyword references in args only point to addresses in `available`."
  [available dist-type]
  (gen/bind (gen-fresh-addr available)
    (fn [addr]
      (gen/fmap
        (fn [args]
          {:addr addr :dist dist-type :args (vec args) :deps (deps-from-args args)})
        (gen-site-args dist-type available)))))

;; ---------------------------------------------------------------------------
;; Sequential site construction
;; ---------------------------------------------------------------------------

(defn gen-sites
  "Build a topologically-ordered site vector via reduce over gen/bind.
   Each step adds one site whose deps reference only earlier addrs."
  [dist-types]
  (reduce
    (fn [acc-gen [_ dtype]]
      (gen/bind acc-gen
        (fn [{:keys [sites available]}]
          (gen/fmap
            (fn [site]
              {:sites     (conj sites site)
               :available (conj available (:addr site))})
            (gen-site available dtype)))))
    (gen/return {:sites [] :available []})
    (map-indexed vector dist-types)))

;; ---------------------------------------------------------------------------
;; Complete model spec
;; ---------------------------------------------------------------------------

(def gen-model-spec
  "Generator for a complete model specification with 1-6 sites,
   topologically ordered, no duplicate addrs, valid dep references."
  (gen/bind (gen/choose 1 6)
    (fn [n]
      (gen/bind (gen/vector gen-dist-type n)
        (fn [dist-types]
          (gen/fmap
            (fn [{:keys [sites]}]
              {:sites sites :args [] :return (:addr (peek sites))})
            (gen-sites dist-types)))))))

(defn gen-sites-from
  "Like gen-sites but starts with initial available addresses."
  [dist-types initial-available]
  (reduce
    (fn [acc-gen [_ dtype]]
      (gen/bind acc-gen
        (fn [{:keys [sites available]}]
          (gen/fmap
            (fn [site]
              {:sites     (conj sites site)
               :available (conj available (:addr site))})
            (gen-site available dtype)))))
    (gen/return {:sites [] :available (vec initial-available)})
    (map-indexed vector dist-types)))

(def gen-model-spec-with-arg
  "Model spec with one argument :x. First site may use :x as a dependency."
  (gen/bind (gen/choose 1 4)
    (fn [n]
      (gen/bind (gen/vector gen-dist-type n)
        (fn [dist-types]
          (gen/fmap
            (fn [{:keys [sites]}]
              {:sites sites :args [:x] :return (:addr (peek sites))})
            (gen-sites-from dist-types [:x])))))))

;; ---------------------------------------------------------------------------
;; Combinator wrapper
;; ---------------------------------------------------------------------------

(def gen-continuous-dist-type
  "Distribution types safe for combinator kernels (no delta/bernoulli)."
  (gen/elements [:gaussian :uniform :exponential :laplace :cauchy]))

(def gen-kernel-spec
  "Model spec for use as a combinator kernel: one arg, continuous dists only."
  (gen/bind (gen/choose 1 3)
    (fn [n]
      (gen/bind (gen/vector gen-continuous-dist-type n)
        (fn [dist-types]
          (gen/fmap
            (fn [{:keys [sites]}]
              {:sites sites :args [:x] :return (:addr (peek sites))})
            (gen-sites-from dist-types [:x])))))))

(def gen-combinator-spec
  "~70% bare model spec, ~30% wrapped in a :map combinator descriptor."
  (gen/frequency
    [[7 gen-model-spec]
     [3 (gen/bind gen-model-spec
          (fn [spec]
            (gen/fmap
              (fn [n] {:combinator :map :kernel spec :n n})
              (gen/choose 2 5))))]]))

;; ---------------------------------------------------------------------------
;; Gap 1: Partial constraint inputs
;; ---------------------------------------------------------------------------

(def gen-continuous-model-spec
  "Model spec using only continuous distributions (no delta/bernoulli).
   Used for partial-constraint testing where delta sites with resampled
   keyword-dep parents produce -∞ log-prob (correct but vacuous)."
  (gen/bind (gen/choose 2 6)
    (fn [n]
      (gen/bind (gen/vector gen-continuous-dist-type n)
        (fn [dist-types]
          (gen/fmap
            (fn [{:keys [sites]}]
              {:sites sites :args [] :return (:addr (peek sites))})
            (gen-sites dist-types)))))))

(def gen-partial-constraint-input
  "Model spec (>= 2 continuous sites) paired with a random non-empty strict
   subset of addresses to constrain. Uses a boolean mask for uniform coverage
   over all possible subsets (including constraining dependent sites while
   parents are free). Excludes delta/bernoulli to avoid -∞ log-prob when
   a delta site's keyword-dep parent is resampled."
  (gen/bind gen-continuous-model-spec
    (fn [spec]
      (let [addrs (mapv :addr (:sites spec))
            n (count addrs)]
        (gen/fmap
          (fn [mask]
            {:spec spec
             :constrained-addrs (set (keep-indexed
                                       (fn [i b] (when b (nth addrs i)))
                                       mask))})
          ;; Boolean mask with at least one true and one false
          (gen/such-that
            (fn [mask] (and (some true? mask) (some false? mask)))
            (gen/vector gen/boolean n)
            100))))))

;; ---------------------------------------------------------------------------
;; Gap 2: Differentiable models with argument
;; ---------------------------------------------------------------------------

(def gen-differentiable-spec-with-arg
  "Differentiable model with one argument :x and at least one site using :x.
   All distributions are gaussian/laplace/cauchy (differentiable with reparameterization).
   Built by construction via gen-sites-from with :x in the initial available pool,
   then filtered for at least one site that actually references :x."
  (gen/such-that
    (fn [spec] (some (fn [s] (some #{:x} (:args s))) (:sites spec)))
    (gen/bind (gen/choose 1 4)
      (fn [n]
        (gen/bind (gen/vector (gen/elements [:gaussian :laplace :cauchy]) n)
          (fn [dist-types]
            (gen/fmap
              (fn [{:keys [sites]}]
                {:sites sites :args [:x] :return (:addr (peek sites))})
              (gen-sites-from dist-types [:x]))))))
    100))

;; ---------------------------------------------------------------------------
;; Gap 3: Branching model specs
;; ---------------------------------------------------------------------------

(def gen-branching-spec
  "Branching model: bernoulli coin with separate true/false branch sites.
   Uses fixed address pools (:th/:ti for true branch, :fh/:fi for false branch)
   to avoid collisions by construction. Branch sites are independent (no
   inter-site dependencies) with continuous distributions only."
  (gen/bind (gen/tuple (gen/choose 1 2)
                       (gen/choose 1 2)
                       (gen/double* {:min 0.2 :max 0.8 :NaN? false :infinite? false}))
    (fn [[n-true n-false coin-prob]]
      (gen/bind
        (gen/tuple
          (gen/vector gen-continuous-dist-type n-true)
          (gen/vector gen-continuous-dist-type n-false))
        (fn [[true-dists false-dists]]
          (gen/bind
            (gen/tuple
              (apply gen/tuple (map #(gen-site-args % []) true-dists))
              (apply gen/tuple (map #(gen-site-args % []) false-dists)))
            (fn [[true-arg-seqs false-arg-seqs]]
              (gen/return
                {:type :branching
                 :pre-sites []
                 :branch {:addr :coin :dist :bernoulli :args [coin-prob]}
                 :true-sites (mapv (fn [dt args addr]
                                     {:addr addr :dist dt :args (vec args) :deps []})
                                   true-dists true-arg-seqs [:th :ti])
                 :false-sites (mapv (fn [dt args addr]
                                      {:addr addr :dist dt :args (vec args) :deps []})
                                    false-dists false-arg-seqs [:fh :fi])
                 :args []}))))))))
