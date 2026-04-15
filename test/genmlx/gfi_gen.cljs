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
