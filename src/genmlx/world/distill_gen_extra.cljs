(ns genmlx.world.distill-gen-extra
  "Extra function specs for the scaled cljs-coder task set (genmlx-7473), authored by the
   catalog-expansion workflow (5 agents, oracle-safe ClojureScript function specs across
   number-theory / sequence / string / logic / applied-math categories). DATA ONLY — these
   are appended to genmlx.world.distill-gen/function-specs; any spec whose reference fails
   its own oracle is pruned by distill-gen (see scripts/gen_tasks.cljs --validate and
   test/genmlx/distill_gen_test.cljs). No requires: the :reference / :desc fields are plain
   strings, evaluated later by the SCI oracle, not by this namespace.")

(def extra-function-specs
  [
   {:id "nth-prime" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)" :desc "that returns the n-th prime number, where n is 1-indexed so n=1 gives 2; treat n=0 as the seed value 1" :reference "(fn [n] (letfn [(prime? [k] (and (> k 1) (loop [d 2] (cond (> (* d d) k) true (zero? (mod k d)) false :else (recur (inc d))))))] (loop [cnt 0 k 1] (if (= cnt n) k (let [k2 (inc k)] (if (prime? k2) (recur (inc cnt) k2) (recur cnt k2)))))))" :tests [[[1] 2] [[2] 3] [[3] 5] [[6] 13] [[0] 1]]}

   {:id "count-divisors" :family :arithmetic :difficulty :easy :sig "(fn [n] ...)" :desc "that returns how many positive integers divide n exactly (including 1 and n itself)" :reference "(fn [n] (loop [d 1 c 0] (cond (> d n) c (zero? (mod n d)) (recur (inc d) (inc c)) :else (recur (inc d) c))))" :tests [[[1] 1] [[6] 4] [[12] 6] [[7] 2] [[28] 6]]}

   {:id "sum-divisors" :family :arithmetic :difficulty :easy :sig "(fn [n] ...)" :desc "that returns the sum of all positive divisors of n (including 1 and n itself)" :reference "(fn [n] (loop [d 1 s 0] (cond (> d n) s (zero? (mod n d)) (recur (inc d) (+ s d)) :else (recur (inc d) s))))" :tests [[[1] 1] [[6] 12] [[12] 28] [[7] 8] [[10] 18]]}

   {:id "aliquot-sum" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)" :desc "that returns the aliquot sum of n: the sum of its proper divisors (all positive divisors strictly less than n)" :reference "(fn [n] (loop [d 1 s 0] (cond (>= d n) s (zero? (mod n d)) (recur (inc d) (+ s d)) :else (recur (inc d) s))))" :tests [[[1] 0] [[6] 6] [[12] 16] [[28] 28] [[10] 8]]}

   {:id "perfect-number?" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)" :desc "that returns true when n is a perfect number, meaning n equals the sum of its proper divisors (e.g. 6 = 1+2+3)" :reference "(fn [n] (and (> n 1) (= n (loop [d 1 s 0] (cond (>= d n) s (zero? (mod n d)) (recur (inc d) (+ s d)) :else (recur (inc d) s))))))" :tests [[[6] true] [[28] true] [[12] false] [[1] false] [[496] true]]}

   {:id "abundant?" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)" :desc "that returns true when n is abundant, meaning the sum of its proper divisors is greater than n itself" :reference "(fn [n] (> (loop [d 1 s 0] (cond (>= d n) s (zero? (mod n d)) (recur (inc d) (+ s d)) :else (recur (inc d) s))) n))" :tests [[[12] true] [[6] false] [[18] true] [[7] false] [[1] false]]}

   {:id "prime-factors" :family :arithmetic :difficulty :hard :sig "(fn [n] ...)" :desc "that returns a vector of the prime factors of n in nondecreasing order, with repetition, so they multiply back to n (returns [] for n=1)" :reference "(fn [n] (loop [n n d 2 acc []] (cond (< n 2) acc (> (* d d) n) (conj acc n) (zero? (mod n d)) (recur (quot n d) d (conj acc d)) :else (recur n (inc d) acc))))" :tests [[[12] [2 2 3]] [[17] [17]] [[1] []] [[60] [2 2 3 5]] [[8] [2 2 2]]]}

   {:id "distinct-prime-factor-count" :family :arithmetic :difficulty :hard :sig "(fn [n] ...)" :desc "that returns how many distinct prime factors n has (e.g. 12 = 2*2*3 has 2 distinct primes), returning 0 for n=1" :reference "(fn [n] (loop [n n d 2 c 0] (cond (< n 2) c (> (* d d) n) (inc c) (zero? (mod n d)) (recur (loop [m n] (if (zero? (mod m d)) (recur (quot m d)) m)) (inc d) (inc c)) :else (recur n (inc d) c))))" :tests [[[12] 2] [[60] 3] [[17] 1] [[1] 0] [[30] 3]]}

   {:id "integer-sqrt" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)" :desc "that returns the integer square root of n, the largest integer k such that k*k <= n (floor of the real square root)" :reference "(fn [n] (loop [k 0] (if (> (* (inc k) (inc k)) n) k (recur (inc k)))))" :tests [[[0] 0] [[1] 1] [[15] 3] [[16] 4] [[26] 5]]}

   {:id "perfect-square?" :family :arithmetic :difficulty :easy :sig "(fn [n] ...)" :desc "that returns true when n is a perfect square (n equals some integer multiplied by itself), and false otherwise" :reference "(fn [n] (and (>= n 0) (loop [k 0] (let [sq (* k k)] (cond (= sq n) true (> sq n) false :else (recur (inc k)))))))" :tests [[[0] true] [[1] true] [[16] true] [[15] false] [[25] true]]}

   {:id "digit-product" :family :arithmetic :difficulty :easy :sig "(fn [n] ...)" :desc "that returns the product of the base-10 digits of n (using its absolute value); the product of no digits, for n=0, is 1 only via the empty loop but here 0 yields 1" :reference "(fn [n] (loop [n (if (neg? n) (- n) n) p 1] (if (zero? n) p (recur (quot n 10) (* p (mod n 10))))))" :tests [[[123] 6] [[0] 1] [[405] 0] [[9] 9] [[111] 1]]}

   {:id "count-digits" :family :arithmetic :difficulty :easy :sig "(fn [n] ...)" :desc "that returns how many base-10 digits n has, using its absolute value, where 0 counts as having one digit" :reference "(fn [n] (let [n (if (neg? n) (- n) n)] (if (zero? n) 1 (loop [n n c 0] (if (zero? n) c (recur (quot n 10) (inc c)))))))" :tests [[[0] 1] [[5] 1] [[123] 3] [[1000] 4] [[-42] 2]]}

   {:id "reverse-number" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)" :desc "that returns the integer formed by reversing the base-10 digits of n (using its absolute value), dropping any leading zeros that result" :reference "(fn [n] (loop [n (if (neg? n) (- n) n) r 0] (if (zero? n) r (recur (quot n 10) (+ (* r 10) (mod n 10))))))" :tests [[[123] 321] [[0] 0] [[100] 1] [[5] 5] [[1200] 21]]}

   {:id "digit-sum-base" :family :arithmetic :difficulty :medium :sig "(fn [n b] ...)" :desc "that returns the sum of the digits of n when written in base b (using the absolute value of n)" :reference "(fn [n b] (loop [n (if (neg? n) (- n) n) s 0] (if (zero? n) s (recur (quot n b) (+ s (mod n b))))))" :tests [[[255 16] 30] [[7 2] 3] [[10 10] 1] [[0 2] 0] [[8 2] 1]]}

   {:id "multiplicative-persistence" :family :arithmetic :difficulty :hard :sig "(fn [n] ...)" :desc "that returns the multiplicative persistence of n: how many times you must replace the number with the product of its base-10 digits before reaching a single digit" :reference "(fn [n] (letfn [(dprod [m] (loop [m m p 1] (if (zero? m) p (recur (quot m 10) (* p (mod m 10))))))] (loop [n (if (neg? n) (- n) n) c 0] (if (< n 10) c (recur (dprod n) (inc c))))))" :tests [[[7] 0] [[39] 3] [[25] 2] [[10] 1] [[77] 4]]}

   {:id "range-product" :family :arithmetic :difficulty :easy :sig "(fn [a b] ...)" :desc "that returns the product of all integers from a to b inclusive, returning 1 when a is greater than b (empty range)" :reference "(fn [a b] (if (> a b) 1 (loop [k a p 1] (if (> k b) p (recur (inc k) (* p k))))))" :tests [[[1 5] 120] [[3 3] 3] [[5 4] 1] [[2 4] 24] [[1 1] 1]]}

   {:id "coprime?" :family :arithmetic :difficulty :medium :sig "(fn [a b] ...)" :desc "that returns true when a and b are coprime, meaning their greatest common divisor is 1 (uses absolute values)" :reference "(fn [a b] (= 1 (loop [a (if (neg? a) (- a) a) b (if (neg? b) (- b) b)] (if (zero? b) a (recur b (mod a b))))))" :tests [[[8 9] true] [[8 12] false] [[1 5] true] [[14 21] false] [[17 5] true]]}

   {:id "gcd-of-list" :family :arithmetic :difficulty :medium :sig "(fn [xs] ...)" :desc "that returns the greatest common divisor of all integers in vector xs (using their absolute values), returning 0 for an empty vector" :reference "(fn [xs] (letfn [(g [a b] (if (zero? b) a (recur b (mod a b))))] (if (empty? xs) 0 (reduce (fn [acc x] (g acc (if (neg? x) (- x) x))) (let [f (first xs)] (if (neg? f) (- f) f)) (rest xs)))))" :tests [[[[12 18 24]] 6] [[[7]] 7] [[[]] 0] [[[10 20 35]] 5] [[[13 26]] 13]]}

   {:id "lcm-of-list" :family :arithmetic :difficulty :hard :sig "(fn [xs] ...)" :desc "that returns the least common multiple of all integers in vector xs, returning 1 for an empty vector" :reference "(fn [xs] (letfn [(g [a b] (if (zero? b) a (recur b (mod a b)))) (l [a b] (if (or (zero? a) (zero? b)) 0 (quot (* a b) (g a b))))] (if (empty? xs) 1 (reduce l (first xs) (rest xs)))))" :tests [[[[2 3 4]] 12] [[[5]] 5] [[[]] 1] [[[6 8]] 24] [[[1 2 3 4 5]] 60]]}

   {:id "clamp-range" :family :arithmetic :difficulty :easy :sig "(fn [x lo hi] ...)" :desc "that clamps integer x into the inclusive range [lo, hi], returning lo if x is below it, hi if above it, otherwise x" :reference "(fn [x lo hi] (cond (< x lo) lo (> x hi) hi :else x))" :tests [[[5 0 10] 5] [[-3 0 10] 0] [[15 0 10] 10] [[0 0 10] 0] [[10 0 10] 10]]}

   {:id "sliding-windows-2" :family :collection :difficulty :easy :sig "(fn [v] ...)"
 :desc "that returns a vector of all consecutive overlapping pairs (windows of size 2) of the input vector, each window itself a vector"
 :reference "(fn [v] (mapv vec (partition 2 1 v)))"
 :tests [[[[1 2 3 4]] [[1 2] [2 3] [3 4]]] [[[5 6]] [[5 6]]] [[[7]] []] [[[]] []] [[[1 1 1]] [[1 1] [1 1]]]]}

   {:id "chunk-into-n" :family :collection :difficulty :easy :sig "(fn [n v] ...)"
 :desc "that partitions the vector v into consecutive chunks of size n (the final chunk may be shorter), returning a vector of vectors"
 :reference "(fn [n v] (mapv vec (partition-all n v)))"
 :tests [[[2 [1 2 3 4 5]] [[1 2] [3 4] [5]]] [[3 [1 2 3 4 5 6]] [[1 2 3] [4 5 6]]] [[2 []] []] [[1 [9 8]] [[9] [8]]] [[5 [1 2 3]] [[1 2 3]]]]}

   {:id "zip-pairs" :family :collection :difficulty :easy :sig "(fn [a b] ...)"
 :desc "that zips two vectors a and b into a vector of [a-elem b-elem] pairs, stopping at the shorter length"
 :reference "(fn [a b] (mapv vector a b))"
 :tests [[[[1 2 3] [4 5 6]] [[1 4] [2 5] [3 6]]] [[[1 2] [9 8 7]] [[1 9] [2 8]]] [[[] [1 2]] []] [[[1] [2]] [[1 2]]] [[[1 2 3] []] []]]}

   {:id "interleave-vec" :family :collection :difficulty :easy :sig "(fn [a b] ...)"
 :desc "that interleaves two vectors a and b into a single flat vector (a0 b0 a1 b1 ...), stopping at the shorter length"
 :reference "(fn [a b] (vec (interleave a b)))"
 :tests [[[[1 2 3] [4 5 6]] [1 4 2 5 3 6]] [[[1 2] [9 8 7]] [1 9 2 8]] [[[] [1]] []] [[[1] [2]] [1 2]] [[[1 2 3] []] []]]}

   {:id "rotate-left-n" :family :collection :difficulty :medium :sig "(fn [n v] ...)"
 :desc "that rotates the vector v to the left by n positions (elements shifted off the front wrap to the back), returning a vector"
 :reference "(fn [n v] (if (empty? v) [] (let [k (mod n (count v))] (vec (concat (drop k v) (take k v))))))"
 :tests [[[1 [1 2 3 4]] [2 3 4 1]] [[2 [1 2 3 4]] [3 4 1 2]] [[0 [1 2 3]] [1 2 3]] [[5 [1 2 3 4]] [2 3 4 1]] [[3 []] []]]}

   {:id "run-lengths" :family :collection :difficulty :medium :sig "(fn [v] ...)"
 :desc "that run-length-encodes the vector v into a vector of [value count] pairs for each maximal run of equal consecutive elements"
 :reference "(fn [v] (mapv (fn [g] [(first g) (count g)]) (partition-by identity v)))"
 :tests [[[[1 1 2 3 3 3]] [[1 2] [2 1] [3 3]]] [[[5 5 5]] [[5 3]]] [[[1 2 3]] [[1 1] [2 1] [3 1]]] [[[]] []] [[[7]] [[7 1]]]]}

   {:id "flatten-one-level" :family :collection :difficulty :medium :sig "(fn [v] ...)"
 :desc "that flattens a vector of vectors by exactly one level, concatenating the inner vectors into one flat vector"
 :reference "(fn [v] (vec (apply concat v)))"
 :tests [[[[[1 2] [3 4]]] [1 2 3 4]] [[[[1] [2 3] [4 5 6]]] [1 2 3 4 5 6]] [[[]] []] [[[[] [1] []]] [1]] [[[[1 2 3]]] [1 2 3]]]}

   {:id "take-then-drop" :family :collection :difficulty :easy :sig "(fn [n v] ...)"
 :desc "that returns a vector containing two parts: the first n elements of v, then the elements of v after dropping the first n, packaged as [taken-vec dropped-vec]"
 :reference "(fn [n v] [(vec (take n v)) (vec (drop n v))])"
 :tests [[[2 [1 2 3 4 5]] [[1 2] [3 4 5]]] [[0 [1 2 3]] [[] [1 2 3]]] [[5 [1 2 3]] [[1 2 3] []]] [[1 []] [[] []]] [[3 [9 8 7 6]] [[9 8 7] [6]]]]}

   {:id "count-above" :family :collection :difficulty :easy :sig "(fn [t v] ...)"
 :desc "that counts how many elements of the integer vector v are strictly greater than the threshold t"
 :reference "(fn [t v] (count (filter (fn [x] (> x t)) v)))"
 :tests [[[3 [1 2 3 4 5]] 2] [[0 [-1 -2 5 0 7]] 2] [[10 [1 2 3]] 0] [[0 []] 0] [[2 [2 2 2]] 0]]}

   {:id "second-smallest" :family :collection :difficulty :medium :sig "(fn [v] ...)"
 :desc "that returns the second-smallest value in the integer vector v (by sorted position, allowing duplicates); returns -1 if v has fewer than two elements"
 :reference "(fn [v] (if (< (count v) 2) -1 (second (sort v))))"
 :tests [[[[3 1 2]] 2] [[[5 5 1]] 5] [[[10 -3 4 -3]] -3] [[[7]] -1] [[[]] -1]]}

   {:id "build-step-range" :family :collection :difficulty :easy :sig "(fn [start n step] ...)"
 :desc "that builds a vector of n integers beginning at start and increasing by step each time"
 :reference "(fn [start n step] (vec (take n (iterate (fn [x] (+ x step)) start))))"
 :tests [[[0 5 1] [0 1 2 3 4]] [[2 4 3] [2 5 8 11]] [[10 0 5] []] [[7 1 2] [7]] [[0 3 -2] [0 -2 -4]]]}

   {:id "cumulative-sum" :family :collection :difficulty :medium :sig "(fn [v] ...)"
 :desc "that returns a vector of running (prefix) sums of the integer vector v, where element i is the sum of v[0..i]"
 :reference "(fn [v] (vec (rest (reductions + 0 v))))"
 :tests [[[[1 2 3 4]] [1 3 6 10]] [[[5 -2 7]] [5 3 10]] [[[]] []] [[[9]] [9]] [[[0 0 0]] [0 0 0]]]}

   {:id "pairwise-diffs" :family :collection :difficulty :medium :sig "(fn [v] ...)"
 :desc "that returns a vector of differences between each pair of consecutive elements (v[i+1] - v[i]) of the integer vector v"
 :reference "(fn [v] (mapv (fn [[a b]] (- b a)) (partition 2 1 v)))"
 :tests [[[[1 3 6 10]] [2 3 4]] [[[5 2 9]] [-3 7]] [[[7]] []] [[[]] []] [[[4 4 4]] [0 0]]]}

   {:id "index-of-val" :family :collection :difficulty :medium :sig "(fn [x v] ...)"
 :desc "that returns the index of the first occurrence of x in the vector v, or -1 if x is not present"
 :reference "(fn [x v] (loop [i 0 s (seq v)] (cond (nil? s) -1 (= (first s) x) i :else (recur (inc i) (next s)))))"
 :tests [[[3 [1 2 3 4]] 2] [[1 [1 1 1]] 0] [[9 [1 2 3]] -1] [[5 []] -1] [[4 [0 4 4]] 1]]}

   {:id "contains-val?" :family :collection :difficulty :easy :sig "(fn [x v] ...)"
 :desc "that returns true if x is an element of the vector v and false otherwise"
 :reference "(fn [x v] (boolean (some (fn [y] (= y x)) v)))"
 :tests [[[3 [1 2 3]] true] [[9 [1 2 3]] false] [[1 []] false] [[0 [0]] true] [[2 [1 2 2 2]] true]]}

   {:id "most-common-count" :family :collection :difficulty :hard :sig "(fn [v] ...)"
 :desc "that returns the count of the most frequently occurring integer in the vector v (the size of the largest group of equal elements); returns 0 for an empty vector"
 :reference "(fn [v] (if (empty? v) 0 (apply max (vals (frequencies v)))))"
 :tests [[[[1 1 2 3 1]] 3] [[[4 5 6]] 1] [[[2 2 2 2]] 4] [[[]] 0] [[[7 7 8 8]] 2]]}

   {:id "char-count" :family :string :difficulty :easy :sig "(fn [s] ...)"
 :desc "that returns the number of characters in string s"
 :reference "(fn [s] (count s))"
 :tests [[["hello"] 5] [[""] 0] [["a"] 1] [["abc def"] 7]]}

   {:id "word-count" :family :string :difficulty :easy :sig "(fn [s] ...)"
 :desc "that counts the words in string s (whitespace-separated)"
 :reference "(fn [s] (if (clojure.string/blank? s) 0 (count (clojure.string/split (clojure.string/trim s) #\"\\s+\"))))"
 :tests [[["hello world"] 2] [[""] 0] [["   "] 0] [["one"] 1] [["  a  b  c  "] 3]]}

   {:id "reverse-words" :family :string :difficulty :medium :sig "(fn [s] ...)"
 :desc "that reverses the order of the words in string s"
 :reference "(fn [s] (if (clojure.string/blank? s) \"\" (clojure.string/join \" \" (reverse (clojure.string/split (clojure.string/trim s) #\"\\s+\")))))"
 :tests [[["hello world"] "world hello"] [[""] ""] [["one"] "one"] [["a b c"] "c b a"]]}

   {:id "capitalize-first" :family :string :difficulty :easy :sig "(fn [s] ...)"
 :desc "that uppercases the first character of string s and leaves the rest unchanged"
 :reference "(fn [s] (if (= s \"\") \"\" (str (clojure.string/upper-case (subs s 0 1)) (subs s 1))))"
 :tests [[["hello"] "Hello"] [[""] ""] [["a"] "A"] [["hello world"] "Hello world"]]}

   {:id "count-uppercase" :family :string :difficulty :medium :sig "(fn [s] ...)"
 :desc "that counts the uppercase letters in string s"
 :reference "(fn [s] (count (filter (fn [c] (let [u (clojure.string/upper-case (str c)) l (clojure.string/lower-case (str c))] (and (not= u l) (= (str c) u)))) s)))"
 :tests [[["Hello World"] 2] [[""] 0] [["ABC"] 3] [["abc"] 0] [["a1B2"] 1]]}

   {:id "remove-vowels" :family :string :difficulty :medium :sig "(fn [s] ...)"
 :desc "that removes all vowels (aeiou, any case) from string s"
 :reference "(fn [s] (clojure.string/join \"\" (remove (fn [c] (contains? #{\"a\" \"e\" \"i\" \"o\" \"u\"} (clojure.string/lower-case (str c)))) s)))"
 :tests [[["hello"] "hll"] [[""] ""] [["aeiou"] ""] [["xyz"] "xyz"] [["AEIOU"] ""]]}

   {:id "repeat-string" :family :string :difficulty :easy :sig "(fn [s n] ...)"
 :desc "that returns string s repeated n times"
 :reference "(fn [s n] (clojure.string/join \"\" (repeat n s)))"
 :tests [[["ab" 3] "ababab"] [["x" 0] ""] [["" 5] ""] [["a" 1] "a"]]}

   {:id "is-substring?" :family :string :difficulty :easy :sig "(fn [s sub] ...)"
 :desc "that returns true if string sub occurs inside string s"
 :reference "(fn [s sub] (clojure.string/includes? s sub))"
 :tests [[["hello world" "world"] true] [["hello" ""] true] [["abc" "xyz"] false] [["" ""] true] [["" "a"] false]]}

   {:id "initials" :family :string :difficulty :medium :sig "(fn [s] ...)"
 :desc "that returns the uppercase first letter of each word in string s joined together"
 :reference "(fn [s] (if (clojure.string/blank? s) \"\" (clojure.string/join \"\" (map (fn [w] (clojure.string/upper-case (subs w 0 1))) (clojure.string/split (clojure.string/trim s) #\"\\s+\")))))"
 :tests [[["john ronald tolkien"] "JRT"] [[""] ""] [["alice"] "A"] [["mary jane"] "MJ"]]}

   {:id "swap-case" :family :string :difficulty :hard :sig "(fn [s] ...)"
 :desc "that swaps the case of every letter in string s (upper to lower and lower to upper)"
 :reference "(fn [s] (clojure.string/join \"\" (map (fn [c] (let [u (clojure.string/upper-case (str c)) l (clojure.string/lower-case (str c))] (cond (= u l) (str c) (= (str c) u) l :else u))) s)))"
 :tests [[["Hello"] "hELLO"] [[""] ""] [["abc"] "ABC"] [["ABC"] "abc"] [["aB3c"] "Ab3C"]]}

   {:id "censor-word" :family :string :difficulty :medium :sig "(fn [s word] ...)"
 :desc "that replaces every occurrence of word in string s with stars of the same length"
 :reference "(fn [s word] (clojure.string/replace s word (clojure.string/join \"\" (repeat (count word) \"*\"))))"
 :tests [[["i hate mondays" "hate"] "i **** mondays"] [["clean text" "bad"] "clean text"] [["" "x"] ""] [["aa bb aa" "aa"] "** bb **"]]}

   {:id "count-occurrences" :family :string :difficulty :easy :sig "(fn [s c] ...)"
 :desc "that counts how many times the single-character string c appears in string s"
 :reference "(fn [s c] (count (filter (fn [ch] (= (str ch) c)) s)))"
 :tests [[["banana" "a"] 3] [["" "a"] 0] [["xyz" "q"] 0] [["aaa" "a"] 3]]}

   {:id "longest-run" :family :string :difficulty :hard :sig "(fn [s] ...)"
 :desc "that returns the length of the longest run of consecutive identical characters in string s"
 :reference "(fn [s] (if (= s \"\") 0 (loop [cs (seq s) prev nil cur 0 best 0] (if (empty? cs) (max best cur) (let [c (first cs)] (if (= c prev) (recur (rest cs) c (inc cur) best) (recur (rest cs) c 1 (max best cur))))))))"
 :tests [[["aabbbc"] 3] [[""] 0] [["a"] 1] [["aaaa"] 4] [["abc"] 1]]}

   {:id "title-case" :family :string :difficulty :hard :sig "(fn [s] ...)"
 :desc "that title-cases string s (uppercase first letter of each word, lowercase the rest)"
 :reference "(fn [s] (if (clojure.string/blank? s) \"\" (clojure.string/join \" \" (map (fn [w] (str (clojure.string/upper-case (subs w 0 1)) (clojure.string/lower-case (subs w 1)))) (clojure.string/split (clojure.string/trim s) #\"\\s+\")))))"
 :tests [[["hello world"] "Hello World"] [[""] ""] [["a b"] "A B"] [["HELLO"] "Hello"]]}

   {:id "trim-ends" :family :string :difficulty :easy :sig "(fn [s] ...)"
 :desc "that removes leading and trailing whitespace from string s"
 :reference "(fn [s] (clojure.string/trim s))"
 :tests [[["  hi  "] "hi"] [[""] ""] [["   "] ""] [["no-space"] "no-space"]]}

   {:id "starts-with-vowel?" :family :string :difficulty :easy :sig "(fn [s] ...)"
 :desc "that returns true if string s starts with a vowel (any case)"
 :reference "(fn [s] (if (= s \"\") false (contains? #{\"a\" \"e\" \"i\" \"o\" \"u\"} (clojure.string/lower-case (subs s 0 1)))))"
 :tests [[["apple"] true] [[""] false] [["banana"] false] [["Orange"] true]]}

   {:id "vowel-count" :family :string :difficulty :easy :sig "(fn [s] ...)"
 :desc "that counts the vowels (aeiou, any case) in string s"
 :reference "(fn [s] (count (filter (fn [c] (contains? #{\"a\" \"e\" \"i\" \"o\" \"u\"} (str c))) (clojure.string/lower-case s))))"
 :tests [[["hello"] 2] [[""] 0] [["xyz"] 0] [["AEIOU"] 5]]}

   {:id "split-words" :family :string :difficulty :medium :sig "(fn [s] ...)"
 :desc "that splits string s into a vector of whitespace-separated words"
 :reference "(fn [s] (if (clojure.string/blank? s) [] (vec (clojure.string/split (clojure.string/trim s) #\"\\s+\"))))"
 :tests [[["a b c"] ["a" "b" "c"]] [[""] []] [["solo"] ["solo"]] [["  x  y  "] ["x" "y"]]]}

   {:id "double-chars" :family :string :difficulty :medium :sig "(fn [s] ...)"
 :desc "that doubles every character of string s"
 :reference "(fn [s] (clojure.string/join \"\" (mapcat (fn [c] (list c c)) s)))"
 :tests [[["abc"] "aabbcc"] [[""] ""] [["x"] "xx"] [["12"] "1122"]]}

   {:id "ends-with-q?" :family :string :difficulty :easy :sig "(fn [s suf] ...)"
 :desc "that returns true if string s ends with string suf"
 :reference "(fn [s suf] (clojure.string/ends-with? s suf))"
 :tests [[["hello" "lo"] true] [["hello" ""] true] [["abc" "z"] false] [["" ""] true]]}

   {:id "all-equal?" :family :logic :difficulty :easy :sig "(fn [xs] ...)"
 :desc "that returns true if all elements of vector xs are equal (and true for an empty vector)"
 :reference "(fn [xs] (if (empty? xs) true (apply = xs)))"
 :tests [[[[1 1 1]] true] [[[1 2 1]] false] [[[]] true] [[[7]] true] [[[3 3 3 3]] true] [[[0 0 1]] false]]}

   {:id "any-even?" :family :logic :difficulty :easy :sig "(fn [xs] ...)"
 :desc "that returns true if at least one element of integer vector xs is even"
 :reference "(fn [xs] (if (some even? xs) true false))"
 :tests [[[[1 3 4]] true] [[[1 3 5]] false] [[[]] false] [[[2]] true] [[[7 7 7 8]] true] [[[9]] false]]}

   {:id "all-odd?" :family :logic :difficulty :easy :sig "(fn [xs] ...)"
 :desc "that returns true if every element of integer vector xs is odd (and true for an empty vector)"
 :reference "(fn [xs] (every? odd? xs))"
 :tests [[[[1 3 5]] true] [[[1 2 3]] false] [[[]] true] [[[4]] false] [[[7]] true] [[[11 13 15 17]] true]]}

   {:id "strictly-increasing?" :family :logic :difficulty :medium :sig "(fn [xs] ...)"
 :desc "that returns true if integer vector xs is strictly increasing (each element greater than the previous; true for empty or single-element vectors)"
 :reference "(fn [xs] (if (< (count xs) 2) true (apply < xs)))"
 :tests [[[[1 2 3]] true] [[[1 1 2]] false] [[[]] true] [[[5]] true] [[[3 2 1]] false] [[[10 20 30 40]] true]]}

   {:id "non-decreasing?" :family :logic :difficulty :medium :sig "(fn [xs] ...)"
 :desc "that returns true if integer vector xs is sorted in non-decreasing order (each element greater than or equal to the previous)"
 :reference "(fn [xs] (if (< (count xs) 2) true (apply <= xs)))"
 :tests [[[[1 1 2 3]] true] [[[1 3 2]] false] [[[]] true] [[[4]] true] [[[5 5 5]] true] [[[9 8]] false]]}

   {:id "is-sorted-desc?" :family :logic :difficulty :medium :sig "(fn [xs] ...)"
 :desc "that returns true if integer vector xs is sorted in non-increasing (descending) order"
 :reference "(fn [xs] (if (< (count xs) 2) true (apply >= xs)))"
 :tests [[[[5 4 3 2]] true] [[[5 5 4]] true] [[[1 2 3]] false] [[[]] true] [[[7]] true] [[[3 1 2]] false]]}

   {:id "bool-xor" :family :logic :difficulty :easy :sig "(fn [a b] ...)"
 :desc "that returns the exclusive-or of two booleans a and b (true when exactly one is true)"
 :reference "(fn [a b] (not= (boolean a) (boolean b)))"
 :tests [[[true false] true] [[false true] true] [[true true] false] [[false false] false]]}

   {:id "bool-nand" :family :logic :difficulty :easy :sig "(fn [a b] ...)"
 :desc "that returns the logical NAND of two booleans a and b (false only when both are true)"
 :reference "(fn [a b] (not (and a b)))"
 :tests [[[true true] false] [[true false] true] [[false true] true] [[false false] true]]}

   {:id "bool-implies" :family :logic :difficulty :easy :sig "(fn [a b] ...)"
 :desc "that returns the logical implication a implies b (false only when a is true and b is false)"
 :reference "(fn [a b] (or (not a) (boolean b)))"
 :tests [[[true false] false] [[true true] true] [[false false] true] [[false true] true]]}

   {:id "majority-int" :family :logic :difficulty :hard :sig "(fn [xs] ...)"
 :desc "that returns the integer appearing in vector xs strictly more than half the time, or -1 if no such majority exists (-1 for an empty vector)"
 :reference "(fn [xs] (let [n (count xs) freqs (frequencies xs) winner (when (seq freqs) (apply max-key val freqs))] (if (and winner (> (* 2 (val winner)) n)) (key winner) -1)))"
 :tests [[[[1 1 1 2]] 1] [[[1 2 3]] -1] [[[5 5 5 5]] 5] [[[]] -1] [[[2 2 1 1]] -1] [[[7 7 7 9 9]] 7]]}

   {:id "triangle-type" :family :logic :difficulty :hard :sig "(fn [a b c] ...)"
 :desc "that classifies a triangle by side lengths a b c, returning the string \"equilateral\" if all sides equal, \"isosceles\" if exactly two are equal, otherwise \"scalene\""
 :reference "(fn [a b c] (cond (= a b c) \"equilateral\" (or (= a b) (= b c) (= a c)) \"isosceles\" :else \"scalene\"))"
 :tests [[[3 3 3] "equilateral"] [[3 3 5] "isosceles"] [[3 4 5] "scalene"] [[5 3 5] "isosceles"] [[2 2 2] "equilateral"] [[7 8 9] "scalene"]]}

   {:id "sign-str" :family :logic :difficulty :easy :sig "(fn [n] ...)"
 :desc "that returns the string \"positive\", \"negative\", or \"zero\" describing the sign of integer n"
 :reference "(fn [n] (cond (pos? n) \"positive\" (neg? n) \"negative\" :else \"zero\"))"
 :tests [[[5] "positive"] [[-3] "negative"] [[0] "zero"] [[100] "positive"] [[-1] "negative"]]}

   {:id "parity-match?" :family :logic :difficulty :easy :sig "(fn [a b] ...)"
 :desc "that returns true if integers a and b have the same parity (both even or both odd)"
 :reference "(fn [a b] (= (even? a) (even? b)))"
 :tests [[[2 4] true] [[1 3] true] [[2 3] false] [[0 7] false] [[6 8] true] [[9 4] false]]}

   {:id "count-in-range" :family :logic :difficulty :medium :sig "(fn [xs lo hi] ...)"
 :desc "that counts how many integers in vector xs fall inclusively between lo and hi"
 :reference "(fn [xs lo hi] (count (filter (fn [x] (and (>= x lo) (<= x hi))) xs)))"
 :tests [[[[1 5 10 15] 5 10] 2] [[[1 2 3] 0 10] 3] [[[] 0 5] 0] [[[1 2 3] 5 10] 0] [[[5 5 5] 5 5] 3] [[[-2 0 4 9] 0 5] 2]]}

   {:id "ordered-chain?" :family :logic :difficulty :medium :sig "(fn [a b c] ...)"
 :desc "that returns true if the three integers satisfy the chain a < b < c"
 :reference "(fn [a b c] (< a b c))"
 :tests [[[1 2 3] true] [[1 3 2] false] [[3 2 1] false] [[1 1 2] false] [[0 5 10] true] [[-1 0 1] true]]}

   {:id "minutes->seconds" :family :arithmetic :difficulty :easy :sig "(fn [m] ...)"
 :desc "converting whole minutes to seconds"
 :reference "(fn [m] (* m 60))"
 :tests [[[1] 60] [[2] 120] [[0] 0] [[10] 600] [[90] 5400]]}

   {:id "seconds->minutes-exact" :family :arithmetic :difficulty :easy :sig "(fn [s] ...)"
 :desc "converting seconds to whole minutes using integer division (truncating)"
 :reference "(fn [s] (quot s 60))"
 :tests [[[60] 1] [[120] 2] [[0] 0] [[150] 2] [[59] 0]]}

   {:id "hours->minutes" :family :arithmetic :difficulty :easy :sig "(fn [h] ...)"
 :desc "converting whole hours to minutes"
 :reference "(fn [h] (* h 60))"
 :tests [[[1] 60] [[3] 180] [[0] 0] [[24] 1440] [[2] 120]]}

   {:id "dollars->cents" :family :arithmetic :difficulty :easy :sig "(fn [d c] ...)"
 :desc "converting a dollars-and-cents amount (whole dollars and whole cents) into a total number of cents"
 :reference "(fn [d c] (+ (* d 100) c))"
 :tests [[[1 0] 100] [[1 50] 150] [[0 0] 0] [[5 25] 525] [[10 99] 1099]]}

   {:id "cents->dollars-and-cents" :family :arithmetic :difficulty :medium :sig "(fn [total] ...)"
 :desc "splitting a total in cents into a two-element vector [dollars cents]"
 :reference "(fn [total] [(quot total 100) (mod total 100)])"
 :tests [[[100] [1 0]] [[150] [1 50]] [[0] [0 0]] [[525] [5 25]] [[99] [0 99]]]}

   {:id "exact-percent" :family :arithmetic :difficulty :medium :sig "(fn [part whole] ...)"
 :desc "computing what percent part is of whole, given the result divides evenly into a whole-number percent (return 0 when whole is 0)"
 :reference "(fn [part whole] (if (zero? whole) 0 (quot (* part 100) whole)))"
 :tests [[[1 4] 25] [[1 2] 50] [[3 4] 75] [[0 10] 0] [[5 0] 0] [[10 10] 100]]}

   {:id "clock-add-12" :family :arithmetic :difficulty :medium :sig "(fn [hour delta] ...)"
 :desc "advancing a 12-hour clock hour (1..12) by delta hours, wrapping correctly so the result is in 1..12"
 :reference "(fn [hour delta] (inc (mod (+ (dec hour) delta) 12)))"
 :tests [[[12 1] 1] [[11 2] 1] [[1 0] 1] [[12 12] 12] [[3 24] 3] [[10 5] 3]]}

   {:id "clock-add-24" :family :arithmetic :difficulty :easy :sig "(fn [hour delta] ...)"
 :desc "advancing a 24-hour clock hour (0..23) by delta hours, wrapping with mod 24"
 :reference "(fn [hour delta] (mod (+ hour delta) 24))"
 :tests [[[23 1] 0] [[0 0] 0] [[20 5] 1] [[12 24] 12] [[10 14] 0]]}

   {:id "countdown-vec" :family :arithmetic :difficulty :medium :sig "(fn [n] ...)"
 :desc "building a countdown vector from n down to 0 inclusive (empty handling: for n=0 return [0])"
 :reference "(fn [n] (loop [i n acc []] (if (neg? i) acc (recur (dec i) (conj acc i)))))"
 :tests [[[3] [3 2 1 0]] [[0] [0]] [[1] [1 0]] [[5] [5 4 3 2 1 0]]]}

   {:id "rect-perimeter" :family :arithmetic :difficulty :easy :sig "(fn [w h] ...)"
 :desc "computing the perimeter of an integer-sided rectangle with width w and height h"
 :reference "(fn [w h] (* 2 (+ w h)))"
 :tests [[[2 3] 10] [[5 5] 20] [[0 0] 0] [[1 1] 4] [[10 4] 28]]}

   {:id "rect-area" :family :arithmetic :difficulty :easy :sig "(fn [w h] ...)"
 :desc "computing the area of an integer-sided rectangle with width w and height h"
 :reference "(fn [w h] (* w h))"
 :tests [[[2 3] 6] [[5 5] 25] [[0 7] 0] [[1 1] 1] [[10 4] 40]]}

   {:id "distance-squared" :family :arithmetic :difficulty :medium :sig "(fn [x1 y1 x2 y2] ...)"
 :desc "computing the squared euclidean distance between two integer points (avoiding sqrt)"
 :reference "(fn [x1 y1 x2 y2] (let [dx (- x2 x1) dy (- y2 y1)] (+ (* dx dx) (* dy dy))))"
 :tests [[[0 0 3 4] 25] [[0 0 0 0] 0] [[1 1 1 1] 0] [[1 2 4 6] 25] [[-1 -1 2 3] 25]]}

   {:id "steps-to-meters" :family :arithmetic :difficulty :medium :sig "(fn [steps] ...)"
 :desc "converting a step count to whole meters assuming each step is exactly 1 meter for every full pair of steps (i.e. integer-divide steps by 2 to get meters), truncating"
 :reference "(fn [steps] (quot steps 2))"
 :tests [[[2] 1] [[4] 2] [[0] 0] [[5] 2] [[1] 0]]}

   {:id "even-average" :family :arithmetic :difficulty :hard :sig "(fn [xs] ...)"
 :desc "computing the exact integer average of a non-empty vector of integers whose sum is evenly divisible by the count (return 0 for an empty vector)"
 :reference "(fn [xs] (if (empty? xs) 0 (quot (reduce + 0 xs) (count xs))))"
 :tests [[[[2 4 6]] 4] [[[10 20]] 15] [[[]] 0] [[[5]] 5] [[[1 2 3 4 5]] 3] [[[0 0 0]] 0]]}

   {:id "days->hours" :family :arithmetic :difficulty :easy :sig "(fn [d] ...)"
 :desc "converting whole days to hours"
 :reference "(fn [d] (* d 24))"
 :tests [[[1] 24] [[2] 48] [[0] 0] [[7] 168] [[10] 240]]}
])
