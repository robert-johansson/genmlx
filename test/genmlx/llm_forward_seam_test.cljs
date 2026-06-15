;; @tier fast
(ns genmlx.llm-forward-seam-test
  "A.4 / f6ov decoupling drift-guard (STATIC source analysis — GPU-free, no model).

   Proves the GenMLX-owned forward path no longer depends on the four fragile
   upstream model-struct methods — .forward / .forwardWithCache / .initCaches /
   .resetCaches — the seam that every mlx-node rebase used to break (see
   genmlx-f6ov / genmlx-dbce).

   This is the durable, repeatable form of f6ov's final done-means box, 'confirm
   a simulated upstream resync touches only genmlx.rs/transforms.rs': if a resync
   DELETED those four methods, the owned default would be unaffected, because
     (1) the owned forward modules (forward/qwen3/qwen35) never call them, and
     (2) backend.cljs reaches them ONLY inside functions gated by
         cljs-forward-model? — the explicit {:cljs-forward? false} fallback.
   If a future edit breaks either invariant, this test fails — the rebase-tax
   mitigation becomes a CI invariant rather than a one-time snapshot.

   See docs/forward-seam.md for the full owned-path mlx-node surface (all in the
   never-rebased genmlx.rs / MLX fast:: set)."
  (:require [cljs.test :refer [deftest is testing]]
            [clojure.string :as str]
            ["fs" :as fs]))

(def ^:private borrowed
  "Upstream per-model-struct methods the owned forward must not depend on."
  ["forward" "forwardWithCache" "initCaches" "resetCaches"])

(defn- slurp* [p] (.readFileSync fs p "utf8"))

(defn- calls?
  "True if `src` contains a CLJS JS-interop method call `(.method ...` — so the
   namespace name `genmlx.llm.forward`, an `:as` require, a defn named `forward`,
   or prose mentioning `.forward` do NOT count; only an actual `(.method obj)`."
  [src method]
  (boolean (re-find (re-pattern (str "\\(\\." method "\\b")) src)))

(deftest owned-forward-modules-have-no-borrowed-calls
  (testing "the owned forward computation (forward/qwen3/qwen35) calls none of the
            upstream model-struct methods"
    (doseq [f ["src/genmlx/llm/forward.cljs"
               "src/genmlx/llm/qwen3_forward.cljs"
               "src/genmlx/llm/qwen35_forward.cljs"]]
      (let [src (slurp* f)]
        (doseq [m borrowed]
          (is (not (calls? src m))
              (str f " must not call ." m " (a borrowed upstream model-struct method)")))))))

(defn- form-name [form] (second (re-find #"\(defn-?\s+([^\s\[\]]+)" form)))

(deftest backend-gates-every-borrowed-call-behind-the-fallback
  (testing "every backend.cljs form that calls a borrowed method is gated by
            cljs-forward-model? (or is the private forward-with-cache leaf, which
            is itself invoked only from gated dispatchers) — so the owned default
            never reaches a borrowed call"
    (let [src   (slurp* "src/genmlx/llm/backend.cljs")
          ;; split into top-level forms (a newline immediately followed by '(')
          forms (str/split src #"\n(?=\()")
          borrowed-forms (filter (fn [form] (some #(calls? form %) borrowed)) forms)]
      (is (seq borrowed-forms) "sanity: backend.cljs still carries the borrowed-forward fallback")
      ;; (1) each form with a borrowed call is gated, or is the one allowed leaf
      (doseq [form borrowed-forms]
        (is (or (str/includes? form "cljs-forward-model?")
                (= "forward-with-cache" (form-name form)))
            (str "borrowed-method call in an ungated backend form: " (pr-str (form-name form)))))
      ;; (2) ...and that leaf is reached only from gated forms
      (doseq [form (filter #(str/includes? % "(forward-with-cache ") forms)]
        (is (str/includes? form "cljs-forward-model?")
            (str "forward-with-cache invoked from an ungated form: " (pr-str (form-name form))))))))

(cljs.test/run-tests)
