# GenMLX Studio: Interactive Probabilistic Programming on Apple Silicon

> A Clerk-based notebook environment for GenMLX, combining Babashka for the
> notebook UI with nbb for GPU-accelerated probabilistic computation via MLX.

**Status:** Plan only. No code yet.

---

## Vision

GenMLX Studio is an interactive notebook environment where you write
probabilistic models, run GPU-accelerated inference on Apple Silicon, and
visualize results — all in ClojureScript, with no build step.

The key insight: **Babashka + Clerk** gives us a production-quality notebook UI
with file watching, caching, and rich viewers. **nbb + MLX** gives us GPU
compute. An **nREPL bridge** connects them — one language family, two runtimes,
zero Python.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Editor (Emacs/VS Code/IntelliJ)                        │
│  Edit .clj notebooks, save triggers Clerk re-evaluation │
└──────────────────────┬──────────────────────────────────┘
                       │ file watch
┌──────────────────────▼──────────────────────────────────┐
│  Babashka + Clerk           (bb process)                │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Clerk                                          │    │
│  │  - File watching + caching                      │    │
│  │  - Evaluates .clj namespaces                    │    │
│  │  - Renders results as HTML in browser           │    │
│  │  - Custom viewers (Observable Plot, Vega-Lite)  │    │
│  │  - ::clerk/sync atoms for live updates          │    │
│  └──────────────┬──────────────────────────────────┘    │
│                 │                                        │
│  ┌──────────────▼──────────────────────────────────┐    │
│  │  genmlx.bridge (bb library, ~50 lines)          │    │
│  │  - Connects to nbb nREPL server                 │    │
│  │  - Sends ClojureScript forms for evaluation     │    │
│  │  - Returns results as Clojure data              │    │
│  │  - Manages nbb process lifecycle                │    │
│  └──────────────┬──────────────────────────────────┘    │
└─────────────────┼───────────────────────────────────────┘
                  │ nREPL protocol (TCP, bencode)
┌─────────────────▼───────────────────────────────────────┐
│  nbb + MLX                  (nbb process)               │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  nbb nrepl-server :port 1337                    │    │
│  │  - Evaluates ClojureScript via SCI              │    │
│  │  - Full GenMLX available (all 29 source files)  │    │
│  │  - @frost-beta/mlx for GPU compute              │    │
│  │  - State persists across evaluations            │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  Apple Silicon GPU (M1/M2/M3/M4)                        │
│  - MLX unified memory (zero CPU-GPU transfer)           │
│  - Metal compute shaders for inference                  │
│  - Lazy graph + explicit eval                           │
└─────────────────────────────────────────────────────────┘
                  │ results as EDN strings
┌─────────────────▼───────────────────────────────────────┐
│  Browser (localhost:7777)                                │
│                                                         │
│  Clerk UI                                               │
│  ├── Rendered notebook (code + results)                 │
│  ├── Observable Plot (loaded via CDN, renders SVG)      │
│  ├── Vega-Lite (built-in Clerk viewer)                  │
│  ├── Custom viewers (SCI in browser)                    │
│  └── Live updates via ::clerk/sync atoms                │
└─────────────────────────────────────────────────────────┘
```

---

## The nREPL Bridge

### Why nREPL?

nbb has a built-in nREPL server (`nbb nrepl-server`). Babashka has an nREPL
client library ([babashka/nrepl-client](https://github.com/babashka/nrepl-client)).
The nREPL protocol is language-agnostic: bb sends code as strings, nbb evaluates
them as ClojureScript, and returns `pr-str`'d results. This has been
[explicitly confirmed working](https://github.com/babashka/nbb/issues/395) by
Michiel Borkent (author of both bb and nbb).

### Bridge implementation (~50 lines)

```clojure
;; src/genmlx/bridge.clj — runs in Babashka
(ns genmlx.bridge
  (:require [babashka.nrepl-client :as nrepl]
            [clojure.edn :as edn]))

(def ^:private conn-opts (atom {:host "localhost" :port 1337}))

(defn configure! [{:keys [host port]}]
  (swap! conn-opts merge (cond-> {}
                           host (assoc :host host)
                           port (assoc :port port))))

(defn eval-mlx
  "Send a ClojureScript form to the nbb nREPL server for evaluation.
   Returns the result as Clojure data (parsed from EDN)."
  [form]
  (let [{:keys [host port]} @conn-opts
        result (nrepl/eval-expr {:host host :port port
                                 :expr (pr-str form)})
        vals (:vals result)]
    (when (seq vals)
      (edn/read-string (last vals)))))

(defn eval-mlx!
  "Like eval-mlx but for side-effecting forms (returns nil)."
  [form]
  (nrepl/eval-expr {:host (:host @conn-opts)
                    :port (:port @conn-opts)
                    :expr (pr-str form)})
  nil)

(defn require-genmlx!
  "Load GenMLX namespaces in the nbb session."
  []
  (eval-mlx! '(do (require '[genmlx.mlx :as mx])
                  (require '[genmlx.mlx.random :as rng])
                  (require '[genmlx.dist :as dist])
                  (require '[genmlx.dynamic :as dyn])
                  (require '[genmlx.protocols :as p])
                  (require '[genmlx.choicemap :as cm])
                  (require '[genmlx.selection :as sel])
                  (require '[genmlx.inference.mcmc :as mcmc])
                  (require '[genmlx.inference.importance :as is])
                  (require '[genmlx.inference.smc :as smc])
                  (require '[genmlx.inference.vi :as vi])
                  (require-macros '[genmlx.gen :refer [gen]]))))
```

### Data boundary

MLX arrays cannot cross the nREPL boundary. The convention is:

- **nbb side:** Run inference, extract results with `mx/item` or `mapv mx/item`
- **Return:** Plain Clojure data (numbers, vectors, maps) as EDN
- **bb side:** Receive Clojure data, pass to Clerk viewers

```clojure
;; This form runs entirely in nbb:
(eval-mlx
  '(let [traces (mcmc/nuts {:samples 500 :warmup 200} model [xs] obs)]
     ;; Extract to plain data before returning
     {:slopes     (mapv #(mx/item (cm/get-choice (:choices %) [:slope])) traces)
      :intercepts (mapv #(mx/item (cm/get-choice (:choices %) [:intercept])) traces)
      :scores     (mapv #(mx/item (:score %)) traces)}))
;; => {:slopes [1.98 2.01 ...] :intercepts [0.95 1.02 ...] :scores [-12.3 ...]}
```

---

## Clerk Viewers for Probabilistic Programming

### Built-in: Vega-Lite (`clerk/vl`)

Clerk has Vega-Lite built in. No setup needed. Covers most standard plots:

```clojure
;; Trace plot
(clerk/vl
  {:data {:values (map-indexed (fn [i s] {:iteration i :slope s}) (:slopes posterior))}
   :mark "line"
   :encoding {:x {:field "iteration" :type "quantitative"}
              :y {:field "slope" :type "quantitative"}}})

;; Posterior scatter
(clerk/vl
  {:data {:values (map (fn [s i] {:slope s :intercept i})
                       (:slopes posterior) (:intercepts posterior))}
   :mark {:type "point" :opacity 0.3}
   :encoding {:x {:field "slope" :type "quantitative"}
              :y {:field "intercept" :type "quantitative"}}})

;; Histogram
(clerk/vl
  {:data {:values (map (fn [s] {:slope s}) (:slopes posterior))}
   :mark "bar"
   :encoding {:x {:field "slope" :bin {:maxbins 40} :type "quantitative"}
              :y {:aggregate "count" :type "quantitative"}}})
```

### Custom: Observable Plot via `with-d3-require`

Observable Plot (same library GenStudio uses) is loadable via CDN in Clerk's
browser runtime:

```clojure
(def plot-viewer
  {:transform-fn clerk/mark-presented
   :render-fn '(fn [data]
                 [nextjournal.clerk.render/with-d3-require
                  {:package ["@observablehq/plot@0.6"]}
                  (fn [Plot]
                    [:div {:ref (fn [el]
                                  (when el
                                    (let [chart (.plot Plot (clj->js data))]
                                      (set! (.-innerHTML el) "")
                                      (.appendChild el chart))))}])])})

;; Usage:
(clerk/with-viewer plot-viewer
  {:marks [{:type "dot" :data posterior-samples :x "slope" :y "intercept"}
           {:type "density" :data posterior-samples :x "slope" :y "intercept"}]
   :grid true})
```

### Custom: Inference Dashboard

A compound viewer showing MCMC diagnostics:

```clojure
(def mcmc-dashboard-viewer
  {:transform-fn clerk/mark-presented
   :render-fn '(fn [{:keys [traces r-hat ess acceptance-rate]}]
                 [:div.grid.grid-cols-2.gap-4
                  ;; Trace plot
                  [:div [render-vega-lite {:data {:values traces}
                                           :mark "line"
                                           :encoding {:x {:field "i" :type "quantitative"}
                                                      :y {:field "v" :type "quantitative"}
                                                      :color {:field "param" :type "nominal"}}}]]
                  ;; R-hat bar chart
                  [:div [render-vega-lite {:data {:values r-hat}
                                           :mark "bar"
                                           :encoding {:x {:field "param" :type "nominal"}
                                                      :y {:field "rhat" :type "quantitative"}}}]]
                  ;; ESS
                  [:div [:h3 "ESS"] [:pre (pr-str ess)]]
                  ;; Acceptance rate
                  [:div [:h3 "Acceptance Rate"] [:pre (str acceptance-rate)]]])})
```

---

## Live Inference Streaming

Clerk's `::clerk/sync` atom mechanism enables real-time visualization of
running inference. When bb updates a synced atom, Clerk pushes the diff to the
browser at up to 60fps via editscript.

### Pattern: Live MCMC

```clojure
;; notebooks/live_mcmc.clj

;; Synced atom — Clerk watches this for changes
^{::clerk/sync true}
(defonce mcmc-progress (atom {:iteration 0 :samples [] :complete? false}))

;; Custom viewer that reactively renders the atom
(clerk/with-viewer
  {:render-fn '(fn [_]
                 (let [state @nextjournal.clerk.render/!mcmc-progress]
                   [:div
                    [:h3 (str "Iteration " (:iteration state)
                              (when (:complete? state) " (done)"))]
                    [nextjournal.clerk.render/render-vega-lite
                     {:data {:values (:samples state)}
                      :mark "line"
                      :encoding {:x {:field "i" :type "quantitative"}
                                 :y {:field "v" :type "quantitative"}}}]]))}
  @mcmc-progress)

;; Run inference in a background thread, updating the atom
(future
  (doseq [batch (partition-all 50 (range 1000))]
    (let [new-samples
          (eval-mlx
            `(let [traces (mcmc/mh {:samples 50} model [xs] obs)]
               (mapv (fn [t i#]
                       {:i (+ ~(first batch) i#)
                        :v (mx/item (cm/get-choice (:choices t) [:slope]))})
                     traces (range))))]
      (swap! mcmc-progress
             #(-> % (update :samples into new-samples)
                    (assoc :iteration (last batch)))))
    (Thread/sleep 100))  ;; let Clerk push updates
  (swap! mcmc-progress assoc :complete? true))
```

### Pattern: Live VI Convergence

```clojure
^{::clerk/sync true}
(defonce vi-progress (atom {:epoch 0 :elbos []}))

(clerk/with-viewer
  {:render-fn '(fn [_]
                 (let [{:keys [elbos]} @nextjournal.clerk.render/!vi-progress]
                   [nextjournal.clerk.render/render-vega-lite
                    {:data {:values (map-indexed (fn [i e] {:epoch i :elbo e}) elbos)}
                     :mark "line"
                     :encoding {:x {:field "epoch" :type "quantitative"}
                                :y {:field "elbo" :type "quantitative"}}}]))}
  @vi-progress)
```

---

## Visualization Catalog

What specific visualizations a probabilistic programming notebook needs:

### Inference Diagnostics

| Visualization | Purpose | Viewer |
|---|---|---|
| Trace plot | Show MCMC chain mixing | Vega-Lite line |
| Autocorrelation plot | Detect chain dependence | Vega-Lite bar |
| R-hat display | Convergence diagnostic | Vega-Lite bar + rule at 1.01 |
| ESS table | Effective sample size per parameter | Clerk table |
| Acceptance rate | MCMC tuning diagnostic | Vega-Lite text |
| ELBO curve | VI convergence | Vega-Lite line (live via sync atom) |
| Pair plot | Joint posterior | Observable Plot density2d |

### Model Visualization

| Visualization | Purpose | Viewer |
|---|---|---|
| Prior/posterior overlay | Show what inference learned | Vega-Lite layer (area + area) |
| Posterior predictive | Model fit with uncertainty bands | Observable Plot area + line |
| Choicemap tree | Inspect trace structure | Custom Clerk viewer (Hiccup tree) |
| Score decomposition | Per-address log-probability | Vega-Lite bar |
| Model DAG | Dependency structure | D3 force-directed (via d3-require) |

### Particle Methods

| Visualization | Purpose | Viewer |
|---|---|---|
| Particle cloud | SMC particle positions + weights | Observable Plot dot (size = weight) |
| Weight histogram | Particle weight distribution | Vega-Lite bar |
| Log-ML curve | Marginal likelihood estimate over steps | Vega-Lite line |
| Resampling events | When ESS drops below threshold | Vega-Lite rule marks |

---

## Project Structure

```
genmlx-studio/
├── bb.edn                          Babashka deps (Clerk, nrepl-client)
├── deps.edn                        Clojure deps (fallback for JVM Clerk)
├── package.json                    npm deps (for nbb: @frost-beta/mlx)
│
├── src/
│   └── genmlx/
│       ├── bridge.clj              nREPL bridge (bb → nbb, ~50 lines)
│       ├── studio.clj              Studio helpers (start/stop, convenience fns)
│       └── viewers.clj             Custom Clerk viewers for PP
│
├── notebooks/
│   ├── getting_started.clj         Intro notebook (model + inference + plot)
│   ├── bayesian_regression.clj     Full regression example
│   ├── hmm.clj                     Hidden Markov Model with SMC
│   ├── mixture_model.clj           Gaussian mixture with Gibbs
│   ├── live_mcmc.clj               Live streaming MCMC dashboard
│   ├── live_vi.clj                 Live streaming VI convergence
│   ├── model_comparison.clj        Compare inference algorithms on same model
│   └── diagnostics.clj             R-hat, ESS, trace plots
│
└── dev/
    └── user.clj                    Dev entry point (starts Clerk + nbb)
```

### `bb.edn`

```clojure
{:deps {io.github.nextjournal/clerk {:mvn/version "0.18.1150"}
        babashka/nrepl-client
        {:git/url "https://github.com/babashka/nrepl-client"
         :git/sha "519f09cbfcfebf5633368f7f34f4ad993b453f49"}}

 :tasks
 {nbb    {:doc "Start nbb nREPL server for MLX compute"
          :task (shell "nbb nrepl-server :port 1337")}

  dev    {:doc "Start Clerk notebook server"
          :task (exec 'nextjournal.clerk/serve!)
          :exec-args {:port 7777 :browse true
                      :watch-paths ["notebooks"]}}

  build  {:doc "Build static site"
          :task (exec 'nextjournal.clerk/build!)
          :exec-args {:paths ["notebooks/getting_started.clj"
                              "notebooks/bayesian_regression.clj"]
                      :out-path "public/build"}}}}
```

### Developer Workflow

```bash
# Terminal 1: Start GPU compute server
bb nbb
# => nREPL server started on port 1337 on host 127.0.0.1

# Terminal 2: Start Clerk notebook UI
bb dev
# => Clerk webserver started on http://localhost:7777

# Edit notebooks/*.clj in your editor
# Clerk re-evaluates on save, delegates MLX calls to nbb via nREPL
# Browser updates automatically
```

Or as a single command via `dev/user.clj`:

```clojure
;; dev/user.clj
(ns user
  (:require [babashka.process :as proc]
            [nextjournal.clerk :as clerk]
            [genmlx.bridge :as mlx]))

(defonce nbb-proc
  (delay
    (println "Starting nbb nREPL server...")
    (proc/process ["nbb" "nrepl-server" ":port" "1337"]
      {:out :inherit :err :inherit})
    (Thread/sleep 2000)  ;; wait for nbb startup
    (mlx/require-genmlx!)
    (println "GenMLX loaded on GPU.")))

(defn go []
  @nbb-proc
  (clerk/serve! {:port 7777 :browse true :watch-paths ["notebooks"]}))
```

---

## Comparison with GenStudio

| Dimension | GenStudio | GenMLX Studio |
|---|---|---|
| Language | Python | ClojureScript (bb + nbb) |
| Notebook | Jupyter | Clerk |
| GPU compute | JAX/NumPy | MLX via @frost-beta/mlx |
| 2D charts | Observable Plot (browser) | Observable Plot + Vega-Lite (browser) |
| 3D rendering | WebGPU (browser) | WebGPU (browser, future) |
| Build step | esbuild for JS bundle | None (Clerk + CDN) |
| Live updates | anywidget traits | ::clerk/sync atoms |
| State management | React + mobx | Clerk SCI + Reagent |
| Distribution | pip install | bb (no install beyond deps) |
| Gen integration | None (visualization only) | Full GenMLX (inference + visualization) |

### What GenMLX Studio adds over GenStudio

1. **Integrated inference.** GenStudio visualizes data; GenMLX Studio runs
   inference AND visualizes results in the same environment.
2. **Live streaming.** Watch MCMC chains converge in real time, see ELBO curves
   update during VI. GenStudio's Jupyter model is request-response.
3. **No Python.** One language family (Clojure/Script) for everything.
4. **No build step.** Clerk + CDN-loaded viewers. No esbuild, no webpack.
5. **Caching.** Clerk caches expensive computations. Re-running a notebook skips
   unchanged inference calls.

### What GenMLX Studio lacks vs GenStudio

1. **3D rendering.** GenStudio's WebGPU scene renderer (point clouds, ellipsoids)
   would need to be ported or loaded as a separate JS bundle. Phase 3 work.
2. **GenStudio's mark library.** 100+ Observable Plot mark wrappers with Python
   composition via `+`. We'd build a smaller, PP-focused set of Clerk viewers.
3. **Jupyter ecosystem.** Jupyter has millions of users. Clerk has thousands.

---

## Implementation Phases

### Phase 0: Proof of Concept (~1 day)

Validate the bb → nbb nREPL bridge works end-to-end:

- [ ] Start nbb nREPL server, connect from bb, send a GenMLX form, get result
- [ ] Render a simple Vega-Lite plot in Clerk from nbb-computed data
- [ ] Verify Clerk caching works (second evaluation skips nbb call)

**Success criterion:** A single Clerk notebook that defines a model in nbb,
runs MH inference on GPU, and displays a posterior histogram.

### Phase 1: Core Bridge + Basic Viewers (~3 days)

- [ ] `genmlx.bridge` — nREPL connection, eval-mlx, require-genmlx!, error handling
- [ ] `genmlx.studio` — start/stop helpers, single-command launcher
- [ ] `genmlx.viewers` — trace-plot, histogram, scatter, diagnostics-table
- [ ] `getting_started.clj` notebook — minimal working example
- [ ] `bb.edn` with tasks for nbb + Clerk

### Phase 2: Live Streaming + Diagnostics (~3 days)

- [ ] `::clerk/sync` integration for live MCMC progress
- [ ] `::clerk/sync` integration for live VI convergence
- [ ] R-hat, ESS, acceptance rate viewers
- [ ] `live_mcmc.clj` notebook
- [ ] `live_vi.clj` notebook

### Phase 3: Observable Plot Integration (~2 days)

- [ ] Observable Plot viewer via `with-d3-require`
- [ ] Density plots (KDE), pair plots, posterior predictive
- [ ] Particle cloud viewer for SMC
- [ ] `hmm.clj` notebook (SMC + particle visualization)

### Phase 4: Rich Notebooks (~3 days)

- [ ] `bayesian_regression.clj` — full tutorial (model, data, inference, diagnostics)
- [ ] `mixture_model.clj` — discrete + continuous, Gibbs sampling
- [ ] `model_comparison.clj` — same model, MH vs HMC vs NUTS vs VI
- [ ] `diagnostics.clj` — comprehensive diagnostic dashboard

### Phase 5: 3D and Polish (~5 days, optional)

- [ ] Port GenStudio's WebGPU scene renderer as a static JS bundle
- [ ] 3D posterior visualization (point clouds in parameter space)
- [ ] Choicemap tree viewer (interactive, expandable)
- [ ] Static site build for sharing notebooks
- [ ] Documentation

---

## Dependencies

### Babashka side (bb.edn)

```clojure
io.github.nextjournal/clerk {:mvn/version "0.18.1150"}
babashka/nrepl-client {:git/url "..." :git/sha "..."}
```

**Requires:** Babashka 1.12.206+ (for Clerk compatibility)

### nbb side (package.json)

```json
{
  "@frost-beta/mlx": "^0.4.0"
}
```

Already in GenMLX's existing `package.json`. No new npm dependencies needed for
the nbb side.

### Browser side (loaded via CDN by Clerk)

- Vega-Lite: `vega-embed@6.11.1` (built into Clerk)
- Observable Plot: `@observablehq/plot@0.6` (loaded via `with-d3-require`)
- D3: `d3@7` (loaded via `with-d3-require`, if needed)

No build step. No `node_modules` on the client side.

---

## Known Limitations and Mitigations

### nREPL serialization boundary

MLX arrays cannot cross the nREPL wire. All GPU results must be extracted to
plain Clojure data (`mx/item`, `mapv mx/item`) before returning.

**Mitigation:** Provide helper macros that automatically extract:

```clojure
(defn extract-traces
  "Extract trace data to plain Clojure maps for visualization."
  [traces params]
  (eval-mlx
    `(mapv (fn [t#]
             (into {:score (mx/item (:score t#))}
                   (map (fn [p#] [p# (mx/item (cm/get-choice (:choices t#) [p#]))]))
                   ~params))
           ~traces)))
```

### nbb startup time

nbb takes ~1-2 seconds to start. MLX module loading adds ~1 second.

**Mitigation:** Start nbb once, keep it running. The nREPL server persists state
across evaluations. `dev/user.clj` starts it automatically.

### Clerk + Babashka limitations

Clerk on Babashka is newer than Clerk on JVM. Some features may have rough edges:
- Separate cache files (`bb_` prefix)
- No editscript (affects `::clerk/sync` diff efficiency)
- Markdown parsing via QuickJS

**Mitigation:** Test with bb first. Fall back to JVM Clerk (`clj`) if needed —
the bridge library works with both since it's plain Clojure.

### Console output from nbb

`console.log` in nbb doesn't propagate to the nREPL `:out` stream. Only
`println` output does ([nbb #305](https://github.com/babashka/nbb/issues/305)).

**Mitigation:** Use `println` in nbb code, not `console.log`. Or check
`:responses` for `:out` keys in the bridge.

---

## Effort Estimate

| Phase | Days | Cumulative |
|---|---|---|
| 0. Proof of concept | 1 | 1 |
| 1. Core bridge + basic viewers | 3 | 4 |
| 2. Live streaming + diagnostics | 3 | 7 |
| 3. Observable Plot integration | 2 | 9 |
| 4. Rich notebooks | 3 | 12 |
| 5. 3D + polish (optional) | 5 | 17 |

**Minimum viable product: Phase 0-1 (4 days).** A working notebook that runs
GenMLX inference on GPU and displays results with Vega-Lite.

**Full system: Phases 0-4 (12 days).** Live streaming, rich diagnostics,
multiple example notebooks, Observable Plot integration.

---

## Open Questions

1. **Clerk on bb: is `::clerk/sync` fully supported?** The Clerk + Babashka PR
   (#232) merged in June 2025, but `::clerk/sync` relies on editscript which
   has alternate handling in bb. Needs testing in Phase 0.

2. **nREPL session management.** Should the bridge use a single persistent
   session (state accumulates) or fresh sessions per eval (clean but slower)?
   Persistent sessions are better for the REPL-like workflow (define model once,
   run inference many times).

3. **Error handling.** When nbb throws an exception, the nREPL response includes
   `:err` and `:ex` fields. The bridge should surface these clearly in Clerk
   (red error box, stack trace).

4. **Large result sets.** Returning 10,000 MCMC samples as EDN over nREPL may be
   slow. Consider: compute summary statistics in nbb, return only what's needed
   for visualization. Or stream batches.

5. **WebGPU 3D.** GenStudio's scene renderer is ~800 lines of TypeScript + WGSL
   shaders. Porting to a Clerk viewer requires bundling it as a JS file and
   loading via `with-dynamic-import`. Worth it only if 3D posterior visualization
   is a priority.
