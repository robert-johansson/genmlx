# GenMLX Studio: The Complete Vision

> Everything is a value. The model is a value. The trace is a value. The
> posterior is a value. The visualization is a value. The session is a value.
> The entire interactive probabilistic programming experience is a pure
> function from events to state.

**Status:** Vision document. Synthesizes and extends plan-studio.md (Clerk/browser)
and plan-studio-wild.md (terminal-native TUI).

---

## The Core Insight

GenMLX already embodies the deepest idea in functional programming: **everything
is data**. Choice maps ARE Clojure maps. Distributions are records with open
multimethods. Traces are immutable values. The handler is a pure state transition
`(fn [state addr dist] -> [value state'])` wrapped in a single `volatile!`.

The studio extends this principle to the interactive environment itself. The
session is not a sequence of side effects — it is a **left fold over an event
log**. The visualization is not a drawing command — it is a **declarative spec**.
The rendering target is not hardcoded — it is a **multimethod dispatch**.

This produces something that doesn't exist anywhere in probabilistic programming:
a purely functional interactive environment where every artifact is a value, every
operation is a function, and every view is derived from the same source of truth.

---

## Architecture: One Atom, Many Views

```
                        ┌──────────────┐
                        │  Event Log   │
                        │  (vector)    │
                        └──────┬───────┘
                               │ reduce session-step
                               ▼
                        ┌──────────────┐
                        │  Universe    │
                        │  (r/atom)    │
                        │              │
                        │  model       │
                        │  data        │
                        │  posterior   │
                        │  viz-specs   │
                        │  history     │
                        └──────┬───────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
               ▼               ▼               ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ Terminal    │  │ Browser    │  │ LaTeX      │
        │ (Ink)      │  │ (Reagent)  │  │ (pgfplots) │
        │            │  │            │  │            │
        │ same       │  │ same       │  │ same       │
        │ render     │  │ render     │  │ render     │
        │ specs      │  │ specs      │  │ specs      │
        └────────────┘  └────────────┘  └────────────┘
```

One atom. One reducer. Many renderers. This is Clojure's vision of state
management applied to probabilistic programming.

### The Atom

```clojure
(defonce universe
  (r/atom
    {:models   {}          ;; name → GenerativeFunction (immutable values)
     :data     {}          ;; name → ChoiceMap (observations, immutable)
     :runs     []          ;; [{:model :algorithm :config :traces :diagnostics}]
     :specs    []          ;; [RenderSpec ...] — declarative visualization queue
     :events   []          ;; complete event log (the session IS this vector)
     :focus    nil}))      ;; which model/run is "active"
```

### The Reducer

```clojure
(defn session-step
  "Pure function: state × event → state'.
   The entire session is (reduce session-step initial-state events)."
  [state event]
  (-> (case (:type event)
        :define-model
        (assoc-in state [:models (:name event)] (:model event))

        :set-data
        (assoc-in state [:data (:name event)] (:observations event))

        :inference-complete
        (update state :runs conj
                {:model     (:model event)
                 :algorithm (:algorithm event)
                 :config    (:config event)
                 :traces    (:traces event)
                 :diagnostics (compute-diagnostics (:traces event))})

        :render
        (update state :specs conj (:spec event))

        :focus
        (assoc state :focus (:target event))

        ;; ... all other event types
        state)
      (update :events conj event)))
```

This gives you:
- **Time-travel.** Replay any prefix of events to reconstruct any prior state.
- **Undo.** `(reduce session-step init (butlast events))`.
- **Branching.** Fork the event log at any point. Explore two models in parallel.
- **Serialization.** `(pr-str (:events @universe))` saves the entire session.
- **Reproducibility.** Replay the event log on another machine. Same results.

---

## Everything is Data (Really, Everything)

### Models Are Values

```clojure
;; A model is an immutable DynamicGF record
(def regression
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 5))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1)))
      slope)))

;; Store it by name — it's just a value in a map
(swap! universe assoc-in [:models :regression] regression)
```

### Visualizations Are Specs

No side effects. No drawing commands. A visualization is a Clojure map:

```clojure
{:type   :posterior-histogram
 :data   (extract-samples traces [:slope :intercept])
 :params {:bins 40
          :overlay-prior? true
          :credible-interval 0.95}}
```

The spec says **what** to show. The renderer decides **how**. The same spec
renders to terminal (Observable Plot → SVG → PNG → inline), browser
(Observable Plot in DOM), LaTeX (pgfplots), or ASCII (asciichart).

### Inference Runs Are Values

```clojure
{:model     regression
 :algorithm :nuts
 :config    {:samples 1000 :warmup 500 :target-accept 0.8}
 :traces    [trace₁ trace₂ ... trace₁₀₀₀]  ;; each an immutable Trace record
 :diagnostics {:r-hat {:slope 1.001 :intercept 1.002}
               :ess   {:slope 892 :intercept 856}
               :acceptance-rate 0.81}}
```

Compare runs by comparing values. No mutable state to confuse which run
produced which samples.

### The Session Is an Event Log

```clojure
[{:type :define-model :name :regression :model regression :t 0}
 {:type :set-data :name :xy :observations obs :t 1}
 {:type :inference-started :model :regression :algorithm :nuts :t 2}
 {:type :inference-progress :iteration 500 :r-hat 1.01 :ess 340 :t 3}
 {:type :inference-complete :model :regression :algorithm :nuts
  :traces [...] :t 4}
 {:type :render :spec {:type :posterior-histogram :data ...} :t 5}
 {:type :user-input :text "Try wider priors" :t 6}
 {:type :define-model :name :regression-v2 :model regression-v2 :t 7}
 ...]
```

Every event is a plain Clojure map. The session is a vector of maps.
`filter`, `group-by`, `frequencies` — all of Clojure's sequence library
works on your session history.

---

## The Rendering Protocol

A single multimethod dispatches on `[target, spec-type]`:

```clojure
(defmulti render
  "Render a visualization spec to a target surface."
  (fn [target spec] [target (:type spec)]))

;; Terminal: Observable Plot → SVG → PNG → inline image
(defmethod render [:terminal :posterior-histogram]
  [_ {:keys [data params]}]
  (let [svg (plot/histogram data params)
        png (sharp/svg->png svg)]
    (terminal-image/display png)))

;; Browser: Observable Plot in DOM (interactive, zoomable)
(defmethod render [:browser :posterior-histogram]
  [_ {:keys [data params]}]
  [:div {:ref (fn [el]
                (when el
                  (let [chart (Plot/plot (clj->js (histogram-spec data params)))]
                    (.replaceChildren el chart))))}])

;; LaTeX: pgfplots for papers
(defmethod render [:latex :posterior-histogram]
  [_ {:keys [data params]}]
  (str "\\begin{axis}[ybar]\n"
       "\\addplot+ [hist={bins=" (:bins params) "}] table {data.csv};\n"
       "\\end{axis}"))

;; ASCII: asciichart for SSH sessions, piping, logs
(defmethod render [:ascii :posterior-histogram]
  [_ {:keys [data params]}]
  (asciichart/plot (clj->js (vec (:values data))) #js {:height 15}))
```

Four renderers. Same spec. Write your visualization once; view it anywhere.

Every GenMLX type becomes renderable:

```clojure
(defprotocol IRenderable
  (render-spec [this]
    "Return a declarative visualization spec for this value."))

;; A trace knows how to visualize itself
(extend-type genmlx.trace/Trace
  IRenderable
  (render-spec [t]
    {:type :choicemap-tree
     :choices (:choices t)
     :score (:score t)
     :highlight-high-score? true}))

;; A VectorizedTrace renders as a particle cloud
(extend-type genmlx.vectorized/VectorizedTrace
  IRenderable
  (render-spec [vt]
    {:type :particle-cloud
     :choices (:choices vt)
     :weights (:weight vt)
     :n (:n-particles vt)}))

;; A distribution renders as its density curve
(extend-type genmlx.dist.core/Distribution
  IRenderable
  (render-spec [d]
    {:type :density-curve
     :dist d
     :range (distribution-range d)
     :n-points 200}))
```

Ask any value "how should I see you?" and it tells you — as data.

---

## One Process, Three Surfaces

The single nbb process serves all three interfaces simultaneously:

```
nbb process
│
├── stdout ──── Ink/Reagent TUI ─── terminal (primary)
│               Flexbox layout (Yoga engine)
│               Reagent atoms → re-render
│               Inline images via terminal-image
│
├── :7777 ──── Express + WebSocket ─── browser (optional)
│               Same Reagent components → hiccup → DOM
│               Observable Plot native (interactive, zoomable)
│               WebGPU 3D (posterior point clouds)
│               WebSocket pushes state diffs
│
├── :1337 ──── nREPL server ─── editor (optional)
│               Emacs/VS Code/IntelliJ CIDER
│               Full REPL into the running studio
│               `(render-spec @universe)` from your editor
│
└── GenMLX ─── MLX GPU ─── Apple Silicon Metal
                All inference. All visualization compute.
                Unified memory. Zero copy. Same address space.
```

**Zero architecture tax.** The browser doesn't require a separate server.
The editor doesn't require a separate REPL. The terminal doesn't require
a separate renderer. It's all the same atom, the same code, the same state.

### Why This Is Different from Every Other PPL Environment

| System | Architecture | Processes | Serialization |
|---|---|---|---|
| Stan + R | C++ engine + R driver + CmdStan CLI | 3+ | CSV files |
| PyMC + Jupyter | Python + Jupyter server + browser | 3 | JSON over WebSocket |
| Turing.jl + Pluto | Julia + Pluto server + browser | 3 | MsgPack |
| NumPyro + Jupyter | Python/JAX + Jupyter server + browser | 3 | JSON |
| Gen.jl + Jupyter | Julia + Jupyter server + browser | 3 | JSON |
| **GenMLX Studio** | **nbb (everything)** | **1** | **None** |

One process means: function call latency, not network latency. Clojure
data structures, not serialization formats. Atoms, not message passing.

---

## The LLM as First-Class Collaborator

The LLM agent lives in the same process. It doesn't call an API to run
inference — it calls a function. It doesn't receive JSON — it receives
Clojure data structures (serialized to tool results, but generated from
the same data).

### The Agent Sees What You See

```clojure
(def tools
  [{:name "define-model"
    :description "Define a generative function. The model is stored as an
                  immutable value. Returns the model's prior predictive."
    :execute (fn [{:keys [name source]}]
               (let [model (eval-model source)]
                 ;; Same event the UI would emit
                 (emit! {:type :define-model :name (keyword name) :model model})
                 ;; Return what the LLM needs to reason about
                 {:status :ok
                  :trace-addresses (model-addresses model)
                  :prior-predictive (prior-summary model 100)}))}

   {:name "run-inference"
    :execute (fn [{:keys [model algorithm samples]}]
               (let [traces (run-algorithm model algorithm samples)]
                 (emit! {:type :inference-complete
                         :model model :algorithm algorithm :traces traces})
                 ;; The LLM receives diagnostics as data
                 {:status :ok
                  :diagnostics (diagnostics-summary traces)}))}

   {:name "compare-models"
    :execute (fn [{:keys [models]}]
               ;; Run inference on all models, return comparison
               (let [results (pmap #(run-and-diagnose %) models)]
                 {:comparison (comparative-summary results)}))}

   {:name "render"
    :description "Create a visualization. The spec is a Clojure map.
                  It will be rendered to the user's current surface
                  (terminal or browser)."
    :execute (fn [{:keys [spec]}]
               (emit! {:type :render :spec spec})
               (render (current-target) spec)
               {:status :rendered})}])
```

The agent emits the same events that manual user actions emit. The session
log doesn't distinguish between human-initiated and LLM-initiated actions.
They're all just events in the fold.

### The Agent as Modeling Collaborator

The system prompt gives the LLM deep knowledge of GenMLX:

```
You are a probabilistic modeling collaborator in GenMLX Studio.
You share a process with GenMLX — a purely functional PPL on
Apple Silicon with 27 distributions, 10 combinators, and 29
inference algorithms.

When the user describes a problem:
1. Define a model as a gen body (data-driven, purely functional)
2. Visualize the prior predictive (always — catch bad priors early)
3. Choose an inference algorithm (NUTS for continuous, Gibbs for
   discrete, SMC for sequential, VI for fast approximation)
4. Run inference, show diagnostics
5. Visualize the posterior, interpret results
6. Suggest model improvements based on diagnostics

You write idiomatic ClojureScript. You use GenMLX's combinators
(Map, Unfold, Switch, Scan) for structured models. You understand
that every value in this system is immutable.
```

This isn't a chatbot with access to a PPL. It's a modeling partner that
thinks in the same language as the system.

---

## GPU Everywhere It Matters

MLX unified memory means the GPU is not "over there." It's right here,
in the same address space, with zero-copy access.

### Inference (obviously)

All 29 algorithms run on Metal. HMC leapfrog steps, NUTS tree building,
SMC particle propagation, VI gradient updates — all GPU.

### Visualization Computation

The expensive part of visualization is computation, not rendering.
Keep it on GPU:

```clojure
;; GPU-accelerated kernel density estimation
(defn gpu-kde
  "Evaluate KDE on GPU. ~1ms for 10,000 samples × 200 grid points."
  [samples grid bandwidth]
  ;; samples: [N] MLX array, grid: [M] MLX array
  ;; Entire computation stays on Metal until final extraction
  (let [diff   (mx/subtract (mx/reshape grid [-1 1])
                             (mx/reshape samples [1 -1]))     ;; [M,N]
        kernel (mx/exp (mx/negate
                         (mx/divide (mx/square diff)
                                    (mx/multiply 2 (mx/square bandwidth))))) ;; [M,N]
        density (mx/mean kernel 1)]                            ;; [M]
    ;; Only extract at the render boundary
    (mx/eval! density)
    density))

;; GPU-accelerated score surface (for 2D posterior heatmaps)
(defn gpu-score-surface
  "Evaluate model log-posterior on a 200×200 grid. ~5ms via vgenerate."
  [model args grid-x grid-y fixed-choices]
  (let [n (* (count grid-x) (count grid-y))
        ;; Build N constraint choicemaps for the grid
        constraints (for [x grid-x, y grid-y]
                      (cm/choicemap :slope x :intercept y))
        ;; Single batched call — GenMLX's vectorized execution
        results (dyn/vgenerate model args (merge-constraints constraints fixed-choices) n)]
    ;; [200,200] shaped score surface, computed in one GPU dispatch
    (mx/reshape (:weight results) [(count grid-x) (count grid-y)])))
```

### Compiled Inference Chains

GenMLX's loop compilation fuses multi-step MCMC chains into single Metal
dispatches. The entire warmup + sampling phase becomes one GPU kernel:

```clojure
;; 1000 NUTS steps compiled into a single Metal dispatch
(def compiled-nuts
  (mx/compile-fn
    (fn [init-state key]
      (loop [state init-state
             key key
             i 0]
        (if (>= i 1000)
          state
          (let [[key' subkey] (rng/split key)]
            (recur (nuts-step state subkey) key' (inc i))))))))
```

This is something no other PPL on consumer hardware can do. JAX has
`jit`+`scan`, but requires NVIDIA hardware and a Python runtime. GenMLX
compiles to Metal in ClojureScript.

---

## The Notebook Is a Value

No file format. No `.ipynb` JSON. A notebook is a vector of cells, and
a cell is a map:

```clojure
(def notebook
  [{:type :markdown
    :content "# Bayesian Linear Regression"}

   {:type :code
    :source "(def model (gen [xs] ...))"
    :result {:type :model :name :regression}}

   {:type :code
    :source "(def posterior (nuts model data {:samples 1000}))"
    :result {:type :inference-run :algorithm :nuts :n-samples 1000}}

   {:type :viz
    :spec {:type :posterior-histogram
           :data (extract-samples posterior [:slope :intercept])}}

   {:type :markdown
    :content "The slope posterior is 2.01 ± 0.12, consistent with the data."}])
```

Standard Clojure operations on notebooks:

```clojure
;; All code cells
(filter #(= (:type %) :code) notebook)

;; All visualization specs
(keep :spec notebook)

;; Add a cell
(conj notebook {:type :code :source "(show-prior model 100)"})

;; Notebook as EDN (serialization is pr-str)
(spit "session.edn" (pr-str notebook))

;; Notebook from EDN
(def restored (edn/read-string (slurp "session.edn")))
```

### Reactive Notebooks via Watches

When a model changes, everything downstream reacts:

```clojure
;; The dependency graph is implicit in the atom watches
(add-watch universe :auto-infer
  (fn [_ _ old new]
    (when (not= (:models old) (:models new))
      ;; Model changed — re-run inference on the active model
      (when-let [focus (:focus new)]
        (async-infer! focus)))))

(add-watch universe :auto-viz
  (fn [_ _ old new]
    (when (not= (:runs old) (:runs new))
      ;; New inference results — re-render active visualizations
      (doseq [spec (:specs new)]
        (render (current-target) spec)))))
```

Change a prior → model updates (immutable, new value) → inference reruns
→ visualization re-renders. All automatic. All via Clojure's watch
mechanism. No custom reactive framework needed.

---

## Transducers as the Universal Pipeline

The visualization pipeline, the data extraction pipeline, the
diagnostics pipeline — all transducers:

```clojure
;; Extract and transform posterior samples
(def posterior-xf
  (comp
    (map :choices)                              ;; Trace → ChoiceMap
    (map #(select-keys-from-cm % [:slope :intercept]))  ;; subset
    (map cm-to-numbers)))                       ;; MLX arrays → JS numbers

;; Apply to traces (lazy, composable, reusable)
(into [] posterior-xf traces)

;; Chain with diagnostics
(def diagnose-xf
  (comp
    posterior-xf
    (partition-all 100)                         ;; batch
    (map compute-batch-stats)))                 ;; mean, var per batch

;; Pipe directly into visualization
(def viz-data (transduce posterior-xf conj traces))
(render target {:type :scatter :data viz-data :x :slope :y :intercept})
```

Transducers give you:
- **Composability.** Snap pipeline stages together like Lego.
- **Laziness.** Process 100,000 samples without holding them all in memory.
- **Reuse.** The same extraction pipeline feeds diagnostics AND visualization.
- **No intermediate collections.** Data flows through without allocating vectors.

---

## Code Editing: Three Tiers

### Tier 1: $EDITOR Delegation (Immediate)

Like OpenCode: press `e` to open your `$EDITOR` (vim/emacs/helix). You
already have paredit, rainbow parens, CIDER integration. When you save,
the studio reads the file and evaluates it.

This works today. Zero code required.

### Tier 2: Syntax-Highlighted Display + LLM Editing (Primary)

The studio displays code cells with full Clojure syntax highlighting
via [Shiki](https://shiki.style/) (TextMate grammars → ANSI) or
[lezer-clojure](https://github.com/lezer-parser/clojure) (incremental
parser → custom colorizer). Read-only in the TUI, beautifully colored.

The LLM writes and modifies code. You guide it in natural language.
This is actually the most natural interaction model for a probabilistic
programming assistant — you describe what you want, the LLM writes the
model.

An input line accepts single expressions for quick evaluation:

```
> (mx/mean (extract-samples posterior [:slope]))
2.012
```

### Tier 3: Structural Editor (Future)

Built on lezer-clojure (the same parser that powers Clerk's editor):

```
lezer-clojure (parse) → syntax tree
    ├── ANSI colorizer (terminal) or CSS spans (browser)
    ├── paredit operations (slurp, barf, expand, contract)
    ├── auto-indent
    └── ink-multiline-input (text buffer + keyboard events)
```

The parser is shared between terminal and browser. The paredit logic
is shared. Only the rendering differs.

The grammar (`@nextjournal/lezer-clojure`, npm) is the hard part, and
it's already done. The structural editing operations are pure functions
on syntax trees. The rendering is a multimethod.

---

## Model Versioning

Every model definition creates a new immutable value. The studio tracks
the lineage:

```
:regression    → (gen [xs] ... (gaussian 0 10) ...)     ;; wide prior
:regression-v2 → (gen [xs] ... (gaussian 0 5) ...)      ;; tighter prior
:regression-v3 → (gen [xs] ... (half-cauchy 1) ...)      ;; robust noise
```

Compare posteriors across versions:

```clojure
(render target
  {:type :model-comparison
   :models [:regression :regression-v2 :regression-v3]
   :param :slope
   :overlay :density})
```

This is **version control for probabilistic models**, built from
immutable values and a vector of events. No Git needed — the event
log IS the history.

### Branching and Merging

Fork a session to explore alternatives:

```clojure
;; Branch: explore two different model structures
(def branch-a (take 5 (:events @universe)))  ;; events up to model definition
(def branch-b (conj branch-a
                {:type :define-model :name :mixture
                 :model (gen [xs] ... (mix ...) ...)}))

;; Reduce both branches
(def state-a (reduce session-step init branch-a))
(def state-b (reduce session-step init branch-b))

;; Compare
(render target
  {:type :branch-comparison
   :branches {:linear state-a :mixture state-b}
   :metric :waic})
```

This is trivial because state is a value and the reducer is a pure function.
In Jupyter, you'd need to restart the kernel.

---

## The Protocols

Everything pluggable. Everything extensible. Everything open.

```clojure
(defprotocol IRenderable
  (render-spec [this]
    "Return a declarative visualization spec for this value."))

(defprotocol ISummarizable
  (summary [this]
    "Return a concise human-readable summary as a string.")
  (detail [this]
    "Return a detailed inspection as structured data."))

(defprotocol ISerializable
  (to-edn [this]
    "Serialize to EDN-compatible data (no MLX arrays — extract first).")
  (from-edn [type data]
    "Deserialize from EDN data."))
```

Every GenMLX type — Trace, ChoiceMap, Distribution, VectorizedTrace,
inference results — implements these. The studio doesn't need to know
about specific types. It just asks values to describe themselves.

### User Extension

Users extend the system the same way GenMLX extends distributions — via
open multimethods and protocol extension:

```clojure
;; Add a custom visualization type
(defmethod render [:terminal :correlation-matrix]
  [_ {:keys [data params]}]
  (let [matrix (compute-correlation data)]
    (render-heatmap-inline matrix)))

;; Add a custom distribution
(defdist student-t [df loc scale]
  :sample (fn [key] ...)
  :log-prob (fn [x] ...))

;; It automatically renders, serializes, and integrates
(render-spec (student-t 3 0 1))
;; => {:type :density-curve :dist ... :range [-10 10]}
```

No plugin framework. No registration API. Just Clojure's open dispatch.

---

## The Full Project Structure

```
genmlx-studio/
├── package.json                   npm: mlx, ink, anthropic-sdk, sharp, etc.
├── studio.cljs                    Entry point: (start! {:mode :tui})
│
├── src/genmlx/studio/
│   │
│   │── core.cljs                  The atom. The reducer. The truth.
│   │                              defonce universe, session-step, emit!
│   │
│   │── event.cljs                 Event type specs. Validation.
│   │                              All events are plain maps.
│   │
│   │── spec.cljs                  Visualization spec builders.
│   │                              histogram, scatter, trace-plot, density,
│   │                              score-surface, particle-cloud, comparison.
│   │                              Pure functions returning data.
│   │
│   │── data.cljs                  Transducers: trace → plot data.
│   │                              GPU-accelerated extraction and summarization.
│   │                              gpu-kde, extract-samples, diagnostics-xf.
│   │
│   │── agent.cljs                 LLM agent: tool definitions, agent loop,
│   │                              streaming, system prompt.
│   │                              Tools emit events, same as user actions.
│   │
│   ├── render/
│   │   ├── protocol.cljs          (defmulti render [target spec])
│   │   ├── terminal.cljs          Observable Plot → SVG → PNG → inline
│   │   ├── browser.cljs           Observable Plot in DOM, Vega-Lite
│   │   ├── latex.cljs             pgfplots for paper figures
│   │   └── ascii.cljs             asciichart, Unicode blocks
│   │
│   ├── ui/
│   │   ├── tui.cljs               Ink/Reagent terminal components
│   │   ├── web.cljs               Reagent browser components (same hiccup)
│   │   └── shared.cljs            Component logic shared between surfaces
│   │
│   ├── edit/
│   │   ├── highlight.cljs         lezer-clojure → ANSI (terminal) or spans (browser)
│   │   └── paredit.cljs           Structural editing on syntax trees
│   │
│   └── io/
│       ├── session.cljs           Save/load event logs as EDN
│       ├── nrepl.cljs             nREPL server for editor integration
│       └── http.cljs              Express + WebSocket for browser mode
│
└── prompts/
    └── system.md                  LLM system prompt
```

### What's Shared Between Terminal and Browser

| Layer | Terminal | Browser | Shared |
|-------|----------|---------|--------|
| State | r/atom | r/atom | **100% — same atom** |
| Events | emit! | emit! | **100% — same reducer** |
| Viz specs | specs.cljs | specs.cljs | **100% — same specs** |
| Data xforms | data.cljs | data.cljs | **100% — same transducers** |
| Agent | agent.cljs | agent.cljs | **100% — same tools** |
| Highlight | lezer → ANSI | lezer → CSS | **Parser shared** |
| Paredit | lezer ops | lezer ops | **100% — same ops** |
| Components | Ink Box/Text | hiccup div/span | Logic shared, elements differ |
| Render | Plot→SVG→PNG→inline | Plot→DOM | Specs shared, final step differs |
| GPU | MLX (same process) | MLX (same process) | **100%** |

**~70% complete code sharing.** The only differences are the final rendering
step (ANSI vs DOM) and the component elements (Ink vs HTML).

---

## Dependencies

```json
{
  "dependencies": {
    "@frost-beta/mlx": "^0.4.0",
    "ink": "^4.0.0",
    "@inkjs/ui": "^2.0.0",
    "@anthropic-ai/sdk": "^0.30.0",
    "@observablehq/plot": "^0.6.0",
    "jsdom": "^24.0.0",
    "sharp": "^0.33.0",
    "terminal-image": "^3.0.0",
    "asciichart": "^1.5.0",
    "@nextjournal/lezer-clojure": "^1.0.0",
    "shiki": "^1.0.0",
    "express": "^4.18.0",
    "ws": "^8.0.0"
  }
}
```

All npm packages. All requireable from nbb. No build step.

### Launch

```bash
# Terminal mode (default, primary)
nbb studio.cljs

# Terminal + browser
nbb studio.cljs --browser 7777

# Terminal + browser + nREPL
nbb studio.cljs --browser 7777 --nrepl 1337

# Just the REPL (minimal, for scripting)
nbb studio.cljs --repl
```

---

## What No Other PPL Has

1. **Single-process architecture.** Every other PPL has at least two
   processes (compute engine + UI layer). GenMLX Studio is one nbb
   process with zero serialization.

2. **Purely functional session.** No other PPL environment treats the
   interactive session as an immutable event log with a pure reducer.
   Jupyter has execution order bugs and hidden mutable state. R/Stan
   scripts have global variables.

3. **Multi-surface rendering from one spec.** Same visualization data
   structure renders to terminal, browser, LaTeX, or ASCII. No other
   PPL has this.

4. **LLM as first-class collaborator.** The LLM isn't a plugin or a
   separate service. It's in the same process, calling the same
   functions, emitting the same events.

5. **Consumer GPU, native.** Apple Silicon unified memory. No CUDA
   drivers. No cloud GPUs. No Docker. `npm install && nbb studio.cljs`.

6. **Formal foundations.** 6,652 lines of proofs backing 11 runtime
   contracts. No other PPL environment can point to a formal proof that
   its inference is correct.

7. **Everything is data.** Models, traces, visualizations, sessions,
   events, specs — all plain Clojure values. All composable with
   standard library functions. All serializable as EDN.

---

## The Dream Interaction

```
$ nbb studio.cljs

  GenMLX Studio
  Apple Silicon GPU · MLX · 27 distributions · 29 algorithms

> I want to model how plant growth depends on sunlight and water,
  with some interaction effect. I have 50 data points.

  Studio: I'll build a Bayesian regression with an interaction term.

  ┌─ model ─────────────────────────────────────────────────────┐
  │ (gen [sunlight water growth]                                │
  │   (let [β₁ (trace :β-sun    (gaussian 0 5))                │
  │         β₂ (trace :β-water  (gaussian 0 5))                │
  │         β₃ (trace :β-inter  (gaussian 0 2))                │
  │         σ  (trace :σ        (half-cauchy 1))]               │
  │     (doseq [[j [s w g]] (map-indexed vector                 │
  │                           (map vector sunlight water growth))]│
  │       (trace (keyword (str "y" j))                          │
  │              (gaussian (+ (* β₁ s) (* β₂ w)                │
  │                           (* β₃ s w))                       │
  │                        σ)))))                                │
  └─────────────────────────────────────────────────────────────┘

  Prior predictive (100 draws):

  ┌─────────────────────────────────────────────────────────────┐
  │ [inline PNG: 3D scatter, wild prior draws vs data points]   │
  └─────────────────────────────────────────────────────────────┘

  The interaction prior N(0,2) looks reasonable. Running NUTS...

  ● NUTS 1000/1000  R̂: 1.001  ESS: 945  accept: 0.83  [1.3s]

  ┌─ posterior ─────────────────────────────────────────────────┐
  │ [inline PNG: 4-panel posterior densities with 95% CIs]      │
  │                                                             │
  │ β-sun:   1.42 ± 0.18  [1.07, 1.77]                         │
  │ β-water: 0.89 ± 0.15  [0.60, 1.18]                         │
  │ β-inter: 0.34 ± 0.09  [0.16, 0.52]  ← significant          │
  │ σ:       0.67 ± 0.07  [0.54, 0.82]                         │
  └─────────────────────────────────────────────────────────────┘

  The interaction effect is credibly positive (95% CI excludes 0).
  Plants that get both sun AND water grow more than the sum of
  individual effects.

> Show me the score surface for β-sun and β-inter

  ┌─────────────────────────────────────────────────────────────┐
  │ [inline PNG: contour plot of log-posterior density           │
  │  x-axis: β-sun, y-axis: β-inter                            │
  │  positive correlation visible in elliptical contours]       │
  └─────────────────────────────────────────────────────────────┘

  The posterior shows mild positive correlation between sun effect
  and interaction effect — makes sense ecologically.

> Export the posterior figure for my paper

  ✓ Saved: figures/posterior.pdf (LaTeX/pgfplots)
  ✓ Saved: figures/posterior.png (300 DPI)

> Save this session

  ✓ Saved: sessions/plant-growth-2026-03-01.edn (23 events)
  To restore: (load-session "sessions/plant-growth-2026-03-01.edn")
```

---

## Implementation Phases

| Phase | What | Days | Cumulative |
|-------|------|------|------------|
| 0 | POC: Ink + Plot→PNG + GenMLX inference in one process | 1 | 1 |
| 1 | Core: atom, reducer, event log, render protocol | 3 | 4 |
| 2 | Terminal: Ink TUI, syntax highlighting, viz pipeline | 3 | 7 |
| 3 | Agent: LLM tools, streaming, system prompt | 3 | 10 |
| 4 | Live: streaming diagnostics, reactive watches | 2 | 12 |
| 5 | Browser: Express + WebSocket + Reagent web components | 3 | 15 |
| 6 | Editor: lezer-clojure highlighting, $EDITOR delegation | 2 | 17 |
| 7 | Polish: LaTeX export, session save/load, model versioning | 3 | 20 |

**MVP (Phases 0-3): 10 days.** Terminal TUI + LLM agent + GPU inference + inline charts.

**Full system (Phases 0-7): 20 days.** Terminal + browser + editor integration + LaTeX export + session management.

---

## Philosophical Coda

Rich Hickey said: "State is the new spaghetti code." Every mutable
assignment is a place where time leaks into your program.

GenMLX Studio takes this seriously. The entire interactive session —
models, inference, visualization, conversation, branching, comparison —
is an immutable value derived from a vector of events via a pure function.
There is no hidden state. There is no execution order bug. There is no
"but I ran that cell before."

This is what happens when you build a probabilistic programming
environment in a language that was designed, from the ground up, to
treat values as sacred.

And it all runs in one process, on your laptop's GPU, with one command.

```bash
nbb studio.cljs
```
