# GenMLX Studio (Wild Edition): Terminal-Native Probabilistic Programming

> One nbb process. One terminal. GPU inference, LLM assistant, inline
> publication-quality charts. No browser, no build step, no second runtime.

**Status:** Brainstorm / plan. No code yet.

---

## Vision

GenMLX Studio is a terminal-native environment where you interact with a
probabilistic programming system through natural language and ClojureScript,
with an LLM assistant that understands models and inference, GPU-accelerated
computation via MLX, and publication-quality visualizations rendered inline
in your terminal.

Everything runs in **one nbb process**:

```
Single nbb process
├── Ink/Reagent TUI          (renders to terminal, reactive)
├── LLM Agent                (Anthropic API, tool loop)
├── GenMLX                   (probabilistic programming)
├── MLX GPU                  (@frost-beta/mlx, Metal compute)
└── Visualization Pipeline   (Observable Plot → SVG → PNG → terminal inline)
```

Zero serialization. Zero network hops. The LLM, the GPU, the visualization,
and the UI are all in the same memory space.

---

## Why Terminal-Native?

### vs Browser-based (Clerk, Jupyter)

| Dimension | Browser | Terminal |
|---|---|---|
| Processes | 2+ (server + browser) | 1 (nbb) |
| GPU → display latency | ~50-100ms (serialize, HTTP, render) | ~5ms (same process) |
| Build step | Often needed (JS bundle) | None |
| LLM integration | Separate service | Same process, native tools |
| Developer context | Switch to browser tab | Stay in terminal |
| tmux/zellij | Awkward (browser is separate) | Natural (another pane) |
| Offline | Needs localhost server | Just run the binary |
| Image quality | Full (HTML/SVG/WebGPU) | Full (iTerm2/Kitty inline PNG) |

### Terminal Image Capabilities (2025-2026)

Modern terminals display **full-resolution inline images**:

| Protocol | Quality | Terminal Support |
|---|---|---|
| iTerm2 Inline Images | Excellent (full PNG) | iTerm2, WezTerm, many others |
| Kitty Graphics | Excellent (full PNG, GPU-accelerated) | Kitty, Ghostty, WezTerm, Konsole |
| Sixel | Good (palette-limited, dithered) | 21+ terminals including VS Code |
| ANSI blocks | Basic (low-res fallback) | Every terminal |

On macOS with Apple Silicon (the GenMLX target platform), iTerm2 and WezTerm
both support inline PNG display at native resolution. Charts rendered by
Observable Plot look identical to browser renders.

---

## Architecture

### The Single-Process Advantage

```
┌─────────────────────────────────────────────────────────────────┐
│  nbb process                                                     │
│                                                                   │
│  ┌─ Ink/Reagent TUI ──────────────────────────────────────────┐  │
│  │  Reactive terminal UI via Reagent atoms                     │  │
│  │  Flexbox layout (Yoga engine, same as React Native)         │  │
│  │  Components: Text, Box, Spinner, ProgressBar, Select, etc.  │  │
│  │  Re-renders on atom changes (same model as web Reagent)     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ LLM Agent ────────────────────────────────────────────────┐  │
│  │  @anthropic-ai/sdk (npm, direct require)                    │  │
│  │  Tool definitions for GenMLX operations                     │  │
│  │  Streaming responses rendered by Ink                        │  │
│  │  Conversation history in memory                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ GenMLX ───────────────────────────────────────────────────┐  │
│  │  Full probabilistic programming system (~10,800 lines)      │  │
│  │  27 distributions, 10 combinators, 29 inference algorithms  │  │
│  │  Called directly by LLM tools — no serialization             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ MLX GPU ──────────────────────────────────────────────────┐  │
│  │  @frost-beta/mlx (npm native addon)                         │  │
│  │  Apple Silicon unified memory (zero CPU-GPU transfer)       │  │
│  │  Metal compute shaders, lazy graph, explicit eval            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ Visualization Pipeline ───────────────────────────────────┐  │
│  │  Observable Plot + JSDOM → SVG string                       │  │
│  │  sharp → PNG buffer (rasterization)                         │  │
│  │  terminal-image → inline display (auto-detects protocol)    │  │
│  │  asciichart → ASCII fallback for unsupported terminals      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Works

1. **Ink + Reagent is documented and working with nbb.** The nbb README
   shows the pattern. Reagent atoms drive re-renders, same mental model as
   web development.

2. **All npm packages.** Every dependency is a standard npm package that
   nbb can `require`. No native compilation, no build step.

3. **Same process = no serialization.** When the LLM tool says "run NUTS",
   it's a function call. The result is a Clojure data structure in memory.
   No JSON encoding, no HTTP, no nREPL.

4. **MLX unified memory.** GPU results are accessible from CPU with zero
   copy. Extract values with `mx/item` only at the visualization boundary.

---

## The Visualization Pipeline

### High-Resolution Charts in Terminal

```clojure
(ns genmlx.studio.viz
  (:require ["@observablehq/plot" :as Plot]
            ["jsdom" :refer [JSDOM]]
            ["sharp$default" :as sharp]
            ["terminal-image$default" :as terminal-image]))

(defn render-chart!
  "Render an Observable Plot spec as an inline terminal image."
  [spec]
  (let [;; 1. Server-side SVG via Observable Plot + JSDOM
        doc (.-document (.-window (JSDOM. "")))
        plot (.plot Plot (clj->js (assoc spec :document doc)))
        _ (.setAttributeNS plot "http://www.w3.org/2000/xmlns/"
                           "xmlns" "http://www.w3.org/2000/svg")
        svg-str (.-outerHTML plot)

        ;; 2. Rasterize SVG → PNG via sharp
        png-buf (.toBuffer (sharp (js/Buffer.from svg-str))
                           #js {:format "png"})]

    ;; 3. Display inline via terminal-image (auto-detects iTerm2/Kitty)
    (.then png-buf
      (fn [buf]
        (.then (.buffer terminal-image buf #js {:width "100%"})
          println)))))
```

### What the Pipeline Produces

On iTerm2/WezTerm/Kitty: full-resolution PNG displayed inline in terminal
output. Scrollable, selectable. Looks identical to a browser render.

On terminals without image support: falls back to ANSI block characters
(lower resolution but universally compatible).

### ASCII Fallback

For Terminal.app, Alacritty, or SSH sessions:

```clojure
(ns genmlx.studio.ascii
  (:require ["asciichart$default" :as chart]))

(defn ascii-trace-plot [samples]
  (println (chart/plot (clj->js samples)
                       #js {:height 12 :padding "      "})))
```

Output:
```
  2.10 ┤                                              ╭─╮
  1.95 ┤                                    ╭─────────╯ ╰──╮
  1.80 ┤                          ╭─────────╯               ╰──╮
  1.65 ┤                ╭─────────╯                             ╰──╮
  1.50 ┤      ╭─────────╯                                         ╰──
  1.35 ┼──────╯
```

### Capability Detection

```clojure
(defn detect-terminal []
  (cond
    (some? js/process.env.TERM_PROGRAM)
    (let [term js/process.env.TERM_PROGRAM]
      (cond
        (= term "iTerm.app")  :iterm2
        (= term "WezTerm")    :wezterm
        (= term "ghostty")    :kitty
        :else                  :ascii))

    (some? js/process.env.KITTY_PID) :kitty
    :else :ascii))
```

Choose rendering strategy based on terminal:
- `:iterm2` / `:wezterm` / `:kitty` → full PNG inline images
- `:ascii` → asciichart + Unicode block drawing

---

## The LLM Agent

### Tool Definitions

The LLM has GenMLX-specific tools that operate directly in the same process:

| Tool | What It Does | Returns |
|---|---|---|
| `define-model` | Evaluate a `gen` body, store as a var | Model definition + prior predictive chart |
| `run-inference` | Run MH/HMC/NUTS/SMC/VI with options | Posterior samples + diagnostics |
| `show-prior` | Forward-sample N times, render chart | Inline prior predictive image |
| `show-posterior` | Render posterior histogram/scatter | Inline posterior image |
| `diagnose` | Compute R-hat, ESS, trace plots | Diagnostic summary + charts |
| `compare-algorithms` | Race multiple algorithms | ESS/sec comparison table + charts |
| `update-model` | Modify model (change prior, add data) | Updated model + new prior predictive |
| `explain-trace` | Show choicemap, score decomposition | Formatted trace tree |
| `eval-cljs` | Evaluate arbitrary ClojureScript | Result value |
| `score-surface` | Evaluate log-posterior on a grid | Inline contour/heatmap image |
| `sensitivity-sweep` | Sweep a hyperparameter, show effect | Inline animation or grid of charts |

### Tool Implementation Pattern

```clojure
(def tools
  [{:name "run-inference"
    :description "Run Bayesian inference on the current model.
                  Available algorithms: mh, hmc, nuts, smc, vi.
                  Returns posterior samples and diagnostics."
    :input_schema {:type "object"
                   :properties
                   {:algorithm {:type "string"
                                :enum ["mh" "hmc" "nuts" "smc" "vi"]}
                    :samples {:type "integer" :default 1000}
                    :warmup {:type "integer" :default 500}
                    :selection {:type "array" :items {:type "string"}
                                :description "Parameter names to infer"}}
                   :required ["algorithm"]}}])

(defn execute-tool [{:keys [name input]}]
  (case name
    "run-inference"
    (let [{:keys [algorithm samples warmup selection]} input
          algo-fn (case algorithm
                    "mh"   mcmc/mh
                    "hmc"  mcmc/hmc
                    "nuts" mcmc/nuts
                    "smc"  smc/smc
                    "vi"   vi/vi)
          ;; Direct function call — same process, same memory
          traces (algo-fn {:samples (or samples 1000)
                           :warmup (or warmup 500)
                           :selection (sel/select-addrs (map keyword selection))}
                          @current-model @current-args @current-obs)]
      ;; Store results, render diagnostics
      (reset! posterior traces)
      (render-diagnostics! traces selection)
      {:status "success"
       :n-samples (count traces)
       :r-hat (compute-r-hat traces selection)
       :ess (compute-ess traces selection)})

    "show-posterior"
    (let [samples (extract-samples @posterior (:params input))]
      (render-chart! (posterior-histogram-spec samples))
      {:status "rendered"})

    ;; ... other tools
    ))
```

### Agent Loop

```clojure
(ns genmlx.studio.agent
  (:require ["@anthropic-ai/sdk$default" :as Anthropic]))

(def client (Anthropic.))

(defn agent-step [messages]
  (.then
    (.create (.-messages client)
      #js {:model "claude-sonnet-4-6"
           :max_tokens 4096
           :system system-prompt
           :tools (clj->js tools)
           :messages (clj->js messages)})
    (fn [response]
      (let [content (js->clj (.-content response) :keywordize-keys true)]
        (doseq [block content]
          (case (:type block)
            "text" (render-assistant-text! (:text block))
            "tool_use" (let [result (execute-tool block)]
                         ;; Continue agent loop with tool result
                         (agent-step
                           (conj messages
                             {:role "assistant" :content content}
                             {:role "user"
                              :content [{:type "tool_result"
                                         :tool_use_id (:id block)
                                         :content (pr-str result)}]})))))))))
```

### System Prompt

```
You are GenMLX Studio, an expert probabilistic programming assistant.
You have direct access to GenMLX, a GPU-accelerated probabilistic
programming system running on Apple Silicon via MLX.

You can:
- Define probabilistic models using the `gen` macro
- Run inference (MH, HMC, NUTS, SMC, VI) on Apple Silicon GPU
- Visualize priors, posteriors, and diagnostics
- Explain model structure and inference results
- Compare inference algorithms
- Compute score surfaces and sensitivity analyses

When the user describes a modeling problem:
1. Define an appropriate model
2. Show the prior predictive to validate the model
3. Run inference with a suitable algorithm
4. Show diagnostics (R-hat, ESS, trace plots)
5. Present the posterior and interpret the results

Always show visualizations. The terminal supports inline images.
Use the visualization tools liberally.
```

---

## Example Conversation

```
┌─ GenMLX Studio ──────────────────────────────────────────────────┐
│                                                                   │
│  You: I have 5 data points from a linear relationship:           │
│       x = [1,2,3,4,5], y = [2.1, 3.9, 6.2, 7.8, 10.1]          │
│       Build me a model and run inference.                        │
│                                                                   │
│  Studio: I'll build a Bayesian linear regression model.          │
│                                                                   │
│  ┌─ Model ───────────────────────────────────────────────────┐   │
│  │ (def model                                                 │   │
│  │   (gen [xs]                                                │   │
│  │     (let [slope     (trace :slope (gaussian 0 5))          │   │
│  │           intercept (trace :intercept (gaussian 0 5))]     │   │
│  │       (doseq [[j x] (map-indexed vector xs)]               │   │
│  │         (trace (keyword (str "y" j))                       │   │
│  │                (gaussian (+ (* slope x) intercept) 1)))    │   │
│  │       slope)))                                             │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Here's the prior predictive — 100 forward samples:              │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ [inline PNG: scatter plot showing wild prior predictive     │   │
│  │  lines — slope and intercept sampled from N(0,5)]          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  The prior is fairly wide. Running NUTS with 1000 samples...     │
│                                                                   │
│  ◐ NUTS warmup 200/500  step-size: 0.31  accept: 0.74           │
│  ◑ NUTS warmup 350/500  step-size: 0.24  accept: 0.81           │
│  ● NUTS sampling 1000/1000  R-hat: 1.001  ESS: 892              │
│                                                                   │
│  ┌─ Posterior ────────────────────────────────────────────────┐   │
│  │ [inline PNG: two histograms side by side                   │   │
│  │  slope: centered at 2.0, tight                             │   │
│  │  intercept: centered at 0.1, tight]                        │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─ Diagnostics ──────────────────────────────────────────────┐  │
│  │ Parameter  │ Mean  │ Std   │ R-hat │ ESS  │ 95% CI        │  │
│  │ slope      │ 2.01  │ 0.12  │ 1.001 │ 892  │ [1.78, 2.24]  │  │
│  │ intercept  │ 0.08  │ 0.38  │ 1.002 │ 856  │ [-0.66, 0.82] │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Inference converged well. The slope posterior is 2.01 ± 0.12,   │
│  consistent with the data. The intercept is near zero.           │
│                                                                   │
│  You: Show me the posterior predictive                            │
│                                                                   │
│  ┌─ Posterior Predictive ─────────────────────────────────────┐   │
│  │ [inline PNG: data points with 95% credible band,           │   │
│  │  posterior mean line, and 50 posterior predictive draws]    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  You: Compare NUTS vs MH on this model                           │
│                                                                   │
│  ┌─ Algorithm Comparison ─────────────────────────────────────┐  │
│  │ Algorithm │ ESS/sec │ R-hat │ Time    │ Verdict             │  │
│  │ NUTS      │ 890     │ 1.001 │ 1.2s    │ Excellent           │  │
│  │ MH        │ 12      │ 1.34  │ 8.1s    │ Not converged       │  │
│  │ HMC       │ 340     │ 1.002 │ 2.0s    │ Good                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  You: _                                                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## The TUI Components (Ink + Reagent)

### Core Layout

```clojure
(ns genmlx.studio.ui
  (:require ["ink" :refer [render Text Box Newline]]
            ["@inkjs/ui" :refer [Spinner ProgressBar Badge StatusMessage
                                  TextInput]]
            [reagent.core :as r]))

(defonce app-state
  (r/atom {:messages []           ;; conversation history
           :input ""              ;; current user input
           :inference nil         ;; running inference state
           :model nil             ;; current model
           :posterior nil}))      ;; current posterior samples

(defn message-view [{:keys [role content]}]
  [:> Box {:flexDirection "column" :marginBottom 1}
   [:> Text {:bold true :color (if (= role "user") "blue" "green")}
    (if (= role "user") "You" "Studio")]
   [:> Text {} content]])

(defn inference-status []
  (when-let [{:keys [algorithm iteration total r-hat ess]} (:inference @app-state)]
    [:> Box {:marginY 1}
     [:> Spinner {:type "dots"}]
     [:> Text {} (str " " algorithm " " iteration "/" total
                      "  R-hat: " (when r-hat (.toFixed r-hat 3))
                      "  ESS: " (when ess (int ess)))]]))

(defn app []
  [:> Box {:flexDirection "column" :padding 1}
   [:> Text {:bold true :color "cyan"} "GenMLX Studio"]
   [:> Box {:flexDirection "column" :marginTop 1}
    (for [[i msg] (map-indexed vector (:messages @app-state))]
      ^{:key i} [message-view msg])]
   [inference-status]
   [:> Box {:marginTop 1}
    [:> Text {:color "blue" :bold true} "> "]
    [:> TextInput {:value (:input @app-state)
                   :onChange #(swap! app-state assoc :input %)
                   :onSubmit handle-submit}]]])

(render (r/as-element [app]))
```

### Live Inference Progress

```clojure
(defn run-inference-with-progress!
  "Run inference, updating the UI atom after each batch."
  [{:keys [algorithm samples warmup] :as opts} model args obs]
  (swap! app-state assoc :inference
         {:algorithm algorithm :iteration 0 :total (+ samples warmup)})

  ;; Run in batches, updating progress
  (let [batch-size 50]
    (loop [i 0 traces []]
      (if (>= i (+ samples warmup))
        (do (swap! app-state assoc :inference nil :posterior traces)
            traces)
        (let [batch (run-batch model args obs batch-size)
              new-traces (into traces batch)
              diagnostics (when (> (count new-traces) 100)
                            (quick-diagnostics new-traces))]
          (swap! app-state update :inference merge
                 {:iteration (min (+ i batch-size) (+ samples warmup))
                  :r-hat (:r-hat diagnostics)
                  :ess (:ess diagnostics)})
          (recur (+ i batch-size) new-traces))))))
```

---

## Inline Image Rendering

### The Full Pipeline

```
MLX GPU compute
    │
    ▼
mx/item (extract to JS numbers)
    │
    ▼
Observable Plot + JSDOM (server-side SVG)
    │
    ▼
sharp (SVG → PNG rasterization)
    │
    ▼
terminal-image (PNG → escape sequences)
    │
    ▼
stdout (terminal renders inline image)
```

### Specific Visualizations

| Visualization | Observable Plot Marks | When Used |
|---|---|---|
| Prior predictive | `Plot.line` + `Plot.dot` | After model definition |
| Posterior histogram | `Plot.rectY` + `Plot.binX` | After inference |
| Posterior scatter | `Plot.dot` + `Plot.density` | Joint posterior of 2 params |
| Trace plot | `Plot.line` (color by parameter) | During/after MCMC |
| Prior vs posterior | `Plot.areaY` (two layers) | After inference |
| Posterior predictive | `Plot.line` + `Plot.areaY` (CI band) | On request |
| Score surface | `Plot.raster` or `Plot.contour` | On request |
| Particle cloud (SMC) | `Plot.dot` (size = weight) | During/after SMC |
| ELBO curve | `Plot.line` | During/after VI |
| Algorithm comparison | `Plot.barX` (ESS/sec) | On request |
| R-hat display | `Plot.barX` + `Plot.ruleX` at 1.01 | After inference |

### Rendering Frequency for Live Updates

For live inference:
- **Text diagnostics** (R-hat, ESS, iteration count): update every iteration
  via Ink's reactive re-render
- **Inline charts** (trace plot): re-render every 50-100 iterations
  (rendering takes ~20-50ms, don't want to spam terminal)
- **Final charts** (posterior, diagnostics): render once at completion

The Ink TUI handles text updates at 60fps. Chart images are appended to
the scrollback — each re-render is a new image below the previous content
(terminals handle this naturally via `Ink.Static`).

---

## Comparison with OpenCode

### What We Take from OpenCode's Design

| OpenCode Pattern | GenMLX Studio |
|---|---|
| LLM agent with tool loop | Same — Anthropic API + GenMLX tools |
| Streaming responses | Same — Ink renders streamed text |
| Tool permission system | Simpler — all tools are read-only or GPU compute |
| File editing tools | Not needed — models defined in-process via `eval` |
| Conversation history | Same — in-memory message list |
| Persistent sessions | SQLite or EDN file for conversation log |
| Plugin system | `require` additional `.cljs` files |
| MCP servers | Could integrate for external data sources |

### What We Don't Take

| OpenCode Feature | Why Not |
|---|---|
| OpenTUI (Zig native core) | Requires Bun-specific FFI, not compatible with nbb |
| Bubble Tea (Go) | Wrong language ecosystem |
| File editing / bash tools | GenMLX Studio operates on models, not files |
| Multi-provider LLM | Start with Anthropic only, add others later |

### What We Add That OpenCode Doesn't Have

| Feature | Why It's Unique |
|---|---|
| GPU-accelerated computation | MLX runs inference on Metal |
| Inline publication-quality charts | Observable Plot → PNG → terminal |
| Domain-specific tools | Probabilistic programming operations |
| Live inference streaming | Watch MCMC/VI converge in real time |
| Prior/posterior visualization | Not possible in a general coding assistant |
| Score surface rendering | GPU evaluates 40K model instances for heatmap |

---

## Project Structure

```
genmlx-studio/
├── package.json                    npm deps
├── studio.cljs                     Entry point
│
├── src/genmlx/studio/
│   ├── core.cljs                   App state, initialization
│   ├── ui.cljs                     Ink/Reagent TUI components
│   ├── agent.cljs                  LLM agent loop + tool definitions
│   ├── tools.cljs                  GenMLX tool implementations
│   ├── viz.cljs                    Observable Plot → terminal pipeline
│   ├── ascii.cljs                  ASCII chart fallback
│   └── terminal.cljs               Terminal capability detection
│
└── prompts/
    └── system.md                   LLM system prompt
```

### npm Dependencies

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
    "asciichart": "^1.5.0"
  }
}
```

### Launch

```bash
npm install
nbb studio.cljs
```

One command. No build step. No browser. No second process.

---

## Implementation Phases

### Phase 0: Proof of Concept (~1 day)

Validate the core pipeline works end-to-end:

- [ ] Ink + Reagent TUI renders in nbb
- [ ] Observable Plot → SVG → sharp → PNG → terminal-image displays inline
- [ ] GenMLX model definition + inference in same process
- [ ] One inline chart showing posterior histogram

**Success criterion:** Run `nbb poc.cljs`, see a posterior histogram rendered
inline in iTerm2 from GPU-computed MCMC samples.

### Phase 1: Basic TUI + Visualization (~3 days)

- [ ] App shell with Ink (conversation view, input, status bar)
- [ ] Terminal capability detection (iTerm2/Kitty/Sixel/ASCII)
- [ ] Visualization pipeline: `render-chart!` with all fallbacks
- [ ] Basic charts: histogram, line plot, scatter
- [ ] Manual REPL mode (type ClojureScript, see results + charts)

### Phase 2: LLM Agent Integration (~3 days)

- [ ] Anthropic API integration via `@anthropic-ai/sdk`
- [ ] Tool definitions for define-model, run-inference, show-posterior
- [ ] Agent loop with streaming text display
- [ ] Conversation history and context management
- [ ] System prompt with GenMLX expertise

### Phase 3: Live Inference Streaming (~2 days)

- [ ] MCMC progress (iteration, R-hat, ESS updating in place via Ink)
- [ ] Trace plot re-rendered every N iterations
- [ ] VI ELBO curve updating live
- [ ] SMC particle count and log-ML updating
- [ ] Final diagnostic summary on completion

### Phase 4: Rich Tools + Visualizations (~3 days)

- [ ] Prior predictive tool (forward-sample + render)
- [ ] Posterior predictive tool (predict with uncertainty bands)
- [ ] Score surface tool (GPU grid evaluation + contour plot)
- [ ] Algorithm comparison tool (race and report)
- [ ] Choicemap tree viewer (formatted text)
- [ ] Diagnostics dashboard (R-hat, ESS, trace plots)

### Phase 5: Polish + Power Features (~3 days)

- [ ] Sensitivity sweep (parameter slider via CLI input)
- [ ] Session save/load (conversation + model state to EDN file)
- [ ] Multiple models in same session
- [ ] Custom model library (save/reuse model definitions)
- [ ] Export charts as PNG/SVG files
- [ ] nREPL server for editor integration (parallel to TUI)

---

## Advanced: GPU-Accelerated Visualization

While MLX cannot render pixels directly (it's a compute framework, not a render
framework), it can accelerate the **computational** part of visualization:

| Computation | MLX Advantage |
|---|---|
| KDE for density plots | Evaluate kernel on GPU grid (~1ms for 10K points) |
| Score surface | 40K model evaluations in one `vgenerate` batch (~5ms) |
| PCA / dimensionality reduction | Matrix operations on GPU |
| Histogram bin counting | Vectorized binning on GPU |
| Summary statistics | Mean, variance, quantiles on GPU arrays |

The pattern: **compute on GPU, extract summary, render on CPU.**

```clojure
;; GPU-accelerated KDE for posterior density plot
(defn gpu-kde [samples grid-points bandwidth]
  ;; samples: [N] MLX array, grid-points: [M] MLX array
  ;; All computation stays on GPU
  (let [diff (mx/subtract (mx/reshape grid-points [-1 1])
                           (mx/reshape samples [1 -1]))
        kernel (mx/exp (mx/divide (mx/negative (mx/square diff))
                                   (mx/multiply 2 (mx/square bandwidth))))
        density (mx/divide (mx/sum kernel 1) (mx/multiply n bandwidth
                                                           (mx/sqrt (* 2 js/Math.PI))))]
    ;; Only extract at the end
    {:x (vec (.tolist (mx/eval! grid-points)))
     :y (vec (.tolist (mx/eval! density)))}))
```

---

## Open Questions

1. **Ink version compatibility with nbb.** Ink 4+ is ESM-only. nbb handles
   ESM imports, but testing is needed. Ink 3 (CommonJS) is the safe fallback.

2. **sharp on Apple Silicon.** sharp has prebuilt binaries for macOS arm64.
   Should work out of the box. If not, `@napi-rs/canvas` is an alternative
   with zero system dependencies.

3. **terminal-image output format.** Does `terminal-image` output work
   correctly when interleaved with Ink's render loop? May need to use
   `Ink.Static` for images and `Ink.Box` for updating content.

4. **Conversation token budget.** Long inference sessions produce lots of
   tool results. Need a strategy for context window management — summarize
   old results, keep recent ones.

5. **Concurrent inference.** Can the Ink render loop and a long MCMC chain
   run concurrently in the same nbb process? Node.js is single-threaded,
   but MLX GPU dispatch is async. May need to yield between inference
   batches via `setTimeout` or `process.nextTick` to let Ink re-render.

6. **Chart dimensions.** Terminal width varies. Observable Plot needs
   explicit width/height. Read terminal dimensions via `process.stdout.columns`
   and scale chart accordingly.

---

## Effort Estimate

| Phase | Days | Cumulative |
|---|---|---|
| 0. Proof of concept | 1 | 1 |
| 1. Basic TUI + viz | 3 | 4 |
| 2. LLM agent | 3 | 7 |
| 3. Live streaming | 2 | 9 |
| 4. Rich tools | 3 | 12 |
| 5. Polish | 3 | 15 |

**MVP (Phases 0-2): 7 days.** A terminal app where you chat with an LLM that
can define models, run GPU inference, and show inline charts.

**Full system (Phases 0-5): 15 days.** Live inference streaming, rich
diagnostics, score surfaces, algorithm comparison, session management.

---

## The Dual-Mode Dream

The terminal TUI is the primary interface. But the same nbb process could
also serve a browser view for when you need richer visualization:

```
nbb process
├── Ink TUI on stdout (primary)
├── Express on :7777 (optional browser view)
│   ├── Same GenMLX state
│   ├── WebSocket push for live updates
│   ├── Observable Plot in browser (richer interaction)
│   └── WebGPU 3D (posterior point clouds, if needed)
└── nREPL on :1337 (editor integration)
```

Three interfaces to the same computation, same state, same process:
- **Terminal** for daily work
- **Browser** for presentations and 3D
- **nREPL** for editor integration

All optional. The terminal is always enough.
