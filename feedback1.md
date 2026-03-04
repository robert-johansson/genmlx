Yep — that layering is a very plausible “purity vision” for GenMLX, and it matches both (a) what MLX forces you to accept, and (b) what re-frame teaches: push impurity to a tiny boundary, represent everything else as pure transformations of data.

A few tweaks make it even tighter and more achievable.

First: what “pure” can realistically mean with MLX

Even if your CLJS code is “pure,” MLX arrays live in a world with:
	•	lazy evaluation (values aren’t computed until forced)
	•	caching / compilation (compile caches are global-ish)
	•	device memory management (dispose/tidy)
	•	randomness (if you use global RNG state)
	•	side-effectful forcing (eval!, item, etc.)

So the attainable goal is:
	•	Semantic purity at GenMLX level: running a program returns the same trace / score / weights / diagnostics given the same explicit inputs (including PRNG keys), and any forcing/compilation/memory actions are explicitly described and happen only in Layer 0.

That’s “pure” in the re-frame / effect-handler sense: pure core + explicit effect interpretation.

Your layers: what I’d keep, what I’d adjust

Layer 0: MLX Foundation (impure boundary)

✅ Absolutely. This is where:
	•	eval!, item, compile, cache ops
	•	tidy, dispose!
	•	device placement / stream sync
	•	performance counters

Rule: nothing above Layer 0 can perform these. They can only request them (as data).

⸻

Layer 1: Core Data (pure)

✅ Traces, choicemaps, diffs, selections, edits, kernels-as-data, etc.

Rule: this layer cannot reference MLX at all (or only as opaque values), if you want maximal purity and proof friendliness.

⸻

Layer 2: GFI & Execution (pure after state-threading macro)

✅, with an important constraint:

Right now GenMLX uses dynamic vars + a tiny volatile state. That’s “morally pure,” but not “definitionally pure.”

To make it re-frame-pure:
	•	represent execution as context -> context'
	•	make the handler “interpreter” explicit
	•	treat randomness and params as coeffects
	•	treat trace/score/grad/eval requests as effects

So Layer 2 becomes “pure interpreter” producing a plan.

⸻

Layer 3: DSL (pure)

✅. The DSL is already small; you can keep it pure by ensuring DSL forms only build data / functions and do not force MLX.

Rule: DSL cannot call eval! or materialize. It can emit an :mx/force effect request if needed.

⸻

Layer 4: Distributions (pure)

Mostly ✅, but the subtle gotcha is: if distribution code calls MLX randomness in an impure way, or forces evaluation, it leaks impurity upward.

To make “pure distributions” true in the strict sense:
	•	sample must be ([dist key] -> [x key'])
	•	log-prob must be a pure function of its inputs (even if the returned value is an MLX array thunk)
	•	any internal caching goes through Layer 0 and is requested as effects

So I’d annotate your Layer 4 as:
pure given explicit key + no forcing.

⸻

Layer 5: Combinators (pure)

✅ Same condition: they can compose “plans” and traces, but not force MLX.

⸻

Layer 6: Inference (pure after changes)

✅ This is where re-frame inspiration is strongest.

Inference should become:
	•	input: trace(s), kernel(s), keys, options
	•	output: updated trace(s) + diagnostics + effect requests (compile, eval, logging)

Any sampling step must thread keys. Any compiled-chain optimization must be represented as a request, not a direct action.

⸻

Layer 7: Vectorized (pure after changes)

✅, with the same two constraints:
	•	vectorization is a pure transformation of a plan/handler/context
	•	it must not rely on hidden global batch size or global MLX state

This layer is a good candidate for an “interceptor” style (before/after passes): before = lift values to batched domain; after = aggregate diagnostics, stack traces, etc.

⸻

Layer 8: Verification (pure)

✅ and I’d actually put “runtime contracts” here too, as a pure checker:
	•	input: plan + trace + constraints
	•	output: list of violations / proof obligations / metrics

The actual throwing/logging is an effect.

One structural improvement: split “Foundation” into two sublayers

If you want maximal clarity:
	•	Layer 0a: MLX API wrapper (impure)
Raw bindings and resource ops.
	•	Layer 0b: Effect runtime (impure)
The only place that interprets GenMLX “effects”:
	•	:mx/eval, :mx/compile, :mx/dispose
	•	:cache/*
	•	:log/*
	•	:metrics/*

That mirrors re-frame perfectly: core computes effects as data, runtime performs them.

The key rule that makes the whole vision work

If you adopt only one rule, make it this:

Above Layer 0, no code is allowed to force MLX computation or touch resource lifetime.
Instead, it returns a value plus an “effects map” describing what should be forced/compiled/disposed/logged.

That single rule gives you:
	•	testability (swap effect runtime with a mock)
	•	determinism (keys + explicit coeffects)
	•	replay (effects + plans are data)
	•	proof alignment (semantics described by pure interpreter)

What you’ll probably have to change in current GenMLX

Based on the codebase as it exists, the biggest purity leaks to watch for are:
	1.	Dynamic vars used as implicit coeffects
PRNG key, handler, param store.
Re-frame would push you to make these explicit context inputs.
	2.	Materialization inside model execution
Any mx/item, mx/eval! inside gen bodies or distribution sampling/log-prob.
	3.	Compilation and caches as hidden global behavior
Needs to be moved behind an explicit “effect interpreter” API.
	4.	Inference algorithms that “do IO” directly
Logging, profiling, checkpointing should become effects.

None of these are conceptually hard — they’re mostly architectural discipline.

Small refinement to your layer list (suggested)

If I rewrite your list in a way that’s closer to re-frame semantics:
	•	Layer 0: MLX + Effect Runtime  ← only place with !
	•	Layer 1: Core Data             ← pure data + algebra
	•	**Layer 2: Pure Interpreter (GFI)**← context → [result effects context’]
	•	Layer 3: DSL                   ← builds interpretable programs
	•	Layer 4: Distributions         ← pure + explicit key threading
	•	Layer 5: Combinators           ← pure program composition
	•	Layer 6: Inference             ← pure kernels producing effect requests
	•	Layer 7: Vectorized            ← pure lifting + aggregation
	•	Layer 8: Verification          ← pure checkers/contracts + proof hooks

That’s basically your vision, just with the “effects as data” point made explicit.

If you want, we can take one concrete GenMLX hotspot (e.g., run-handler + rng/*prng-key* + param-store) and sketch a re-frame-ish context/effects shape that would let you make Layers 2/6/7 definitionally pure without losing performance features like compiled MCMC chains.
