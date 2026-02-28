# Paper 5: ClojureScript as a PPL Host Language

**Title:** "Persistent Data Structures and Open Multimethods as PPL Primitives: A Case Study in Language-PPL Fit"

**Target venue:** NOT TOPML — this is a PL/language design paper, not a
probabilistic ML paper.

**Recommended venues:**
- [Onward!](https://2026.splashcon.org/track/splash-2026-Onward-papers) (co-located
  with SPLASH) — values reflective, argumentative papers about software design
- [ICFP Experience Report](https://icfp26.sigplan.org/) — functional programming
  in practice
- [JFP](https://www.cambridge.org/core/journals/journal-of-functional-programming)
  (Journal of Functional Programming) — longer, reflective papers welcome
- [DLS](https://conf.researchr.org/home/dls-2026) (Dynamic Languages Symposium)

**Status:** No draft. Evidence throughout codebase and `paper/genmlx.md`.

---

## Why Not TOPML

TOPML's scope is probabilistic machine learning: inference algorithms, Bayesian
modelling, uncertainty quantification. This paper's thesis is about *programming
language design* — why ClojureScript's features reduce PPL implementation
complexity. TOPML reviewers would find it interesting but off-scope. The right
audience is PL researchers who design languages and DSLs.

---

## Thesis

> ClojureScript's core language features — immutable persistent data structures,
> open multimethods, protocols, and a minimal macro system — are not just
> *convenient* for implementing a probabilistic programming language; they are
> *structurally natural* in a way that reduces implementation complexity by 2-5x.
> We support this claim by comparing GenMLX (~10,800 LOC) with Gen.jl (~20,000+
> LOC) and GenJAX (~10,000+ LOC), all implementing the same Generative Function
> Interface, and showing that the features enabling compact implementation also
> enable formal verification (6,652 lines of proofs, vs 0 for Gen.jl/GenJAX).

**Corrected claims:**
- GenMLX LOC: ~10,800 (verified)
- Property tests: 153 (not 162)
- gen macro: 19 lines (verified)
- Trace defrecord: 1 line (verified)

---

## The Central Argument

| PPL Need | ClojureScript | Gen.jl (Julia) | GenJAX (Python/JAX) |
|----------|--------------|----------------|---------------------|
| Immutable traces | defrecord (1 line) | Mutable struct | Dataclass |
| Extensible dispatch | Protocols + multimethods | Multiple dispatch | Class hierarchy |
| Hierarchical data | Persistent maps (built-in) | Custom trie (~500 LOC) | Custom pytree |
| DSL | gen macro (19 lines) | @gen macro (~hundreds of lines) | Tracing-based |
| GPU numerics | MLX interop | N/A (CPU only) | JAX native |

---

## Paper Outline (target: 15-18 pages for Onward!/ICFP)

```
1. Introduction (2 pages)
2. The Generative Function Interface (2 pages)
3. ClojureScript's PPL Primitives (4 pages)
   3.1 Persistent Data Structures as Traces
   3.2 Protocols as GFI Operations
   3.3 Multimethods as Distribution Dispatch
   3.4 Persistent Maps as Choice Maps
   3.5 The gen Macro: 19 Lines
4. Quantitative Comparison (3 pages)
   4.1 Lines of Code by Layer
   4.2 What Accounts for the Difference?
   4.3 Code Density Analysis
5. Formal Verification as a Consequence (2 pages)
   5.1 Purity Enables Proofs (6,652 lines vs 0)
   5.2 The Mutable State Inventory (6 locations)
6. Limitations and Tradeoffs (1.5 pages)
7. Broader Lessons (1 page)
8. Related Work (1.5 pages)
9. Conclusion (0.5 pages)
```

---

## Assessment

**Strongest points:**
- gen macro = 19 lines is remarkable
- 10K vs 20K+ LOC with feature parity
- 6,652 lines of proofs (enabled by purity) vs 0 for alternatives

**Weakest points:**
- LOC comparisons are always debatable
- "Language fit" is hard to quantify rigorously
- The three systems differ in scope and maturity

**Risk level: Medium-high.** Best suited for Onward! (which explicitly values
opinion papers) or an ICFP experience report (which values practical insights).

---

## Effort Estimate

- Gather LOC data from Gen.jl and GenJAX repos: 1 day
- Write paper: 5 days
- Polish: 1 day

**Total: ~7 days**

---

## Priority

**Lowest priority of the five papers.** Submit Papers 1 and 2 to TOPML first.
If time permits and the LOC analysis reveals clean, compelling data, write this
for Onward! or ICFP. Otherwise, the language design argument is adequately made
in Paper 1's introduction and architecture sections.
