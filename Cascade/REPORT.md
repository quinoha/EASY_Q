# Implementing Cascade in PyTorch — Recommended Approach

This report covers the full recommended path for implementing "Cascade" (Gu, Bonilla
Ataides, Lukin & Yelin, "Scalable Neural Decoders for Practical Fault-Tolerant Quantum
Computation"). Code in this repository currently implements **Phase 1 only** (the
surface-code backbone); the rest is scoped here so the roadmap is explicit.

No Python interpreter was available in the environment this was written in (only a
Windows Store alias stub for `python`), so none of this has been executed. Set up a real
environment (`requirements.txt`) and run `tests/test_shapes.py` before trusting any of
the shape/gradient claims below.

## 1. Architecture mapping

The paper's Fig. 1a decomposes the decoder into `Embed → [Geometry-Aware Convolution] × L
→ Aggregate → Classify`. Extended Data Fig. 1 gives the literal per-layer structure (a
bottleneck residual block). This maps to PyTorch modules as:

| Paper element | Module (this repo) |
|---|---|
| "Binary detection events embedded into H-dim representations" | `cascade.model.embedding.SyndromeEmbedding` |
| Bottleneck residual block (Ext. Data Fig. 1) | `cascade.model.bottleneck.BottleneckBlock3d` |
| "Standard 3D convolution over the spacetime lattice" (surface codes) | `nn.Conv3d(kernel_size=3, padding=1)`, the default `message_passing` module inside `BottleneckBlock3d` |
| "Generalized convolution on the torus... weights indexed by relative offset" (BB codes, Eq. 2) | **not implemented yet** — see §4 |
| "Scattered to data qubits, pooled over support of each logical operator" | `cascade.model.readout.Readout` |
| "Prediction head... probability for each logical observable" | 2-layer MLP inside `Readout` |
| Full backbone | `cascade.model.surface_cascade.SurfaceCascade` |

### Why a shared bottleneck block works across code families

The paper's own Extended Data Fig. 3 makes the point explicit: every variant (bottleneck,
ConvNeXt, transformer) decomposes into two pointwise projections (H→H/b, H/b→H) plus one
*spatial* operation that varies by architecture/code family. `BottleneckBlock3d` is built
around exactly that seam: it accepts the spatial "message passing" step as an
already-constructed `nn.Module` (default: the surface code's 3x3x3 `Conv3d`). When BB
codes are added (Phase 3), only a new module gets passed in — the block itself, the
residual scaling, and the pre-activation BN/SiLU structure are unchanged.

### Extended Data Fig. 1, literally

```
h -> BN -> SiLU -> Conv1x1(H -> H/4)         [reduce]
  -> BN -> SiLU -> code-specific conv(H/4)   [message passing]
  -> BN -> SiLU -> Conv1x1(H/4 -> H)         [restore]
h_out = h + (1/sqrt(2*depth)) * (...)
```

`bias=False` on the two convs that are immediately followed by a `BatchNorm3d` (the bias
term would be redundant — BN's learned shift subsumes it); the final `restore` conv keeps
its bias since nothing normalizes its output afterward.

## 2. Data pipeline strategy

**Surface codes (Phase 2, not yet built):** stim ships a real, stable API for this —
`stim.Circuit.generated("surface_code_rotated_memory_z", distance=d, rounds=R,
after_clifford_depolarization=p, ...)` plus `circuit.get_detector_coordinates()`, which
returns `(x, y, t)`-style coordinates for every detector. That coordinate metadata is
exactly what should replace the placeholder masks in `geometry/surface_code.py`: bucket
detectors by `(x, y)` into the `(d+1, d+1)` grid, bucket by `t` into the round axis, and
derive `data_qubit_mask` / `logical_masks` from `circuit.get_final_qubit_coordinates()`
and the circuit's `OBSERVABLE_INCLUDE` qubit support. This is mechanical, not
research-risky — recommend building it next.

**BB codes (Phase 4, not yet built, and not just an engineering afterthought):** stim has
**no built-in generator** for bivariate bicycle codes. Building one requires:
1. Constructing `H_X = [A | B]`, `H_Z = [B^T | A^T]` over `Z_l × Z_m` from monomial
   exponent lists for `A`, `B` (straightforward — this part is pure math, testable via
   the CSS orthogonality check `H_X @ H_Z.T ≡ 0 (mod 2)`).
2. Deriving a circuit-level CNOT schedule that measures every check using its 6
   neighboring data qubits without ever touching the same qubit twice in one layer. A
   *generic* solution (greedy edge-coloring of the bipartite check–qubit interaction
   graph) is always correct in the sense of "produces a valid, deadlock-free circuit,"
   but is **not** guaranteed to match the original paper's specific schedule, which was
   chosen to avoid "hook errors" (a single circuit fault propagating to an
   effectively-lower-weight logical error) — that requires a careful choice of which
   monomial gets applied in which layer, not just any valid coloring. Treat a from-scratch
   schedule as a documented approximation and validate it (e.g., via `stim`'s
   `circuit.detector_error_model().shortest_graphlike_error()` to confirm the resulting
   circuit's effective distance actually matches `d`) before trusting decoder numbers
   trained on it.

Given the risk in point 2, recommend tackling Phase 4 as its own focused pass, likely
worth a literature check for existing open circuit generators (searches at implementation
time, not guessed here) before hand-deriving one.

## 3. Training recipe (Phase 2+, not yet built)

Summarized from Methods, for when a training loop is built:

- **Loss**: binary cross-entropy on the logical-observable logit(s); averaged across
  observables for `k > 1` (BB codes).
- **Data**: generated on-the-fly from stim (no fixed dataset) — an `IterableDataset`
  wrapping a compiled stim sampler is the natural fit, not a pre-materialized tensor file.
- **Curriculum**: train briefly at a low noise level, linearly anneal to the target level,
  then hold at the target level for the bulk of training — this avoids a "grokking"-like
  stall from training directly at high `p`. First two stages ≤2% of total steps.
- **Optimizer**: Muon (Newton-Schulz-orthogonalized updates) for matrix-valued conv/linear
  weights, Lion for scalar parameters (biases, norm params, embeddings, readout head).
  Both are drop-in optimizer classes — the main integration work is *parameter grouping*
  (route `weight` tensors of convs/linears to Muon, everything else to Lion), not the
  optimizers themselves. Reference implementations: Keller Jordan's public Muon writeup,
  and the Lion paper's reference PyTorch snippet.
- **MuP** (Maximal Update Parameterization): rescales init and per-layer LR as a function
  of width `H` so one set of hyperparameters works across `(H, L)` configs. This affects
  module *construction* (init std) and the optimizer's per-parameter-group LR, not the
  forward-pass code already built — implement as a thin wrapper that reads
  `(hidden_dim, depth)` off `SurfaceCascade`/its future BB counterpart.
- **EMA**: maintain an exponential moving average of weights (`decay=0.9998`); report
  metrics using EMA weights only.

## 4. Phased roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Surface-code backbone (embedding, bottleneck stack, readout) | **Done (this repo)** |
| 2 | Stim data pipeline + training loop for surface codes | Not started |
| 3 | BB-code geometry (monomial-derived Tanner edges) + bipartite/relational torus convolution, swapped into `BottleneckBlock3d`'s `message_passing` slot | Not started |
| 4 | BB-code stim circuit (parity checks → qubit layout → CNOT schedule → noise injection) | Not started — see risk note in §2 |
| 5 (optional) | FP8 post-training quantization, roofline latency estimates, dataflow/FPGA notes (Supplementary Information) | Not started |

## 5. Explicit caveats — where the paper under-specifies and what this repo assumes

- **Temporal handling at readout.** The paper says representations are "scattered to data
  qubits, pooled over the support of each logical operator" but doesn't say what happens
  to the time axis first. This repo collapses time via a *learned* convolution (kernel
  spanning the full round axis, no time padding) rather than a hardcoded time-average —
  more expressive, but it fixes the model to a specific `rounds` value at construction
  time (consistent with the paper's "separate model per (H, L, and implicitly R,d) config"
  methodology, so not considered a real limitation).
- **`depth` vs. code distance.** The paper says "L ~ d" so the receptive field spans the
  full code distance, but doesn't pin an exact proportionality constant. `SurfaceCascade`
  takes `depth` as an explicit required argument rather than silently deriving it from
  `distance`.
- **Data-qubit / ancilla grid geometry.** The paper states the syndrome tensor shape as
  `(R, d+1, d+1)` (checks only). At real distances there is no room for a disjoint
  data-qubit mask on that same small grid (`d^2` data qubits vs. `2d+2` leftover sites),
  so the literal "shared grid, disjoint boolean mask" picture used by the current
  placeholder geometry will not survive contact with real stim coordinates. This is
  already accounted for in the code: `Readout` converts any `(G,G)` boolean mask to flat
  indices (or, if needed later, a many-to-one incidence matrix) internally, so swapping in
  real geometry in Phase 2 requires no changes to `forward()`.
- **BB-code circuit correctness.** Flagged prominently in §2 — this is a real research
  risk, not a paperwork detail, and should be validated numerically before trusting any
  BB-code decoder trained on a from-scratch circuit.

## 6. What's actually in this repo right now

```
cascade/
  model/
    embedding.py         SyndromeEmbedding, syndrome_indices_from_detections()
    bottleneck.py          BottleneckBlock3d
    readout.py              Readout
    surface_cascade.py       SurfaceCascade
  geometry/
    surface_code.py            placeholder mask builders (NOT physically validated)
tests/test_shapes.py              shape + gradient-flow self-test (torch only, no stim)
examples/surface_cascade_smoketest.py   printed shape-trace walkthrough
```

Run once a real Python + torch environment is set up:

```
pip install -r requirements.txt
python -m pytest tests/test_shapes.py
python examples/surface_cascade_smoketest.py
```
