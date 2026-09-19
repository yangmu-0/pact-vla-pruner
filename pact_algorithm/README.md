# PACT-VLA controller

The PACT visual-token budget controller, as an **overlay** on the upstream
VLA-Pruner checkout. Only source is stored here — no model weights and no
conda environments.

## Upstream base

| | |
|---|---|
| Repository | `https://github.com/MINT-SJTU/VLA-Pruner.git` |
| Base commit | `a5d4cbd` — "feat: add PACT-VLA pruning to OpenVLA-OFT" |
| Controller head | `23d57fa` — "Add PACT ablation variants and latter-half attention aggregation" |
| Diff | 8 files changed, **+701 / −313** |

Local commits that make up this overlay, oldest first:

- `d9f9066` Rewrite PACT-VLA budget controller and vectorise its token selection
- `be61a2a` Set PACT theta0 to 0.4 and unify its default across the stack
- `3dc8de7` Adapt the OpenVLA (non-OFT) backend for the unified LIBERO evaluation
- `65dcb2f` Add the DivPrune LIBERO adaptation used by the unified evaluation
- `23d57fa` Add PACT ablation variants and latter-half attention aggregation

## Layout and how to apply

`files/` mirrors the upstream directory structure. To reproduce:

```bash
git clone https://github.com/MINT-SJTU/VLA-Pruner.git
cd VLA-Pruner && git checkout a5d4cbd
cp -r /path/to/pact_algorithm/files/. .
```

`upstream.json` records the base/head commits, which files are PACT and which
are not, and a SHA-256 for every file so the overlay can be verified.

## What the controller implements

`pact_vla.py` is the whole algorithm; the other files wire it into the model and
the evaluator.

| Function | Method |
|---|---|
| `normalize_distribution` | Eq. 4 — normalised attribution |
| `build_action_prior` | Eq. 5 — causal, time-decayed action prior |
| `normalized_jsd` | Eqs. 6–8 — normalised Jensen–Shannon divergence |
| `compute_theta` | Eq. 9 — conflict-dependent coverage threshold |
| `build_candidate_budgets` | Eq. 10 — hardware-supported candidate budgets |
| `top_b_union` | Eq. 11 — both branches nominate their top-B tokens |
| `select_top_priority` | Eqs. 12–13 — keep the B highest joint-priority nominees |
| `compute_dual_coverage` | Eqs. 14–16 — separate perception/action coverage |
| `search_minimum_budget` | Eq. 17 — smallest budget meeting dual coverage |
| `PACTController.select` | Orchestrates Eqs. 4–17; falls back to all N tokens on any invalid input |

`select_top_priority` is a **single vectorised `topk`** over the nomination union.
An earlier implementation performed greedy max-min cosine-diversity selection in
an O(B) Python loop; it cost 10.7–19.6 ms per filter and 11–71 ms per policy call
because each of the ~150 µs iterations launched several tiny GPU kernels and the
loop is inherently serial. The vectorised form costs 0.13 ms per filter and makes
the same budget decisions, so latency at matched retention now matches the
original inexpensive controller.

### Ablation variants

`PACTController(variant=...)` selects one of four configurations, so the
components can be ablated independently:

| variant | action prior | conflict / threshold |
|---|---|---|
| `full` (default) | full history, gamma-decayed | normalised JSD drives `theta` |
| `no-conflict` | full history | conflict skipped; base `theta0` only |
| `perception-only` | prior = perception (cold start always) | conflict forced to 0 |
| `last-action-prior` | only the most recent action attribution | normalised JSD drives `theta` |

The variant is part of `signature`, so changing it rebuilds the controller. It is
plumbed through `pact_variant` in the backend config and validated there.

### Attention aggregation

`modeling_llama.py` aggregates the PACT perception scores over the **latter half
of the decoder layers** (`_reduce_pact_action_attention`) rather than reading a
single layer's attention.

### Tracing

Setting `PACT_TRACE_FOCUS=1` adds, for each decision, the perception focus index
set, the action focus index set and their overlap. Decision stats also report the
variant, the effective history size and the available history size.

### Integration points

- `modeling_prismatic.py` — packs the controller settings into `fastv_config`
  and keeps the action-attention history `q_t` causal: a query only becomes
  visible to PACT after its action chunk has been executed.
- `openvla_utils.py` — pushes the `pact_*` settings onto the loaded model.
- `run_libero_eval.py` — the `GenerateConfig` fields, validation, logging, and
  per-episode visual-token / FLOP-ratio / retention-bucket accounting.
- `test_pact_vla.py` — unit tests over the controller API.

## Current parameters

```
pact_variant      = full
pact_gamma        = 0.8
pact_theta0       = 0.4
pact_alpha_d      = 0.10
pact_theta_min    = 0.15
pact_theta_max    = 0.7
pact_budget_rates = 0.125,0.25,0.5,0.75,1.0
```

The threshold is dynamic: `theta_t = clip(theta0 + alpha_d * conflict, theta_min, theta_max)`,
i.e. **0.4–0.5** at the current settings. Note that `theta_min = 0.15` and
`theta_max = 0.7` are therefore inert; raising `alpha_d` would let conflict
actually modulate the threshold.

## Verified behaviour

LIBERO, OpenVLA-OFT, layer 15, seed 7, 3 episodes per task (30 per suite):

| suite | θ₀ | success | mean retention | FLOPs/call |
|---|---|---|---|---|
| spatial | 0.3 | 30/30 | 23.1% | 1.610 T |
| object | 0.3 | 26/30 | 23.2% | 1.577 T |
| object | **0.4** | **28/30** | **36.9%** | 1.999 T |
| goal | 0.3 | 29/30 | 23.3% | 1.562 T |
| long | 0.3 | 29/30 | 22.1% | 1.565 T |

Vanilla OFT on the same conditions is 3.94–4.00 T per call, so the controller
removes ~60% of the first-LLM-pass FLOPs at θ₀=0.3 and ~50% at θ₀=0.4. These
figures predate the ablation variants and the latter-half aggregation.

## Files that are not PACT

Two files in the overlay are shared with other work in the same working tree and
are kept so the overlay applies cleanly:

- `prismatic/vla/constants.py` — `PIPER_CONSTANTS` and the
  `OPENVLA_ROBOT_PLATFORM` override for the AGX Piper bimanual platform.
- `experiments/robot/libero/libero_utils.py` — `save_rollout_video` takes an
  `fps` argument.

## Not included

- **Model weights** — upstream keeps `src/openvla/checkpoints` and
  `src/openvla-oft/checkpoints` untracked (~117 GB). Deliberately excluded.
- **Conda environments** — never inside the repository.
- **Other baselines** — FastV, SparseVLM, VLA-Cache and DivPrune adaptations live
  in the respective upstream forks, not here.
