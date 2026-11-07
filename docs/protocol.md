# The ACCORD reporting protocol

ACCORD evaluates designed RNA sequences for a target secondary structure against a
fixed panel of structure predictors ("oracles") and reports how much the verdict
depends on which oracle is asked. This page is the operational definition that the
`accord` package implements.

## Panel

| oracle | family | invocation |
|---|---|---|
| ViennaRNA 2.7.0 | physics (Turner 2004) | MFE, 37 C, Python bindings |
| NUPACK 4.0.0 | physics (Turner-type) | `nupack.mfe`, single strand, configuration `canonical` in `accord/configs/nupack.json`; co-optimal ties broken by the lexicographically smallest dot-bracket |
| EternaFold | learned (CONTRAfold engine) | `contrafold predict --params EternaFoldParams.v1` (MEA) |
| CONTRAfold | learned (CONTRAfold engine) | `contrafold predict` with the built-in default parameters (MEA) |

Both learned oracles use the `contrafold` binary of the EternaFold source tree at
commit 13d2487 (`scripts/install_eternafold.sh`); other CONTRAfold builds return
different MEA structures for a sizeable fraction of sequences.

The four oracles span about two effective parameter families (Hill's N2 between 1.4
and 2.3 depending on the pooled designs), so ACCORD is a detector of disagreement
between parameter families, not a vote among four independent judges.

Every score is the base-pair Matthews correlation coefficient (MCC) between the
predicted structure and the design target. A pair counts as correct only if both
partners match.

## Rules

1. **The requested output slot is the unit.** A method asked for K candidates per
   target contributes K slots per target. A slot whose generation failed, or for
   which any panel oracle failed, is *incomplete*: it stays in the denominator and
   counts as a failure in every pass rate. Multi-target summaries are computed per
   target and then averaged, so every target carries equal weight.

2. **Report the full block, never a composite.** The primary comparison scalar is
   the raw all-oracle yield

   `Y_all = P(slot complete and min_j MCC_j >= 0.5)`,

   always reported together with `C_complete` (completion), `C_any = P(slot
   complete and max_j MCC_j >= 0.5)`, `Q_mean` and `Q_worst` (mean over complete
   slots of the mean and minimum oracle MCC), the per-oracle mean MCC and the
   disagreement. No weighted combination of these quantities is formed.

3. **Disagreement is a robustness flag, not a quality score.** `D` is the sample
   standard deviation (`ddof = 1`) of `MCC_j` across the panel. `D_raw` summarises
   all complete slots; `D_cond` only slots that pass the gate. Rank candidates by
   `Q_mean` or `Q_worst`. A design with low `D` and mediocre `Q_mean` is a
   consistently mediocre design.

4. **The gate only removes shared failures.** The gate `max_j MCC_j >= 0.5` is
   fixed before analysis and applied identically to every method. Designs that fail
   it on every oracle get no uncertainty tier. The gate is not a ranking
   denominator. For tasks without a target structure ACCORD prescribes no default.

5. **Tiers belong to one panel.** The uncertainty tiers (`High` if `D < 0.05`,
   `Moderate` if `0.05 <= D <= 0.15`, `Low` if `D > 0.15`) are defined on the
   ViennaRNA + EternaFold + CONTRAfold panel and are applied to designs that pass
   the 0.5 gate on that panel (`accord/configs/tiers.json`). The gate stays at 0.5
   when a different pass threshold is chosen for `C_any` and `Y_all`. A different
   panel, oracle configuration or disagreement statistic inherits no cut-offs without
   recalibration.

6. **Fixed panel, plus the design oracle left out.** Direct comparisons use one
   fixed common panel for all methods. When a method uses an oracle inside its
   objective (for example NUPACK designs scored by NUPACK), the report adds a
   leave-design-oracle-out view; claims about self-evaluation or transfer must use
   that view. Per-oracle scores are always reported, and rankings never mix views.

7. **Reference ratios need shared targets.** The ratio of designed to reference
   disagreement (for example against native sequences) is formed only on shared
   targets scored with the same panel, configuration, post-processing and
   aggregation. A ratio above 2 is a heuristic prompt for inspection, not a test.
   Without a matched reference the two distributions are reported separately.

8. **State the scope.** The tiers were calibrated retrospectively against
   SHAPE-derived pairing accuracy (Ribonanza, replicated on OpenKnot). They do not
   predict biological function, are not calibrated for 3D structure, and have not
   been validated prospectively on newly assayed designs.

## Calibration statistic

The Ribonanza calibration cohort has no design target, so the tiers were fitted on
the target-free pairing disagreement (`accord.pairing_disagreement`: per-position
standard deviation of paired/unpaired calls across the three tier-panel oracles,
`ddof = 0`, averaged over positions). For designs ACCORD applies the cut-offs to
the target-MCC disagreement `D`. The transfer is supported by OpenKnot, where both
statistics exist: with the gate applied, `D` keeps the monotone High > Moderate >
Low ordering of SHAPE accuracy under the same cut-offs.

## Report fields

`accord evaluate` and `accord summarize` write `report.json` with one block per
view (`common_panel`, and `leave_design_oracle_out` when a design oracle is given):

| field | meaning |
|---|---|
| `panel` | oracles of the view |
| `weighting` | `target` (equal target weight) or `slot` |
| `n_slots`, `n_targets`, `n_incomplete` | denominators |
| `C_complete`, `C_any`, `Y_all` | coverage and yield over all slots |
| `Q_mean`, `Q_worst` | quality over complete slots |
| `mean_mcc_<oracle>` | per-oracle mean MCC over complete slots |
| `D_raw`, `D_cond` | median disagreement over complete / gated slots (with target weighting: the mean over targets of the per-target median) |
| `n_pass_all`, `n_pass_some_not_all`, `n_fail_all` | pooled outcome counts |
| `oracle_dependent_conditional`, `oracle_dependent_raw` | pass-some-not-all fraction among any-pass slots / among all slots |

plus `tiers` (counts per tier on the tier panel), `reference_ratio` when a
reference is given, and, for `accord evaluate`, the NUPACK configuration and the
backend versions under `meta`.
