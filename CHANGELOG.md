# Changelog

## 2.0.0

First public release, accompanying the NeurIPS 2026 camera-ready paper.

- Reporting protocol: requested output slots as the unit with failures kept in the
  denominator; `Y_all` as the primary scalar, reported with `C_complete`, `C_any`,
  `Q_mean`, `Q_worst`, per-oracle scores and disagreement; equal target weighting;
  a leave-design-oracle-out view whenever a design oracle is declared; a reference
  disagreement ratio formed only on shared targets.
- Uncertainty tiers defined on the ViennaRNA + EternaFold + CONTRAfold panel and
  applied only to designs that pass the 0.5 gate.
- An unavailable oracle marks the affected slots incomplete; the panel is never
  shrunk silently, and `accord oracles` reports where each backend was found or why
  it is missing.
- Input validation: malformed target structures, missing target identifiers and
  repeated panel entries are rejected; existing score columns are recomputed rather
  than duplicated.
- Frozen NUPACK invocation (`accord/configs/nupack.json`) with a deterministic
  tie-break among co-optimal MFE structures and four named sensitivity
  configurations.
- All scores use the base-pair MCC.
- Derived data tables and scripts that reproduce the paper's core results.
