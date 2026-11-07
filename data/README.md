# Data

Derived tables behind the paper's core results. Every file is a (gzipped) CSV; row
counts and SHA-256 checksums are listed in `MANIFEST.tsv`, and
`reproduce/run_all.py` verifies them before running. All MCC values are base-pair
MCC against the design target; NUPACK values use the `canonical` configuration of
`accord/configs/nupack.json` unless a `config` column says otherwise.

## Targets

**`das_targets.csv`** (98 rows). The Das/PDB test targets of the gRNAde benchmark
split. `target_id` is the PDB chain identifier, `target_structure` the reference
secondary structure. `native_scored` marks the 95 single-chain targets whose native
sequences were scored; `in_crossed_57`, `in_strict_20` and `in_rhodesign_heldout`
mark the 57 common targets of the candidate-matched comparison, its 20-target strict
subset, and RhoDesign's held-out split without training overlap.

Eterna100 target structures are stored with each design, because the V1 and V2
versions of the benchmark differ for some puzzles.

## Designs

**`designs/cohorts_2d.csv.gz`** (19,114 designs, 17 cohorts). Every design of the
audit with its four oracle scores. Columns: `cohort`, `method`, `target_set`
(`Eterna100`, `Das`, `Das held-out 44`, `length-matched`), `paradigm`,
`leakage_status`, `target_id`, `slot` (generation order within target), `length`,
`sequence`, `target_structure`, `mcc_vienna`, `mcc_nupack`, `mcc_eternafold`,
`mcc_contrafold`, and `mcc_ribonanzanet_ss` (RibonanzaNet-SS, a sensitivity oracle;
empty for cohorts it was not run on). The cohorts use different target sets and
candidate counts, so comparisons across cohorts are descriptive.

| cohort | method | targets | designs |
|---|---|---|---|
| `rnainverse_eterna100` | RNAinverse | 71 | 3,550 |
| `nupack_eterna100` | NUPACK | 66 | 521 |
| `samfeo_eterna100` | SAMFEO | 71 | 710 |
| `samplingdesign_eterna100` | SamplingDesign | 69 | 412 |
| `learna_eterna100` | LEARNA | 66 | 1,658 |
| `learna_ed_eterna100` | LEARNA with an ensemble-defect reward | 66 | 1,980 |
| `metalearna_eterna100` | Meta-LEARNA | 71 | 659 |
| `eterna100_solutions` | Eterna player solutions | 99 | 376 |
| `random_baseline` | length-matched random sequences | 71 | 1,420 |
| `rnainverse_das` | RNAinverse | 98 | 980 |
| `nupack_das` | NUPACK | 97 | 970 |
| `grnade_das` | gRNAde (released checkpoint) | 44 | 2,200 |
| `grnade_retrain_das` | gRNAde retrained without sequence-cluster overlap | 98 | 1,568 |
| `ribodiffusion_das` | RiboDiffusion | 76 | 760 |
| `rdesign_das` | RDesign | 95 | 475 |
| `na_mpnn_das` | NA-MPNN | 49 | 245 |
| `rhodesign_heldout44` | RhoDesign on its held-out split | 44 | 630 |

`leakage_status` records what is known about training-set overlap with the Das
targets: `non-learned`, `meta-pretrained policy` (Meta-LEARNA), `clean retrain`,
`held-out subset`, `seq-cluster leak 44%` (the released gRNAde checkpoint), or
`cutoff undocumented`. `method`, `paradigm` and `leakage_status` use the same labels in
every design table.

**`designs/natives_das.csv.gz`** (95 rows). Native sequences of the Das targets with
their four oracle scores; the reference pool for native baselines.

**`designs/crossed_57.csv.gz`** (2,565 rows). Candidate-matched comparison: nine
methods x 57 common Das targets x 5 requested slots, each candidate selected only by
the method's own objective, with no post-selection by an evaluation oracle. Every slot
was generated and scored.

**`designs/strict_20.csv.gz`** (1,000 rows). Ten methods (the nine submitted methods,
gRNAde as the clean retrain, plus RDesign) x 20 targets x 5 slots.

**`designs/nupack_sweep.csv.gz`** (80,895 rows). The NUPACK MCC of every design under
five configurations (`config`: `canonical`, `rna99`, `wc_only`, `celsius_25`,
`mg_0p01`), for the 12 swept cohorts (`pool = cohort`), the 57-target pool
(`crossed_57`) and the natives (`natives_das`). `n_cooptimal` is the number of
co-optimal MFE structures from which the lexicographic tie-break chose.

**`designs/objective_swap_els.csv.gz`** (111 rows). Hamming local search with the
ensemble-defect objective (three replicates on 37 Eterna100 puzzles), with the final
ViennaRNA ensemble defect and three oracle scores.

**`designs/objective_swap_matched.csv.gz`** (1,332 rows). The 2 x 2 objective-by-mechanism
experiment (local search or PPO, MFE or ensemble-defect objective; 37 targets x 3
seeds) at wall-clock checkpoints of 60, 150 and 300 s: best sequence so far, objective
calls, ViennaRNA ensemble defect, MFE match and MCC.

## Chemical-mapping calibration

Derived per-sequence statistics only; no sequences, structures or reactivity
profiles. Disagreement columns starting with `d_pair` are the target-free pairing
disagreement (`accord.pairing_disagreement`: per-position standard deviation of
paired/unpaired calls across oracles, `ddof = 0`, averaged over positions);
`auc_<oracle>` is the area under the ROC curve separating SHAPE reactivities of
positions the oracle calls unpaired from those it calls paired.

**`calibration/ribonanza_shape.csv.gz`** (14,271 rows). The Ribonanza GPN15k
sub-library (2A3 SHAPE, signal-to-noise filter passed; 170 nt), keyed by the upstream
`seqID`. `cluster` groups sequences into 99 parent constructs or source genomes and is
the resampling unit. Columns: `d_pair3` (ViennaRNA, EternaFold, CONTRAfold),
`d_pair4` (plus NUPACK), `d_pair_no_<oracle>` (two-oracle disagreement for the
held-out-oracle control), `d_family3`, `d_family4` (family-balanced), per-oracle
`auc_*`, `mean_auc3`, `mean_auc4`, `dms_auc_vienna` (DMS, 5,000 sequences), and the
covariates `gc`, `paired_frac`, `mfe`, `ens_gap`, `ens_diversity`, `pos_entropy`. Two
sequences are missing from the four-oracle computation, so `d_pair4`, the
`d_pair_no_*` columns, `d_family3`, `d_family4`, `auc_nupack` and `mean_auc4` have
14,269 values.

**`calibration/openknot_shape.csv.gz`** (25,930 rows). OpenKnotAIDesignData designs
(2A3 SHAPE; 177 nt) from 13 design-method entries (`method`, the resampling unit),
keyed by the upstream `id`. Columns: `d_pair`, `d_family`, target MCC per oracle
(`mcc_vienna`, `mcc_eternafold`, `mcc_contrafold`, scored on the design window),
`q_mean`, `q_worst`, `q_best`, `d_mcc` (sample standard deviation of the target MCCs),
per-oracle `auc_*` and `mean_auc`.

## 3D structure scores

TM-scores from US-align (`-mol RNA`), normalised by the reference PDB chain. No
predicted structure files are included.

**`structure3d/native_panel.csv.gz`** (461 rows). Native sequences of the Das targets
predicted by RhoFold+, Boltz-2, RoseTTAFold2NA, AlphaFold3 (single sequence) and
DRfold2 (`predictor`), with `tm_score` and `rmsd`.

**`structure3d/matched_rhofold.csv.gz`** (276 rows). RhoFold+ predictions for up to
five designs per method from the released gRNAde checkpoint and RiboDiffusion on the
36 Das targets where both have designs (`method`, `slot`, `sequence`, `tm_score`,
`rmsd`).

**`structure3d/alphafold3.csv.gz`** (744 rows). AlphaFold3 3.0.1 predictions of
natives and one design per target from RNAinverse, NUPACK, RiboDiffusion and the
gRNAde retrain, in single-sequence mode (464 rows) and, on 57 targets, with the
searched multiple sequence alignment (280 rows; `mode`, `msa_depth`).

## Sources and licences

Designed sequences were generated for this study with the public releases of each
method and are released under this repository's MIT licence. Upstream material:
Eterna100 target structures and player solutions (eternagame/eterna100-benchmarking,
MIT); Das/PDB targets and native sequences from the gRNAde test split (PDB entries are
in the public domain; gRNAde, MIT). No raw Ribonanza or OpenKnot sequences or
reactivity profiles are redistributed; the calibration tables contain only derived
statistics keyed by upstream identifiers (Ribonanza: the Stanford Ribonanza RNA
Folding release; OpenKnot: eternagame/OpenKnotAIDesignData, Apache-2.0). Some
OpenKnot identifiers contain non-ASCII characters from Eterna design titles; they are
kept verbatim so that rows can be joined back to the source.
