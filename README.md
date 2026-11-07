# ACCORD

**A**cross-oracle **C**alibration for **CO**mputational **R**NA **D**esign.

Code and data for the NeurIPS 2026 paper *Multi-Oracle Agreement Reveals the Limits
of Self-Consistency Evaluation in RNA Design*
([OpenReview](https://openreview.net/forum?id=fkSfgYJc4x)).

RNA design methods are usually scored by refolding each designed sequence with a
structure predictor and comparing the prediction with the target. When the same
predictor was used during optimisation, a high score can reflect that predictor's
idiosyncrasies rather than a good design. ACCORD scores designs against a fixed
panel of secondary-structure predictors from two parameter families (ViennaRNA and
NUPACK; EternaFold and CONTRAfold), keeps failures in the denominator, reports
per-oracle quality next to the all-oracle pass rate, and flags cross-oracle
disagreement with uncertainty tiers calibrated retrospectively against SHAPE
chemical-mapping data.

The repository contains

- `accord/`, a Python package and command-line tool implementing the protocol
  (`docs/protocol.md`);
- `data/`, the derived tables behind the paper's core results (`data/README.md`);
- `reproduce/`, scripts that recompute those results from `data/` and check them
  against the values reported in the paper.

## Installation

```bash
git clone https://github.com/minghao-sun-sc/ACCORD.git
cd ACCORD
conda env create -f environment.yml          # Python 3.10, ViennaRNA 2.7.0, package in editable mode
conda activate accord
bash scripts/install_eternafold.sh           # EternaFold/CONTRAfold binary at a pinned commit
conda env config vars set ACCORD_ETERNAFOLD_HOME=$PWD/external/EternaFold
conda activate accord
accord oracles                               # shows which backends are available
```

- **NUPACK 4.0** requires a free academic licence. Download it from
  [nupack.org](https://www.nupack.org) and `pip install` the wheel into the
  environment. Without it, `nupack` slots are reported as incomplete; nothing is
  dropped silently.
- **EternaFold and CONTRAfold** are both run with the `contrafold` binary built from
  the EternaFold source tree (commit `13d2487`): with `EternaFoldParams.v1` for
  EternaFold and with the built-in defaults for CONTRAfold. The Bioconda
  `contrafold` 2.02 build returns different MEA structures for a sizeable fraction of
  sequences, so use the script above to match the paper exactly.
- The reproduction scripts need only the `accord` package and its dependencies
  (`numpy`, `pandas`, `scipy`); no oracle is run.
- Tested on Linux x86_64 with Python 3.10, numpy 2.2, pandas 2.3, scipy 1.15,
  ViennaRNA 2.7.0 and NUPACK 4.0.0. Building EternaFold needs `make`, `perl` and a
  C++ compiler (`environment.yml` provides them).

## Quick start

Input is a CSV with one row per requested output slot: `target_id`, `sequence`,
`target_structure` (dot-bracket), and optionally `design_id`. Leave `sequence`
empty for a slot whose generation failed.

```bash
accord evaluate examples/designs.csv --design-oracle vienna --out-dir accord_out
# without NUPACK installed, use the three-oracle panel:
accord evaluate examples/designs.csv --panel vienna eternafold contrafold --design-oracle vienna --out-dir accord_out
```

With the default four-oracle panel and no NUPACK, every slot is reported as
incomplete rather than scored on a smaller panel.

```python
import pandas as pd
import accord

designs = pd.read_csv("examples/designs.csv", dtype={"target_id": str})
report = accord.evaluate(designs, design_oracle="vienna")   # oracle used by the design objective
report.summary["views"]["leave_design_oracle_out"]["Y_all"]
report.per_design[["target_id", "mcc_vienna", "mcc_nupack", "mcc_eternafold", "mcc_contrafold", "tier"]]
report.save("accord_out")                                    # per_design.csv and report.json
```

If per-oracle MCC values were computed elsewhere, `accord summarize scores.csv
--out-dir accord_out` (or `accord.report_from_scores`) builds the same report from
`mcc_<oracle>` columns without running any oracle.

The report contains, for the common panel and, when a design oracle is declared,
for the panel without it: completion `C_complete`, any-oracle pass rate `C_any`,
all-oracle yield `Y_all` (the primary comparison scalar), `Q_mean`, `Q_worst`,
per-oracle mean MCC, raw and gated disagreement, outcome counts, and the tier
counts on the ViennaRNA + EternaFold + CONTRAfold panel. Rank designs by `Q_mean`
or `Q_worst`; treat disagreement and tiers as a robustness flag.

## Reproducing the paper

```bash
python reproduce/run_all.py
```

runs every analysis below from `data/` (a few minutes on one CPU), writes tables to
`reproduce/outputs/`, and compares each recomputed value with the value reported in
the paper (`reproduce/expected.json`).

| script | result |
|---|---|
| `single_oracle_failure.py` | RNAinverse outcome counts and oracle dependence under both denominators; native baselines; per-cohort end-to-end table |
| `objective_swap.py` | ensemble-defect versus MFE objective with the search fixed; 2 x 2 objective-by-mechanism comparison |
| `crossed_benchmark.py` | candidate-matched comparison on 57 common targets, with and without NUPACK in the panel |
| `strict_sensitivity.py` | all submitted methods on 20 shared targets |
| `nupack_sensitivity.py` | five NUPACK configurations |
| `panel_structure.py` | effective number of oracles (Hill's N2) and the PCA spectrum |
| `shape_calibration.py` | disagreement versus SHAPE accuracy on Ribonanza and OpenKnot |
| `structure_3d.py` | 3D predictor disagreement and the target-matched 3D comparison |

`reproduce/rescore_designs.py` closes the loop from sequence to score: it refolds
released designs with the installed oracles (a random sample by default, every design
with `--n 0`) and checks that each released MCC is reproduced exactly.

Not included: design generation with the upstream methods, 3D structure prediction
runs, and the functional-assay analyses (ribozyme, toehold and 5'UTR libraries). The
released tables contain the designed sequences and the derived scores those steps
produced.

## Scope

The tiers describe structural risk for designs with a target secondary structure,
calibrated retrospectively against SHAPE-derived pairing accuracy. They do not
predict biological function, are not calibrated for 3D structure, and have not
been validated prospectively. The four oracles behave as about two effective
families, so ACCORD detects disagreement between parameter families rather than
counting four independent votes. See `docs/protocol.md`.

## Citation

```bibtex
@inproceedings{sun2026accord,
  title     = {Multi-Oracle Agreement Reveals the Limits of Self-Consistency Evaluation in RNA Design},
  author    = {Sun, Lawrence Minghao and Cao, Hanqun and Wu, Fang and Zhang, Zhou and Liu, Zhiyuan and Fu, Tianfan and Heng, Pheng-Ann and Zhang, Yang},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=fkSfgYJc4x}
}
```

## License

MIT (see `LICENSE`). Upstream datasets and tools keep their own licences; see
`data/README.md`.
