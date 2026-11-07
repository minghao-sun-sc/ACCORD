"""Secondary-structure oracles with declared, frozen invocations.

Four oracles form the default panel. They span two parameter families:

* physics (Turner-type nearest-neighbour energies): ViennaRNA, NUPACK
* learned (CONTRAfold inference engine): EternaFold, CONTRAfold

ViennaRNA is called through its Python bindings and NUPACK through its Python
package. EternaFold and CONTRAfold are both run with the ``contrafold`` binary
built from the EternaFold source tree (``scripts/install_eternafold.sh``): with
``--params EternaFoldParams.v1`` for EternaFold and with the built-in default
parameters for CONTRAfold. Other CONTRAfold builds give different MEA structures
for a sizeable fraction of sequences, so the reported numbers are tied to this
build. Locations are resolved in this order:

``ACCORD_ETERNAFOLD_HOME``
    EternaFold source tree; uses ``src/contrafold`` and
    ``parameters/EternaFoldParams.v1``.
``ACCORD_CONTRAFOLD``, ``ACCORD_ETERNAFOLD``, ``ACCORD_ETERNAFOLD_PARAMS``
    Explicit binary or parameter paths (override the above).
``PATH``
    A ``contrafold`` executable on ``PATH``, used for both oracles.

A missing backend never shrinks the panel silently: the affected oracle returns
a failed :class:`FoldResult`, and the design slot is counted as incomplete.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Optional

from .structure import dotbracket_to_pairs

#: Oracles of the default four-oracle panel, in reporting order.
PANEL_4: tuple[str, ...] = ("vienna", "nupack", "eternafold", "contrafold")
#: Panel on which the uncertainty tiers were calibrated.
TIER_PANEL: tuple[str, ...] = ("vienna", "eternafold", "contrafold")
#: Parameter family of every supported oracle.
FAMILIES: dict[str, str] = {
    "vienna": "physics",
    "nupack": "physics",
    "eternafold": "learned",
    "contrafold": "learned",
}
SUPPORTED: tuple[str, ...] = tuple(FAMILIES)

_CANONICAL_PAIRS = {("A", "U"), ("U", "A"), ("C", "G"), ("G", "C"), ("G", "U"), ("U", "G")}


@dataclass
class FoldResult:
    """Prediction returned by one oracle for one sequence."""

    oracle: str
    sequence: str
    structure: Optional[str]
    ok: bool = True
    error: str = ""
    energy: Optional[float] = None
    n_cooptimal: int = 1
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NupackConfig:
    """A fully declared NUPACK invocation.

    Attributes:
        name: Label used in reports.
        material: NUPACK parameter set (``rna`` or ``rna99``).
        celsius: Temperature in degrees Celsius.
        sodium: Sodium concentration (M).
        magnesium: Magnesium concentration (M).
        ensemble: NUPACK ensemble type.
        wc_only_filter: Strip pairs that are not Watson-Crick or GU from both the
            prediction and the target before scoring.
        primary: True only for the canonical configuration.
        notes: Free-text description.
    """

    name: str
    material: str = "rna"
    celsius: float = 37.0
    sodium: float = 1.0
    magnesium: float = 0.0
    ensemble: str = "stacking"
    wc_only_filter: bool = False
    primary: bool = False
    notes: str = ""

    def as_dict(self) -> dict:
        """Return the configuration as a plain dictionary."""
        return asdict(self)


@lru_cache(maxsize=None)
def nupack_configs() -> dict[str, NupackConfig]:
    """Load the canonical NUPACK configuration and its sensitivity arms.

    Returns:
        Mapping from configuration name to :class:`NupackConfig`; the entry
        ``"canonical"`` is the only primary configuration.
    """
    text = resources.files("accord").joinpath("configs/nupack.json").read_text()
    raw = json.loads(text)
    keys = set(NupackConfig.__dataclass_fields__)
    return {name: NupackConfig(name=name, **{k: v for k, v in cfg.items() if k in keys})
            for name, cfg in raw["configurations"].items()}


def normalise_sequence(seq: str) -> str:
    """Upper-case a sequence and map T to U.

    Args:
        seq: Raw nucleotide sequence.

    Returns:
        RNA sequence string.
    """
    return str(seq).strip().upper().replace("T", "U")


def is_rna(seq: str) -> bool:
    """Return True when ``seq`` is non-empty and uses only A, C, G and U.

    Args:
        seq: Normalised sequence.

    Returns:
        Whether the sequence is a valid canonical RNA string.
    """
    return bool(seq) and set(seq) <= {"A", "C", "G", "U"}


def canonical_pairs_only(db: str, seq: str) -> str:
    """Remove pairs that are not Watson-Crick or GU.

    Args:
        db: Dot-bracket structure.
        seq: Matching RNA sequence.

    Returns:
        Structure with non-canonical pairs replaced by dots.
    """
    out = list(db)
    for i, j in dotbracket_to_pairs(db):
        if (seq[i], seq[j]) not in _CANONICAL_PAIRS:
            out[i] = out[j] = "."
    return "".join(out)


# --------------------------------------------------------------------------
# Backend discovery
# --------------------------------------------------------------------------

_INSTALL_HINT = "run scripts/install_eternafold.sh and set ACCORD_ETERNAFOLD_HOME"
_PATH_NOTE = ("found on PATH; reproduces the paper only if built from EternaFold commit "
              "13d2487 (scripts/install_eternafold.sh)")


def _env_path(var: str) -> tuple[Optional[Path], str]:
    """Path named by an environment variable, and a problem message if it is wrong."""
    value = os.environ.get(var)
    if not value:
        return None, ""
    p = Path(value).expanduser()
    return (p, "") if p.exists() else (None, f"{var}={value} does not exist")


def _locate_binary(var: str) -> tuple[Optional[Path], str, str]:
    """Locate a CONTRAfold-engine binary: (path, source, problem)."""
    path, problem = _env_path(var)
    if path is not None or problem:
        return path, var, problem
    home, problem = _env_path("ACCORD_ETERNAFOLD_HOME")
    if problem:
        return None, "ACCORD_ETERNAFOLD_HOME", problem
    if home is not None:
        binary = home / "src" / "contrafold"
        if binary.exists():
            return binary, "ACCORD_ETERNAFOLD_HOME", ""
        return None, "ACCORD_ETERNAFOLD_HOME", f"{binary} does not exist; {_INSTALL_HINT}"
    found = shutil.which("contrafold")
    if found:
        return Path(found), "PATH", ""
    return None, "", f"no contrafold binary; {_INSTALL_HINT}"


def _locate_params() -> tuple[Optional[Path], str, str]:
    """Locate ``EternaFoldParams.v1``: (path, source, problem)."""
    path, problem = _env_path("ACCORD_ETERNAFOLD_PARAMS")
    if path is not None or problem:
        return path, "ACCORD_ETERNAFOLD_PARAMS", problem
    home, problem = _env_path("ACCORD_ETERNAFOLD_HOME")
    if problem:
        return None, "ACCORD_ETERNAFOLD_HOME", problem
    if home is not None:
        params = home / "parameters" / "EternaFoldParams.v1"
        if params.exists():
            return params, "ACCORD_ETERNAFOLD_HOME", ""
        return None, "ACCORD_ETERNAFOLD_HOME", f"{params} does not exist"
    return None, "", f"EternaFoldParams.v1 not found; {_INSTALL_HINT}"


def contrafold_binary() -> Optional[Path]:
    """Binary used for the ``contrafold`` oracle (run with default parameters)."""
    return _locate_binary("ACCORD_CONTRAFOLD")[0]


def eternafold_binary() -> Optional[Path]:
    """Binary used for the ``eternafold`` oracle (run with EternaFold parameters)."""
    return _locate_binary("ACCORD_ETERNAFOLD")[0]


def eternafold_params() -> Optional[Path]:
    """Location of ``EternaFoldParams.v1``."""
    return _locate_params()[0]


def _import_vienna():
    import RNA  # noqa: N811 - module name fixed by ViennaRNA

    return RNA


def _import_nupack():
    import nupack

    return nupack


def availability() -> dict[str, dict]:
    """Report whether every supported oracle can actually run.

    Unlike a check for a wrapper function, this imports each Python backend and
    resolves each binary, so a missing installation is reported here instead of
    surfacing later as a smaller panel.

    Returns:
        Mapping oracle -> ``{"available": bool, "detail": str}``.
    """
    out: dict[str, dict] = {}
    try:
        rna = _import_vienna()
        out["vienna"] = {"available": True, "detail": f"ViennaRNA {getattr(rna, '__version__', '?')}"}
    except ImportError as exc:
        out["vienna"] = {"available": False, "detail": f"cannot import RNA: {exc}"}
    try:
        npk = _import_nupack()
        out["nupack"] = {"available": True, "detail": f"NUPACK {getattr(npk, '__version__', '?')}"}
    except ImportError as exc:
        out["nupack"] = {"available": False, "detail": f"cannot import nupack: {exc}"}
    for name, var in (("contrafold", "ACCORD_CONTRAFOLD"), ("eternafold", "ACCORD_ETERNAFOLD")):
        binary, source, problem = _locate_binary(var)
        if binary is None:
            out[name] = {"available": False, "detail": problem}
            continue
        detail = f"{binary} (via {source})"
        if name == "eternafold":
            params, _, params_problem = _locate_params()
            if params is None:
                out[name] = {"available": False, "detail": params_problem}
                continue
            detail = f"{binary} --params {params} (via {source})"
        if source == "PATH":
            detail += "; " + _PATH_NOTE
        out[name] = {"available": True, "detail": detail}
    return out


# --------------------------------------------------------------------------
# Individual oracles
# --------------------------------------------------------------------------

def fold_vienna(seq: str, celsius: float = 37.0) -> FoldResult:
    """Minimum-free-energy structure from ViennaRNA (Turner 2004 parameters).

    Args:
        seq: RNA sequence.
        celsius: Temperature in degrees Celsius.

    Returns:
        :class:`FoldResult`.
    """
    seq = normalise_sequence(seq)
    try:
        rna = _import_vienna()
        md = rna.md()
        md.temperature = celsius
        structure, energy = rna.fold_compound(seq, md).mfe()
        return FoldResult("vienna", seq, structure, energy=float(energy))
    except Exception as exc:
        return FoldResult("vienna", seq, None, ok=False, error=repr(exc))


def _parse_contrafold(stdout: str) -> str:
    """Extract the dot-bracket line from ``contrafold predict`` output."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line and not line.startswith(">") and set(line) <= set(".()[]{}<>"):
            return line
    raise ValueError(f"no structure line in output: {stdout[:200]!r}")


def _run_contrafold(seq: str, binary: Optional[Path], params: Optional[Path],
                    oracle: str, timeout: int = 600) -> FoldResult:
    """Run a CONTRAfold-engine binary on one sequence (MEA prediction)."""
    seq = normalise_sequence(seq)
    if binary is None:
        return FoldResult(oracle, seq, None, ok=False, error="binary not found")
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".fasta", prefix="accord_",
                                         delete=False) as fh:
            fh.write(f">seq\n{seq}\n")
            path = fh.name
        cmd = [str(binary), "predict", path]
        if params is not None:
            cmd += ["--params", str(params)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return FoldResult(oracle, seq, None, ok=False,
                              error=f"exit {proc.returncode}: {proc.stderr[:200]}")
        return FoldResult(oracle, seq, _parse_contrafold(proc.stdout))
    except Exception as exc:
        return FoldResult(oracle, seq, None, ok=False, error=repr(exc))
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def fold_eternafold(seq: str) -> FoldResult:
    """MEA structure from EternaFold (CONTRAfold engine, EternaFold parameters).

    Args:
        seq: RNA sequence.

    Returns:
        :class:`FoldResult`; failed when the binary or parameter file is missing.
    """
    binary, _, problem = _locate_binary("ACCORD_ETERNAFOLD")
    params, _, params_problem = _locate_params()
    if binary is None or params is None:
        return FoldResult("eternafold", normalise_sequence(seq), None, ok=False,
                          error=problem or params_problem)
    return _run_contrafold(seq, binary, params, "eternafold")


def fold_contrafold(seq: str) -> FoldResult:
    """MEA structure from CONTRAfold with its built-in default parameters.

    The paper's values use the ``contrafold`` binary of the EternaFold source tree
    (commit 13d2487) without ``--params``.

    Args:
        seq: RNA sequence.

    Returns:
        :class:`FoldResult`.
    """
    binary, _, problem = _locate_binary("ACCORD_CONTRAFOLD")
    if binary is None:
        return FoldResult("contrafold", normalise_sequence(seq), None, ok=False, error=problem)
    return _run_contrafold(seq, binary, None, "contrafold")


_NUPACK_MODELS: dict[tuple, object] = {}


def fold_nupack(seq: str, config: str | NupackConfig = "canonical") -> FoldResult:
    """MFE structure from NUPACK under a declared configuration.

    The sequence is folded as a single strand. When several structures share the
    minimum free energy, the lexicographically smallest dot-bracket string is
    returned, so the result does not depend on the order NUPACK reports ties.

    Args:
        seq: RNA sequence.
        config: Configuration name from ``configs/nupack.json`` or an explicit
            :class:`NupackConfig`.

    Returns:
        :class:`FoldResult` with ``meta["nupack_config"]`` set.
    """
    seq = normalise_sequence(seq)
    if isinstance(config, str):
        if config not in nupack_configs():
            raise ValueError(f"unknown NUPACK configuration {config!r}; "
                             f"available: {', '.join(nupack_configs())}")
        cfg = nupack_configs()[config]
    else:
        cfg = config
    try:
        npk = _import_nupack()
    except ImportError as exc:
        return FoldResult("nupack", seq, None, ok=False, error=f"nupack not importable: {exc}",
                          meta={"nupack_config": cfg.name})
    try:
        key = (cfg.material, cfg.celsius, cfg.sodium, cfg.magnesium, cfg.ensemble)
        if key not in _NUPACK_MODELS:
            _NUPACK_MODELS[key] = npk.Model(material=cfg.material, celsius=cfg.celsius,
                                            sodium=cfg.sodium, magnesium=cfg.magnesium,
                                            ensemble=cfg.ensemble)
        result = npk.mfe(strands=[seq], model=_NUPACK_MODELS[key])
        if not result:
            return FoldResult("nupack", seq, None, ok=False, error="empty MFE result",
                              meta={"nupack_config": cfg.name})
        energies = [float(r.energy) for r in result]
        best = min(energies)
        tied = sorted(str(r.structure) for r, e in zip(result, energies) if abs(e - best) < 1e-9)
        structure = tied[0]
        if cfg.wc_only_filter:
            structure = canonical_pairs_only(structure, seq)
        return FoldResult("nupack", seq, structure, energy=best, n_cooptimal=len(tied),
                          meta={"nupack_config": cfg.name})
    except Exception as exc:
        return FoldResult("nupack", seq, None, ok=False, error=repr(exc),
                          meta={"nupack_config": cfg.name})


def fold(seq: str, oracle: str, nupack_config: str | NupackConfig = "canonical") -> FoldResult:
    """Fold one sequence with one oracle.

    Args:
        seq: RNA sequence.
        oracle: One of :data:`SUPPORTED`.
        nupack_config: NUPACK configuration, used only for ``oracle="nupack"``.

    Returns:
        :class:`FoldResult`.

    Raises:
        ValueError: For an unknown oracle name.
    """
    name = oracle.lower()
    if name == "vienna":
        return fold_vienna(seq)
    if name == "nupack":
        return fold_nupack(seq, nupack_config)
    if name == "eternafold":
        return fold_eternafold(seq)
    if name == "contrafold":
        return fold_contrafold(seq)
    raise ValueError(f"unknown oracle {oracle!r}; supported: {', '.join(SUPPORTED)}")


def environment() -> dict:
    """Versions and locations of every backend, for inclusion in reports."""
    info = {"python": sys.version.split()[0]}
    for name, status in availability().items():
        info[name] = status["detail"] if status["available"] else f"unavailable ({status['detail']})"
    return info
