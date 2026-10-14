"""
Real-world labour-force datasets used in the paper's illustrations.

Two national household surveys, loaded through a small scikit-learn-style
interface: each loader returns a `sklearn.utils.Bunch` (with ``data``,
``target``, ``frame``, ``feature_names``, ``groups``, ``provenance`` and
``DESCR``), or the ready-to-model ``(X, y)`` pair when called with
``return_X_y=True``::

    from uxplain.datasets import fetch_pnadc, load_geih

    X, y = fetch_pnadc(return_X_y=True)                 # Brazil, regression
    X, y = load_geih("data/geih", return_X_y=True)      # Colombia, classification

- **PNAD Contínua** (IBGE, Brazil) — conformal *regression* on monthly labour
  income. `fetch_pnadc` downloads the quarter from the IBGE public server
  and caches it, so nothing is shipped with the package.
- **GEIH** (DANE, Colombia) — conformal *classification* of occupational group.
  `load_geih` reads a copy the user has downloaded from DANE; see the note
  on data terms below.

Coding of the modelling columns
-------------------------------
Categorical survey codes are recoded to indicators before modelling, so that no
explainer treats a nominal code as a magnitude: ``female`` (1 = woman),
``black_or_brown`` (PNAD, 1 = *preta* or *parda*, the grouping IBGE uses to
report race inequality; code 9, "ignored", is missing), ``urban`` (PNAD,
1 = urban) and ``informal`` (1 = does not contribute to social security,
0 = contributes, missing otherwise; GEIH workers who already receive a pension
are missing). Education is the ordinal level attained (PNAD ``VD3004``, 1–7;
GEIH ``P3042``, 1–13), and ``hours`` are the usual weekly hours worked.

Households and sampling
-----------------------
Members of one household share unobserved characteristics, so a person-level
random split places correlated observations on both sides of a calibration
split. Each loader therefore returns household identifiers in ``groups``, and
``one_per_household=True`` keeps a single randomly chosen eligible person per
household. With that option the conformal exchangeability assumption concerns
households rather than household members. Neither option implements the
survey's sampling design (strata, primary sampling units, weights).

Reproducibility
---------------
Producers re-release microdata under new file names (IBGE date-stamps every
release). Pass ``archive`` and ``sha256`` to `fetch_pnadc` to pin one
release; both loaders report the files they read, with their SHA-256 digests, in
``provenance``.

Data terms
----------
No survey data is bundled with or redistributed by this package; the loaders
either fetch from the producer's own server or read a local copy the user
obtained themselves. When you use these datasets you must cite the producer:

- IBGE microdata are public and may be reproduced with attribution
  (see `PNADC_CITATION`).
- DANE authorises use, transformation and analysis of its anonymised microdata
  **provided the source is cited** (`GEIH_CITATION`); it does not
  authorise redistribution or commercial use, which is exactly why
  `load_geih` never downloads or ships the data. Obtain a month from
  https://microdatos.dane.gov.co and point the loader at it.

These helpers prepare predictive examples. They do not establish survey-design
validity or exchangeability for conformal coverage.
"""

from __future__ import annotations

import hashlib
import io
from numbers import Integral
import os
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

__all__ = [
    "GEIH_CITATION",
    "GEIH_FEATURES",
    "GEIH_VARIABLES",
    "MAJOR_OCCUPATION_GROUPS",
    "PNADC_CITATION",
    "PNADC_FEATURES",
    "PNADC_VARIABLES",
    "default_cache_dir",
    "fetch_pnadc",
    "load_geih",
]

_IBGE_ROOT = (
    "https://ftp.ibge.gov.br/Trabalho_e_Rendimento/"
    "Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados"
)
_IBGE_DICT = f"{_IBGE_ROOT}/Documentacao/Dicionario_e_input_20221031.zip"

PNADC_CITATION = (
    "Instituto Brasileiro de Geografia e Estatistica (IBGE), Pesquisa Nacional "
    "por Amostra de Domicilios Continua (PNAD Continua), microdata. "
    "Available from https://www.ibge.gov.br/."
)
GEIH_CITATION = (
    "Fuente: Departamento Administrativo Nacional de Estadistica (DANE), "
    "Gran Encuesta Integrada de Hogares (GEIH), bases anonimizadas: "
    "www.dane.gov.co."
)

#: PNAD Contínua source columns kept by `fetch_pnadc`, mapped to readable
#: names. ``UPA``, ``V1008`` and ``V1014`` identify the household within a
#: quarter.
PNADC_VARIABLES: dict[str, str] = {
    "UPA": "psu",                 # primary sampling unit
    "V1008": "dwelling",          # dwelling selection number within the PSU
    "V1014": "panel",             # rotation group
    "UF": "state",
    "V1022": "urban_code",        # 1 urban, 2 rural
    "V2007": "sex",               # 1 man, 2 woman
    "V2009": "age",
    "V2010": "race",              # 1 white, 2 black, 3 Asian, 4 brown, 5 Indigenous, 9 ignored
    "VD3004": "education",        # highest level attained, 1-7
    "VD4001": "in_labour_force",
    "VD4002": "employed",
    "VD4009": "job_position",     # employee w/ or w/o formal contract, etc.
    "V4010": "occupation",        # 4-digit COD occupation code (ISCO-08 based)
    "VD4012": "pension_contrib",  # 1 contributes to social security, 2 does not
    "VD4020": "income",           # effective monthly labour income, all jobs (target)
    "VD4031": "hours",            # usual weekly hours, all jobs
    "V1028": "weight",            # calibrated survey weight
}

#: GEIH source columns expected by `load_geih`, mapped to the same schema.
#:
#: These are the names used since the 2022 questionnaire redesign, which renamed
#: several fields (sex moved from ``P6020`` to ``P3271``, education from
#: ``P6210`` to ``P3042``, and the occupation code became ``OFICIO_C8``). Pass an
#: explicit mapping to `load_geih` for earlier years.
GEIH_VARIABLES: dict[str, str] = {
    "P3271": "sex",               # 1 man, 2 woman
    "P6040": "age",
    "P3042": "education",         # highest level attained, 1-13
    "P6800": "hours",             # usual weekly hours in the main job
    "INGLABO": "income",          # monthly labour income
    "OFICIO_C8": "occupation",    # 4-digit CIUO-08 A.C. occupation code
    "RAMA2D_R4": "industry",      # 2-digit industry code
    "P6920": "pension_contrib",   # 1 contributes, 2 does not, 3 already pensioned
    "FEX_C18": "weight",          # survey expansion factor
}

#: Default modelling features for each survey (the ones used in the paper).
PNADC_FEATURES: list[str] = [
    "age", "hours", "education", "female", "black_or_brown", "urban", "informal",
]
GEIH_FEATURES: list[str] = [
    "age", "hours", "education", "female", "income", "informal",
]

#: Person key that links GEIH modules to each other; the first two columns
#: identify the household.
GEIH_MERGE_KEYS: tuple[str, ...] = ("DIRECTORIO", "SECUENCIA_P", "ORDEN")

#: Major occupational groups: the leading digit of an ISCO-08-based code. GEIH
#: uses the Colombian adaptation (CIUO-08 A.C.) and PNAD Contínua the Brazilian
#: one (COD); both keep the ISCO-08 major groups. In PNAD, group 0 also contains
#: military police and military firefighters.
MAJOR_OCCUPATION_GROUPS: dict[str, str] = {
    "0": "Armed forces",
    "1": "Managers",
    "2": "Professionals",
    "3": "Technicians",
    "4": "Clerical support",
    "5": "Services and sales",
    "6": "Skilled agricultural",
    "7": "Craft and trades",
    "8": "Plant and machine operators",
    "9": "Elementary occupations",
}
#: Backwards-compatible alias.
CIUO_MAJOR_GROUPS = MAJOR_OCCUPATION_GROUPS


def default_cache_dir() -> Path:
    """Directory used to cache downloaded microdata (override with
    ``UXPLAIN_DATA_HOME``).

    Returns
    -------
    pathlib.Path
        Directory given by ``UXPLAIN_DATA_HOME`` when set, otherwise
        ``.cache/uxplain`` under the user's home directory.
    """
    root = os.environ.get("UXPLAIN_DATA_HOME")
    return Path(root) if root else Path.home() / ".cache" / "uxplain"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Shared recoding and packaging
# ---------------------------------------------------------------------------

def _codes(series: pd.Series) -> pd.Series:
    """Survey codes as stripped strings, with ``"2.0"`` read as ``"2"``."""
    codes = series.astype("string").str.strip()
    return codes.str.replace(r"\.0+$", "", regex=True)


def _indicator(series: pd.Series, true_codes, false_codes) -> pd.Series:
    """1.0 for ``true_codes``, 0.0 for ``false_codes``, missing otherwise."""
    codes = _codes(series)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[codes.isin(list(true_codes)).fillna(False).to_numpy(dtype=bool)] = 1.0
    out[codes.isin(list(false_codes)).fillna(False).to_numpy(dtype=bool)] = 0.0
    return out


def _informality(contribution: pd.Series, informal_codes=("2",)) -> pd.Series:
    """1 = does not contribute, 0 = contributes; other or missing codes stay missing."""
    return _indicator(contribution, informal_codes, ("1",))


def _ordinal(series: pd.Series, low: int, high: int) -> pd.Series:
    """Numeric ordinal codes inside ``[low, high]``; anything else is missing."""
    values = pd.to_numeric(_codes(series), errors="coerce").astype(float)
    return values.where(values.between(low, high))


def _major_group(occupation: pd.Series) -> pd.Series:
    """Map a 1-4 digit occupation code to its ISCO-08 major group name."""
    codes = _codes(occupation)
    codes = codes.where(codes.str.fullmatch(r"\d{1,4}", na=False))
    return codes.str.zfill(4).str[0].map(MAJOR_OCCUPATION_GROUPS)


def _finalize(frame, features, target, *, return_X_y, subsample, random_state,
              descr, citation, one_per_household=False, provenance=None):
    """Coerce the modelling columns, drop unusable rows, and package the result.

    Returns ``(X, y)`` when ``return_X_y`` is set, otherwise a
    `sklearn.utils.Bunch`.
    """
    features = list(features)
    if not features:
        raise ValueError("features must contain at least one predictor column.")
    if target in features:
        raise ValueError(f"The target column {target!r} cannot also be a feature.")
    if len(set(features)) != len(features):
        raise ValueError("features must not contain duplicate columns.")
    if subsample is not None and (
        isinstance(subsample, bool) or not isinstance(subsample, Integral) or subsample < 1
    ):
        raise ValueError("subsample must be a positive integer or None.")
    missing = [c for c in features + [target] if c not in frame.columns]
    if missing:
        raise KeyError(f"Requested columns not in the data: {missing}")

    # Features are coerced to float (not int): partial dependence and several
    # explainers reject integer columns, and float avoids implicit rounding.
    work = frame.copy()
    for column in features:
        if work[column].dtype == bool:
            work[column] = work[column].astype(float)
        else:
            work[column] = pd.to_numeric(work[column], errors="coerce").astype(float)

    subset = features + [target]
    work[subset] = work[subset].replace([float("inf"), float("-inf")], float("nan"))
    work = work.dropna(subset=subset)

    has_groups = "household" in work.columns and work["household"].notna().all()
    if one_per_household:
        if not has_groups:
            raise ValueError(
                "one_per_household=True needs household identifiers, which these "
                "files do not provide."
            )
        # One eligible person per household, chosen at random but reproducibly.
        work = work.groupby("household", sort=False).sample(
            n=1, random_state=random_state
        )
    if subsample is not None and subsample < len(work):
        work = work.sample(n=subsample, random_state=random_state)
    work = work.reset_index(drop=True)

    X = work[features]
    y = work[target].to_numpy()
    if return_X_y:
        return X, y
    return Bunch(
        data=X, target=y, frame=work, feature_names=features,
        target_name=target, DESCR=descr, citation=citation,
        groups=work["household"].to_numpy() if has_groups else None,
        provenance=dict(provenance or {}),
    )


# ---------------------------------------------------------------------------
# PNAD Contínua (IBGE, Brazil)
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    tmp.replace(dest)
    return dest


def _parse_sas_layout(text: str) -> dict[str, tuple[int, int]]:
    """Read the IBGE SAS input file into ``{name: (start, width)}``."""
    layout: dict[str, tuple[int, int]] = {}
    pattern = re.compile(r"^@(\d+)\s+(\w+)\s+\$?(\d+)\.")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            start, name, width = match.groups()
            layout[name] = (int(start), int(width))
    return layout


def _pnadc_layout(cache_dir: Path) -> tuple[dict[str, tuple[int, int]], Path]:
    archive = _download(_IBGE_DICT, cache_dir / "pnadc_dictionary.zip")
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".sas"))
        return _parse_sas_layout(zf.read(name).decode("latin-1")), archive


def _latest_quarter_archive(year: int, quarter: int) -> str:
    """Name of the newest IBGE release for a quarter, read from the server listing."""
    listing_url = f"{_IBGE_ROOT}/{year}/"
    with urllib.request.urlopen(listing_url) as response:
        listing = response.read().decode("utf-8", errors="replace")
    prefix = f"PNADC_{quarter:02d}{year}"
    names = re.findall(rf'href="({re.escape(prefix)}\w*\.zip)"', listing)
    if not names:
        raise FileNotFoundError(
            f"No PNAD Continua archive for {year} Q{quarter} at {listing_url}"
        )
    # The newest date-stamped file sorts last lexicographically.
    return max(names)


def _pnadc_archive(year, quarter, cache_dir: Path, archive=None, sha256=None):
    """Locate (or download) one quarter's archive and verify its checksum.

    Returns ``(path, release_name, digest)``. ``release_name`` is ``None`` when
    only a legacy cache file of unknown release is available.
    """
    prefix = f"PNADC_{quarter:02d}{year}"
    if archive is not None and not re.fullmatch(rf"{re.escape(prefix)}\w*\.zip", archive):
        raise ValueError(
            f"archive must be an IBGE file name for {year} Q{quarter}, "
            f"e.g. '{prefix}_YYYYMMDD.zip'; got {archive!r}."
        )
    expected = sha256.lower() if sha256 is not None else None
    legacy = cache_dir / f"pnadc_{year}q{quarter}.zip"

    if archive is not None:
        candidates = [(cache_dir / archive, archive)]
        # A legacy cache file can stand in for the requested release only when
        # its checksum proves that it is that release.
        if expected is not None:
            candidates.append((legacy, archive))
    else:
        cached = sorted(cache_dir.glob(f"{prefix}*.zip"), reverse=True)
        candidates = [(path, path.name) for path in cached] + [(legacy, None)]

    for path, release in candidates:
        if not (path.exists() and path.stat().st_size > 0):
            continue
        digest = _sha256(path)
        if expected is None or digest == expected:
            return path, release, digest
        if path != legacy:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected}, got {digest}. "
                "The producer may have re-released the file; remove it to download "
                "again, or update the pinned checksum."
            )

    release = archive or _latest_quarter_archive(year, quarter)
    path = _download(f"{_IBGE_ROOT}/{year}/{release}", cache_dir / release)
    digest = _sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(
            f"Checksum mismatch for the downloaded {release}: expected {expected}, "
            f"got {digest}."
        )
    return path, release, digest


def _recode_pnadc(frame: pd.DataFrame) -> pd.DataFrame:
    """Add modelling columns (indicators, ordinal codes, household key)."""
    frame = frame.copy()
    for column in ("age", "income", "hours", "weight"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["female"] = _indicator(frame["sex"], ("2",), ("1",))
    frame["black_or_brown"] = _indicator(frame["race"], ("2", "4"), ("1", "3", "5"))
    frame["urban"] = _indicator(frame["urban_code"], ("1",), ("2",))
    frame["informal"] = _informality(frame["pension_contrib"])
    frame["education"] = _ordinal(frame["education"], 1, 7)
    frame["occupation_group"] = _major_group(frame["occupation"])
    keys = [_codes(frame[c]) for c in ("psu", "dwelling", "panel")]
    frame["household"] = keys[0] + "-" + keys[1] + "-" + keys[2]
    return frame


def _pnadc_frame(year, quarter, cache_dir, employed_only, min_income,
                 archive=None, sha256=None):
    """Download and parse one quarter of PNAD Contínua into a tidy frame.

    Returns ``(frame, provenance)``.
    """
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    layout, dictionary = _pnadc_layout(cache_dir)
    absent = [v for v in PNADC_VARIABLES if v not in layout]
    if absent:
        raise KeyError(f"Variables absent from the IBGE layout: {absent}")

    # Cached archives are used without network access.
    path, release, digest = _pnadc_archive(year, quarter, cache_dir, archive, sha256)

    wanted = sorted(PNADC_VARIABLES, key=lambda v: layout[v][0])
    colspecs = [(layout[v][0] - 1, layout[v][0] - 1 + layout[v][1])
                for v in wanted]
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        with zf.open(name) as handle:
            frame = pd.read_fwf(
                io.TextIOWrapper(handle, encoding="latin-1"),
                colspecs=colspecs, names=wanted, dtype=str,
            )

    frame = _recode_pnadc(frame.rename(columns=PNADC_VARIABLES))
    if employed_only:
        employed = (_codes(frame["employed"]) == "1").fillna(False).to_numpy(dtype=bool)
        frame = frame[employed & (frame["income"] >= min_income).to_numpy()]
    provenance = {
        "archive": release,
        "archive_sha256": digest,
        "cached_file": path.name,
        "dictionary": _IBGE_DICT.rsplit("/", 1)[-1],
        "dictionary_sha256": _sha256(dictionary),
    }
    return frame.reset_index(drop=True), provenance


def fetch_pnadc(
    year: int = 2024,
    quarter: int = 1,
    *,
    features: list[str] | None = None,
    target: str = "income",
    return_X_y: bool = False,
    subsample: int | None = None,
    random_state: int | None = None,
    employed_only: bool = True,
    min_income: float = 1.0,
    one_per_household: bool = False,
    archive: str | None = None,
    sha256: str | None = None,
    cache_dir: str | Path | None = None,
):
    """
    Load one quarter of PNAD Contínua (IBGE, Brazil).

    The ~200 MB archive is downloaded from the IBGE public server on first use
    and cached, so later calls need no network. Nothing is shipped with the
    package. When you use this data, cite the source (`PNADC_CITATION`).

    Parameters
    ----------
    year, quarter
        Reference quarter, e.g. ``2024, 1``.
    features
        Modelling columns. Defaults to `PNADC_FEATURES`.
    target
        Response column. ``"income"`` (regression, default). Other useful
        choices: ``"informal"`` and ``"occupation_group"`` (classification).
    return_X_y
        If ``True``, return ``(X, y)`` instead of a
        `sklearn.utils.Bunch`.
    subsample
        If given, randomly keep this many rows (useful to keep an illustration
        fast; the conformal guarantee does not depend on the sample size).
    random_state
        Seed for ``one_per_household`` and ``subsample``.
    employed_only, min_income
        Restrict to employed respondents with a labour income at least
        ``min_income`` (zeros mean "no income", not low income).
    one_per_household
        Keep one randomly chosen eligible person per household (identified by
        ``UPA``, ``V1008`` and ``V1014``) before subsampling.
    archive
        Exact IBGE file name, e.g. ``"PNADC_012024_20250815.zip"``. IBGE
        re-releases quarters under new date stamps; pin one for reproducible
        results. By default the newest cached release is used, or else the
        newest release on the server.
    sha256
        Expected SHA-256 digest of the archive. A mismatch raises an error.
    cache_dir
        Download cache. Defaults to `default_cache_dir`.

    Returns
    -------
    sklearn.utils.Bunch or tuple
        A ``Bunch`` with ``data``, ``target``, ``frame`` (the parsed rows used),
        ``feature_names``, ``target_name``, ``groups`` (household identifiers),
        ``provenance`` (archive name and SHA-256 digests), ``DESCR`` and
        ``citation``; or ``(X, y)`` when ``return_X_y=True``.
    """
    if isinstance(quarter, bool) or not isinstance(quarter, Integral) or quarter not in (1, 2, 3, 4):
        raise ValueError("quarter must be an integer from 1 to 4.")
    frame, provenance = _pnadc_frame(
        year, quarter, cache_dir, employed_only, min_income, archive, sha256,
    )
    if features is None:
        features = [name for name in PNADC_FEATURES if name != target]
    descr = (f"PNAD Continua {year} Q{quarter} (IBGE). "
             f"Target: {target}. {PNADC_CITATION}")
    return _finalize(
        frame, features, target,
        return_X_y=return_X_y, subsample=subsample, random_state=random_state,
        descr=descr, citation=PNADC_CITATION,
        one_per_household=one_per_household, provenance=provenance,
    )


# ---------------------------------------------------------------------------
# GEIH (DANE, Colombia)
# ---------------------------------------------------------------------------

def _is_wanted_geih_module(name: str) -> bool:
    """True for the two person-level GEIH modules `load_geih` needs.

    ``"No ocupados"`` is disjoint from ``"Ocupados"`` (merging either in would
    empty the result), so the match on the employed module is exact.
    """
    stem = Path(name).stem.strip().lower()
    return stem == "ocupados" or stem.startswith("caracter")


def _read_geih_module(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str,
                       encoding="latin-1")


def _geih_frame(path, variables, min_income):
    """Merge the GEIH modules under ``path`` into one tidy person-level frame.

    Returns ``(frame, provenance)``.
    """
    variables = variables or GEIH_VARIABLES

    if isinstance(path, (str, Path)):
        path = Path(path)
        if path.is_dir():
            found = sorted(set(path.rglob("*.csv")) | set(path.rglob("*.CSV")))
            if not found:
                raise FileNotFoundError(f"No CSV files found under {path}")
            paths = [p for p in found if _is_wanted_geih_module(p.name)]
            if not paths:
                raise FileNotFoundError(
                    "Neither the 'Ocupados' nor the 'Caracteristicas generales' "
                    f"module was found under {path}. Files present: "
                    f"{[p.name for p in found]}"
                )
        else:
            paths = [path]
    else:
        paths = [Path(p) for p in path]

    if not paths:
        raise ValueError("Pass at least one GEIH CSV file.")

    frame = None
    for module in (_read_geih_module(p) for p in paths):
        keep = [c for c in module.columns
                if c in variables or c in GEIH_MERGE_KEYS]
        module = module[keep]
        if len(paths) > 1:
            missing = [key for key in GEIH_MERGE_KEYS if key not in module.columns]
            if missing:
                raise KeyError(f"Cannot merge GEIH modules: missing person keys {missing}.")
            if module[list(GEIH_MERGE_KEYS)].isna().any().any():
                raise ValueError("Cannot merge GEIH modules with missing person keys.")
        if frame is None:
            frame = module
            continue
        shared = list(GEIH_MERGE_KEYS)
        new_cols = [c for c in module.columns
                    if c not in frame.columns or c in shared]
        frame = frame.merge(module[new_cols], on=shared, how="inner", validate="one_to_one")

    household = None
    if all(key in frame.columns for key in GEIH_MERGE_KEYS[:2]):
        household = _codes(frame["DIRECTORIO"]) + "-" + _codes(frame["SECUENCIA_P"])

    present = {src: dst for src, dst in variables.items()
               if src in frame.columns}
    if "income" not in present.values():
        raise KeyError(
            f"No income column found. Columns present: "
            f"{sorted(frame.columns)[:25]}. Pass an explicit `variables` mapping."
        )
    frame = frame[list(present)].rename(columns=present)
    if household is not None:
        frame["household"] = household.to_numpy()

    for column in ("age", "income", "hours", "weight"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column].str.replace(",", ".", regex=False), errors="coerce",
            )
    if "sex" in frame.columns:
        frame["female"] = _indicator(frame["sex"], ("2",), ("1",))
    if "education" in frame.columns:
        frame["education"] = _ordinal(frame["education"], 1, 13)
    if "pension_contrib" in frame.columns:
        # Code 3 ("already pensioned") has no contribution status: missing.
        frame["informal"] = _informality(frame["pension_contrib"])
    if "occupation" in frame.columns:
        frame["occupation_group"] = _major_group(frame["occupation"])

    frame = frame[frame["income"].fillna(0) >= min_income]
    provenance = {"files": {p.name: _sha256(p) for p in paths}}
    return frame.reset_index(drop=True), provenance


def load_geih(
    path: str | Path | list[str | Path],
    *,
    features: list[str] | None = None,
    target: str = "occupation_group",
    return_X_y: bool = False,
    subsample: int | None = None,
    random_state: int | None = None,
    variables: dict[str, str] | None = None,
    min_income: float = 1.0,
    one_per_household: bool = False,
):
    """
    Load a locally downloaded GEIH extract (DANE, Colombia).

    This function never downloads anything: DANE authorises using and analysing
    its anonymised microdata with attribution but not redistributing it, so you
    obtain a month yourself and point ``path`` at it. Cite the source when you
    use the data (`GEIH_CITATION`).

    To get the data: open https://microdatos.dane.gov.co, choose *Gran Encuesta
    Integrada de Hogares (GEIH)* for a month, download and unzip it, and pass the
    folder that contains the CSV modules. A month ships one file per
    questionnaire module; this loader merges the *Ocupados* module (income,
    hours, occupation, pension contribution) with *Características generales,
    seguridad social en salud y educación* (sex, age, education) on the person
    key ``DIRECTORIO``/``SECUENCIA_P``/``ORDEN``.

    Parameters
    ----------
    path
        A directory holding the module CSVs (scanned recursively; only the two
        person-level modules are merged), a list of CSV paths, or a single CSV.
    features
        Modelling columns. Defaults to `GEIH_FEATURES`.
    target
        Response column. ``"occupation_group"`` (classification, default); the
        4-digit CIUO-08 A.C. occupation collapsed to its major group. Other
        choices: ``"income"`` (regression), ``"informal"`` (binary
        classification).
    return_X_y
        If ``True``, return ``(X, y)`` instead of a
        `sklearn.utils.Bunch`.
    subsample, random_state
        Optionally keep a random subset of ``subsample`` rows; ``random_state``
        also seeds ``one_per_household``.
    variables
        Override the source-column mapping (DANE renames fields between years).
        Defaults to `GEIH_VARIABLES`.
    min_income
        Drop incomes below this value.
    one_per_household
        Keep one randomly chosen eligible person per household (identified by
        ``DIRECTORIO`` and ``SECUENCIA_P``) before subsampling.

    Returns
    -------
    sklearn.utils.Bunch or tuple
        Same shape as `fetch_pnadc`; ``provenance`` lists the SHA-256
        digest of every module file read.
    """
    frame, provenance = _geih_frame(path, variables, min_income)
    if features is None:
        features = [name for name in GEIH_FEATURES if name != target]
    descr = (f"GEIH (DANE), local extract. Target: {target}. {GEIH_CITATION}")
    return _finalize(
        frame, features, target,
        return_X_y=return_X_y, subsample=subsample, random_state=random_state,
        descr=descr, citation=GEIH_CITATION,
        one_per_household=one_per_household, provenance=provenance,
    )
