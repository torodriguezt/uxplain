"""Data-loader checks use tiny local extracts, never remote survey downloads."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from uxplain import datasets
from uxplain.datasets import fetch_pnadc, load_geih


def _geih_rows():
    return pd.DataFrame({
        "DIRECTORIO": ["1", "1", "2"],
        "SECUENCIA_P": ["1", "1", "1"],
        "ORDEN": ["1", "2", "1"],
        "P3271": ["1", "2", "1"],
        "P6040": ["30", "40", "50"],
        "P3042": ["4", "5", "6"],
        "P6800": ["40", "35", "45"],
        "INGLABO": ["100", "200", "300"],
        "OFICIO_C8": ["1210", "5223", "9112"],
        "P6920": ["1", "2", "1"],
    })


def _write(frame, tmp_path, name="extract.csv"):
    path = tmp_path / name
    frame.to_csv(path, sep=";", index=False)
    return path


@pytest.mark.parametrize("target", ["income", "informal"])
def test_geih_alternative_targets_are_not_predictors(tmp_path, target):
    result = load_geih(_write(_geih_rows(), tmp_path), target=target)
    assert target not in result.feature_names
    assert result.data.shape == (3, 5)


def test_pnadc_informality_target_is_not_predictor(monkeypatch):
    frame = pd.DataFrame({
        "age": [30], "hours": [40], "education": [4], "female": [0.],
        "black_or_brown": [1.], "urban": [1.], "informal": [1.],
    })
    monkeypatch.setattr("uxplain.datasets._pnadc_frame", lambda *args: (frame, {}))
    result = fetch_pnadc(target="informal")
    assert "informal" not in result.feature_names
    assert result.target.tolist() == [1.]


def _pnadc_raw():
    return pd.DataFrame({
        "psu": ["110000016", "110000016", "110000017"],
        "dwelling": ["01", "01", "02"],
        "panel": ["07", "07", "07"],
        "urban_code": ["1", "2", "1"],
        "sex": ["2", "1", "1"],
        "age": ["30", "41", "52"],
        "race": ["4", "9", "1"],
        "education": ["7", "3", "8"],
        "employed": ["1", "1", "1"],
        "occupation": ["0110", "2421", " "],
        "pension_contrib": ["2", "1", None],
        "income": ["1500", "3200", "2000"],
        "hours": ["40", "44", "30"],
        "weight": ["100.5", "98.2", "120.0"],
    })


def test_pnadc_codes_become_indicators():
    frame = datasets._recode_pnadc(_pnadc_raw())
    np.testing.assert_array_equal(frame.female, [1., 0., 0.])
    # Race 9 ("ignored") is missing, not a category with a magnitude.
    np.testing.assert_array_equal(frame.black_or_brown, [1., np.nan, 0.])
    np.testing.assert_array_equal(frame.urban, [1., 0., 1.])
    np.testing.assert_array_equal(frame.informal, [1., 0., np.nan])
    # Education outside the documented 1-7 range is missing.
    np.testing.assert_array_equal(frame.education, [7., 3., np.nan])
    assert frame.occupation_group.tolist()[:2] == ["Armed forces", "Professionals"]
    assert pd.isna(frame.occupation_group.iloc[2])
    assert frame.household.tolist() == [
        "110000016-01-07", "110000016-01-07", "110000017-02-07",
    ]


def test_geih_codes_become_indicators(tmp_path):
    frame = _geih_rows()
    frame["P6920"] = ["1", "2", "3"]
    frame["P3042"] = ["4", "99", "13"]
    result = load_geih(_write(frame, tmp_path), features=["age", "female", "education"])
    np.testing.assert_array_equal(result.frame.female, [0., 0.])
    np.testing.assert_array_equal(result.frame.education, [4., 13.])
    # Workers who already receive a pension have no contribution status.
    assert result.frame.informal.isna().tolist() == [False, True]


def test_groups_and_provenance(tmp_path):
    path = _write(_geih_rows(), tmp_path)
    result = load_geih(path)
    assert result.groups.tolist() == ["1-1", "1-1", "2-1"]
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.provenance == {"files": {path.name: expected}}


def test_one_per_household_keeps_a_single_member(tmp_path):
    path = _write(_geih_rows(), tmp_path)
    result = load_geih(path, one_per_household=True, random_state=3)
    assert sorted(result.groups.tolist()) == ["1-1", "2-1"]
    again = load_geih(path, one_per_household=True, random_state=3)
    pd.testing.assert_frame_equal(result.frame, again.frame)


def test_one_per_household_requires_household_keys(tmp_path):
    frame = _geih_rows().drop(columns=["DIRECTORIO", "SECUENCIA_P", "ORDEN"])
    with pytest.raises(ValueError, match="household identifiers"):
        load_geih(_write(frame, tmp_path), one_per_household=True)


def _no_network(*args, **kwargs):
    raise AssertionError("network access attempted")


def test_pnadc_legacy_cache_is_accepted_when_checksum_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "_download", _no_network)
    monkeypatch.setattr(datasets, "_latest_quarter_archive", _no_network)
    legacy = tmp_path / "pnadc_2024q1.zip"
    legacy.write_bytes(b"quarter")
    digest = hashlib.sha256(b"quarter").hexdigest()
    path, release, found = datasets._pnadc_archive(
        2024, 1, tmp_path, "PNADC_012024_20250815.zip", digest,
    )
    assert (path, release, found) == (legacy, "PNADC_012024_20250815.zip", digest)


def test_pnadc_legacy_cache_without_release_name(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "_download", _no_network)
    monkeypatch.setattr(datasets, "_latest_quarter_archive", _no_network)
    legacy = tmp_path / "pnadc_2024q1.zip"
    legacy.write_bytes(b"quarter")
    path, release, _ = datasets._pnadc_archive(2024, 1, tmp_path)
    assert path == legacy and release is None


def test_pnadc_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "_download", _no_network)
    (tmp_path / "PNADC_012024_20250815.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        datasets._pnadc_archive(
            2024, 1, tmp_path, "PNADC_012024_20250815.zip", "0" * 64,
        )


def test_pnadc_archive_name_must_match_quarter(tmp_path):
    with pytest.raises(ValueError, match="archive must be"):
        datasets._pnadc_archive(2024, 1, tmp_path, "PNADC_022024_20250815.zip")


def test_explicit_target_leakage_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="cannot also be a feature"):
        load_geih(_write(_geih_rows(), tmp_path), target="income", features=["age", "income"])


def test_missing_occupation_is_not_armed_forces(tmp_path):
    frame = _geih_rows()
    frame["OFICIO_C8"] = [None, "invalid", "110"]
    result = load_geih(_write(frame, tmp_path))
    assert len(result.target) == 1
    assert result.target.tolist() == ["Armed forces"]
    assert result.frame.age.tolist() == [50.]


def test_missing_contribution_is_not_informal(tmp_path):
    frame = _geih_rows()
    frame.loc[0, "P6920"] = None
    result = load_geih(_write(frame, tmp_path))
    assert result.frame.age.tolist() == [40., 50.]
    assert result.data.informal.tolist() == [1., 0.]


def test_module_merge_uses_complete_person_key(tmp_path):
    frame = _geih_rows()
    keys = ["DIRECTORIO", "SECUENCIA_P", "ORDEN"]
    demographics = frame[keys + ["P3271", "P6040", "P3042"]].iloc[::-1]
    employed = frame.drop(columns=["P3271", "P6040", "P3042"])
    _write(demographics, tmp_path, "Caracteristicas generales.csv")
    _write(employed, tmp_path, "Ocupados.csv")
    result = load_geih(tmp_path)
    np.testing.assert_array_equal(result.data.age, [50, 40, 30])
    np.testing.assert_array_equal(result.data.income, [300, 200, 100])


def test_partial_person_keys_are_rejected(tmp_path):
    frame = _geih_rows()
    first = _write(frame.drop(columns="ORDEN"), tmp_path, "first.csv")
    second = _write(frame, tmp_path, "second.csv")
    with pytest.raises(KeyError, match="missing person keys"):
        load_geih([first, second])


def test_duplicate_person_keys_cannot_multiply_observations(tmp_path):
    frame = _geih_rows()
    first = _write(pd.concat([frame, frame.iloc[:1]]), tmp_path, "first.csv")
    second = _write(frame, tmp_path, "second.csv")
    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        load_geih([first, second])


@pytest.mark.parametrize("subsample", [0, -1, 1.5, True])
def test_invalid_subsample_is_rejected(tmp_path, subsample):
    with pytest.raises(ValueError, match="subsample"):
        load_geih(_write(_geih_rows(), tmp_path), subsample=subsample)


def test_seeded_subsample_is_reproducible(tmp_path):
    path = _write(_geih_rows(), tmp_path)
    first = load_geih(path, subsample=2, random_state=12)
    second = load_geih(path, subsample=2, random_state=12)
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_empty_module_list_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        load_geih([])
