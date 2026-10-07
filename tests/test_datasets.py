"""Data-loader checks use tiny local extracts, never remote survey downloads."""

import numpy as np
import pandas as pd
import pytest

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
        "age": [30], "hours": [40], "education": [4], "sex": [1],
        "race": [1], "urban": [1], "informal": [True],
    })
    monkeypatch.setattr("uxplain.datasets._pnadc_frame", lambda *args: frame)
    result = fetch_pnadc(target="informal")
    assert "informal" not in result.feature_names
    assert result.target.tolist() == [True]


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
