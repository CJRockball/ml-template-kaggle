"""Tests for the Silver build: gates, label report, and Parquet output.

These tests use small in-memory frames, so they need no Bronze database.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from ml_template.data import silver

N_ROWS = 6
OUTCOMES = ["twf", "hdf", "pwf", "osf", "rnf"]
ROLES = {
    "id": ["udi", "product_id"],
    "feature": ["type", "torque_nm", "rotational_speed_rpm", "tool_wear_min"],
    "target": ["machine_failure"],
    "outcome": OUTCOMES,
}
CONFIG = {
    "bronze_source_contract": {"rows": N_ROWS},
    "silver_eda_contract": {
        "canonical_names": {
            "UDI": "udi",
            "Product ID": "product_id",
            "Type": "type",
            "Torque [Nm]": "torque_nm",
            "Rotational speed [rpm]": "rotational_speed_rpm",
            "Tool wear [min]": "tool_wear_min",
            "Machine failure": "machine_failure",
            **{c.upper(): c for c in OUTCOMES},
        },
        "canonical_dtypes": {
            "udi": "int64",
            "product_id": "object",
            "type": "object",
            "torque_nm": "float64",
            "rotational_speed_rpm": "int64",
            "tool_wear_min": "int64",
            "machine_failure": "int8",
            **dict.fromkeys(OUTCOMES, "int8"),
        },
    },
}


def make_df() -> pd.DataFrame:
    """Six rows: 0 normal, 1 consistent failure, 2 failure without mode,
    3 mode without failure, 4 compound modes, 5 normal."""
    modes = {c: [0] * N_ROWS for c in OUTCOMES}
    modes["twf"][1] = 1
    modes["rnf"][3] = 1
    modes["hdf"][4] = 1
    modes["pwf"][4] = 1
    failure = [0, 1, 1, 0, 1, 0]
    df = pd.DataFrame(
        {
            "dataset_version_id": "ds_test",
            "source_row_number": range(1, N_ROWS + 1),
            "udi": range(1, N_ROWS + 1),
            "product_id": [f"M{i}" for i in range(N_ROWS)],
            "type": ["L", "M", "H", "L", "M", "H"],
            "torque_nm": [40.0, 42.5, 38.1, 60.2, 45.0, 39.9],
            "rotational_speed_rpm": [1500, 1450, 1600, 1300, 1550, 1480],
            "tool_wear_min": [0, 10, 20, 30, 40, 50],
            "machine_failure": failure,
            **modes,
        }
    )
    for col in ["machine_failure", *OUTCOMES]:
        df[col] = df[col].astype("int8")
    return df


def gate(results: list[dict], name: str) -> dict:
    return next(g for g in results if g["check"] == name)


def failed(results: list[dict]) -> set[str]:
    return {g["check"] for g in results if not g["passed"]}


@pytest.fixture
def df() -> pd.DataFrame:
    return make_df()


def test_clean_frame_passes_all_gates(df):
    results = silver.check_gates(df, ROLES, CONFIG)
    assert failed(results) == set()


def test_wrong_row_count_fails(df):
    results = silver.check_gates(df.iloc[:-1], ROLES, CONFIG)
    assert "row_count" in failed(results)


def test_row_count_uses_configured_expectation(df):
    cfg = {**CONFIG, "bronze_source_contract": {"rows": 10_000}}
    results = silver.check_gates(df, ROLES, cfg)
    assert "row_count" in failed(results)
    assert "observed=6" in gate(results, "row_count")["detail"]


def test_missing_column_fails(df):
    results = silver.check_gates(df.drop(columns=["torque_nm"]), ROLES, CONFIG)
    assert "expected_columns" in failed(results)


def test_null_after_cast_fails(df):
    df.loc[2, "torque_nm"] = None
    results = silver.check_gates(df, ROLES, CONFIG)
    assert "no_new_nulls_after_cast" in failed(results)


def test_dtype_mismatch_fails(df):
    df["tool_wear_min"] = df["tool_wear_min"].astype("float64")
    results = silver.check_gates(df, ROLES, CONFIG)
    assert "declared_dtypes" in failed(results)


def test_binary_outside_0_1_fails(df):
    df.loc[0, "hdf"] = 2
    results = silver.check_gates(df, ROLES, CONFIG)
    assert "binary_fields_in_0_1" in failed(results)


def test_unknown_type_fails(df):
    df.loc[0, "type"] = "X"
    results = silver.check_gates(df, ROLES, CONFIG)
    assert "type_in_L_M_H" in failed(results)


def test_duplicate_udi_fails(df):
    df.loc[1, "udi"] = df.loc[0, "udi"]
    results = silver.check_gates(df, ROLES, CONFIG)
    assert "udi_unique" in failed(results)


def test_duplicate_source_record_with_new_lineage_fails(df):
    dup = df.iloc[[0]].copy()
    dup["source_row_number"] = 99
    df = pd.concat([df, dup], ignore_index=True)
    cfg = {**CONFIG, "bronze_source_contract": {"rows": N_ROWS + 1}}
    results = silver.check_gates(df, ROLES, cfg)
    assert "no_duplicate_source_records" in failed(results)


def test_label_inconsistencies_are_reported_not_changed(df):
    before = df.copy(deep=True)
    report = silver.report_label_inconsistencies(df, ROLES)

    assert report["failure_without_mode"]["count"] == 1
    assert report["failure_without_mode"]["source_row_numbers"] == [3]
    assert report["mode_without_failure"]["count"] == 1
    assert report["mode_without_failure"]["source_row_numbers"] == [4]
    assert report["compound_modes"]["count"] == 1
    assert report["compound_modes"]["source_row_numbers"] == [5]
    pd.testing.assert_frame_equal(df, before)


def test_label_inconsistencies_do_not_fail_gates(df):
    results = silver.check_gates(df, ROLES, CONFIG)
    assert all("label" not in g["check"] for g in results)
    assert failed(results) == set()


def _write(df, tmp_path, gates):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    labels = silver.report_label_inconsistencies(df, ROLES)
    code = silver.write_outputs(
        df,
        ROLES,
        gates,
        labels,
        dataset_name="ai4i",
        dataset_version_id="ds_test",
        config_path=config_path,
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "quality",
    )
    return code


def test_passing_build_writes_parquet_manifest_and_report(df, tmp_path):
    gates = silver.check_gates(df, ROLES, CONFIG)
    assert _write(df, tmp_path, gates) == 0

    version_dirs = list((tmp_path / "silver" / "ai4i").iterdir())
    assert len(version_dirs) == 1
    assert version_dirs[0].name.startswith("silver_ds_test_")

    loaded = pd.read_parquet(version_dirs[0] / "data.parquet")
    assert len(loaded) == N_ROWS
    assert not loaded.isna().any().any()
    pd.testing.assert_frame_equal(
        loaded.astype({"type": "object"}).reset_index(drop=True),
        df.reset_index(drop=True),
        check_dtype=False,
    )

    manifest = json.loads((version_dirs[0] / "manifest.json").read_text())
    assert manifest["row_count"] == N_ROWS
    assert manifest["validation_passed"] is True
    assert len(manifest["parquet_sha256"]) == 64

    report_path = tmp_path / "quality" / "ai4i" / version_dirs[0].name / "validation.json"
    report = json.loads(report_path.read_text())
    assert report["passed"] is True
    assert report["label_inconsistencies"]["compound_modes"]["count"] == 1


def test_failing_build_writes_report_but_no_parquet(df, tmp_path):
    df.loc[0, "type"] = "X"
    gates = silver.check_gates(df, ROLES, CONFIG)
    assert _write(df, tmp_path, gates) == 1

    assert not (tmp_path / "silver").exists()
    reports = list((tmp_path / "quality" / "ai4i").glob("*/validation.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["passed"] is False


def test_existing_silver_version_is_never_overwritten(df, tmp_path):
    gates = silver.check_gates(df, ROLES, CONFIG)
    assert _write(df, tmp_path, gates) == 0
    with pytest.raises(FileExistsError):
        _write(df, tmp_path, gates)
