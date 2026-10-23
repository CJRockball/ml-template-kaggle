"""Immutable, explicitly approved operational reference populations."""

from pathlib import Path

from ml_template.db import monitoring_registry as registry
from ml_template.monitoring.artifacts import new_id, read_json, seal, verify, write_json
from ml_template.monitoring.drift import profile
from ml_template.monitoring.monitor_inputs import load_batch


def build_reference(engine, config, *, output_root, check_only=False):
    batch = load_batch(engine, config.scoring_batch_id)
    if check_only:
        return {"ready": True, "n_rows": len(batch.source), "files_written": False}
    identity = new_id("ref")
    directory = Path(output_root).resolve() / identity
    directory.mkdir(parents=True, exist_ok=False)
    profiles = {}
    for item in batch.contract["source_schema"]:
        name = item["name"]
        profiles[f"input:{name}"] = profile(
            batch.source[name],
            numeric=item["kind"] in {"integer", "float"},
            bins=config.numeric_bins,
        )
    for name in ("p_raw", "p_calibrated", "p_release"):
        if name in batch.predictions:
            profiles[f"score:{name}"] = profile(
                batch.predictions[name], numeric=True, bins=config.numeric_bins
            )
    result = {
        "schema_version": 1,
        "reference_id": identity,
        "kind": "operational",
        "release_id": batch.release_id,
        "scoring_batch_id": batch.batch_id,
        "name": config.name,
        "approved_population": config.approved_population,
        "contract": batch.contract,
        "profiles": profiles,
        "source_hashes": batch.artifact_hashes,
        "flagged_fraction": batch.summary.get("flagged_fraction"),
        "n_rows": len(batch.source),
    }
    write_json(directory / "config.json", config.model_dump(mode="json"))
    write_json(directory / "reference.json", result)
    manifest_hash = seal(directory)
    registry.register_object(
        engine,
        identity,
        "reference",
        batch.batch_id,
        batch.release_id,
        directory,
        manifest_hash,
        {"name": config.name},
    )
    return {"reference_id": identity, "artifact_dir": str(directory)}


def load_reference(engine, identity, batch):
    record = registry.get_object(engine, identity, "reference")
    directory = Path(record["artifact_dir"])
    verify(directory, record["manifest_sha256"])
    result = read_json(directory / "reference.json")
    if (
        result["reference_id"] != identity
        or result["release_id"] != record["release_id"]
        or result["scoring_batch_id"] != record["scoring_batch_id"]
    ):
        raise ValueError("reference identity mismatch")
    if result["release_id"] != batch.release_id or result["contract"] != batch.contract:
        raise ValueError("reference must use the same release and source contract")
    return result
