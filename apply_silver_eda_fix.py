"""Focused local repair; preview first, then --apply. No network calls."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil

KEY_BLOCK = 'raw_key = config.get("data", {}).get("entity_id")\nkey = contract["canonical_names"].get(raw_key, raw_key) if raw_key else None\nif key is None:\n    key = next(iter(roles.get("id", [])), None)\nadd("entity_key_configured", key is not None, f"key={key!r}")\nif key is not None:\n    exists = key in df.columns\n    add("entity_key_exists", exists, f"key={key!r}")\n    add("entity_key_in_id_role", key in roles.get("id", []), f"key={key!r}")\n    if exists:\n        nulls = int(df[key].isna().sum())\n        add("entity_key_non_null", nulls == 0, f"key={key!r}, nulls={nulls}")\n        if config.get("validation", {}).get("require_unique_entity_id", True):\n            duplicates = int(df[key].duplicated().sum())\n            add("id_unique", duplicates == 0, f"key={key!r}, duplicates={duplicates}")\nfor target_col in roles.get("target", []):\n    exists = target_col in df.columns\n    add(f"target_exists:{target_col}", exists, f"column={target_col!r}")\n    if exists:\n        nulls = int(df[target_col].isna().sum())\n        add(f"target_non_null:{target_col}", nulls == 0, f"nulls={nulls}")\n'
EDA_FIXTURE = '@pytest.fixture\ndef config():\n    """Self-contained contract for ai4i_small.csv, not the active project."""\n    names = {\n        "UDI": "udi", "Product ID": "product_id", "Type": "type",\n        "Air temperature [K]": "air_temperature_k",\n        "Process temperature [K]": "process_temperature_k",\n        "Rotational speed [rpm]": "rotational_speed_rpm",\n        "Torque [Nm]": "torque_nm", "Tool wear [min]": "tool_wear_min",\n        "Machine failure": "machine_failure",\n        **{c: c.lower() for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]},\n    }\n    outcomes = ["TWF", "HDF", "PWF", "OSF", "RNF"]\n    features = [\n        "Type", "Air temperature [K]", "Process temperature [K]",\n        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",\n    ]\n    return {\n        "data": {\n            "entity_id": "UDI", "target": "Machine failure",\n            "task": "binary_classification", "positive_label": 1,\n        },\n        "bronze_source_contract": {\n            "rows": 27, "columns": len(names), "col_names": list(names),\n            "roles": {\n                "id": ["UDI", "Product ID"], "feature": features,\n                "target": ["Machine failure"], "outcome": outcomes,\n            },\n        },\n        "validation": {\n            "require_unique_entity_id": True, "max_missing_rate": 0.0,\n            "allowed_target_values": [0, 1],\n        },\n        "silver_eda_contract": {\n            "canonical_names": names,\n            "canonical_dtypes": {\n                "udi": "int64", "product_id": "object", "type": "object",\n                "air_temperature_k": "float64", "process_temperature_k": "float64",\n                "rotational_speed_rpm": "int64", "torque_nm": "float64",\n                "tool_wear_min": "int64", "machine_failure": "int8",\n                **{c.lower(): "int8" for c in outcomes},\n            },\n            "value_mappings": {"machine_failure": {"0": 0, "1": 1}},\n            "missingness": {"default_max_rate": 0.0},\n        },\n    }\n'
REQUIRED_TEST = 'def test_row_count_is_required(df, roles, config):\n    config["bronze_source_contract"].pop("rows")\n    with pytest.raises(KeyError, match="rows"):\n        silver.check_gates(df, roles, config)\n'


def function(tree, name):
    found = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one top-level function {name!r}")
    return found[0]


def edit(text, edits):
    lines = text.splitlines(keepends=True)
    for first, last, replacement in sorted(edits, reverse=True):
        lines[first - 1:last] = [replacement.rstrip() + "\n"]
    output = "".join(lines)
    ast.parse(output)
    return output


def patch_silver(text):
    tree = ast.parse(text)
    node = function(tree, "check_gates")
    segment = ast.get_source_segment(text, node)
    if '"entity_key_configured"' in segment:
        return text
    candidates = [n for n in node.body if isinstance(n, ast.If) and ast.unparse(n.test) == "'id' in df.columns"]
    if len(candidates) != 1:
        raise ValueError("Expected the known hard-coded id check; refusing an ambiguous patch")
    old = candidates[0]
    block = "\n".join("    " + line if line else "" for line in KEY_BLOCK.splitlines())
    return edit(text, [(old.lineno, old.end_lineno, block)])


def patch_silver_tests(text):
    tree = ast.parse(text)
    function(tree, "test_configured_key_duplicates_rejected")
    edits = []
    optional = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "test_unspecified_row_count_is_optional"]
    for node in optional:
        edits.append((node.lineno, node.end_lineno, REQUIRED_TEST))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if ast.unparse(node.targets[0]) == "config['validation']['max_missing_rate']":
                value = ast.unparse(node.value)
                indent = " " * node.col_offset
                replacement = indent + 'config["silver_eda_contract"]["missingness"] = {"default_max_rate": ' + value + '}'
                edits.append((node.lineno, node.end_lineno, replacement))
                count += 1
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "silver.write_outputs":
            if len(node.args) != 3:
                raise ValueError("The writer helper must already use three positional arguments")
    if not count and '"missingness"' not in text and "'missingness'" not in text:
        raise ValueError("Expected the supplied generic missingness tests")
    return edit(text, edits)


def patch_eda_tests(text):
    tree = ast.parse(text)
    node = function(tree, "config")
    if 'Self-contained contract for ai4i_small.csv' in ast.get_source_segment(text, node):
        return text
    if 'ai4i_small.csv' not in text:
        raise ValueError("EDA tests no longer use the inspected legacy fixture")
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    edits = [(first, node.end_lineno, EDA_FIXTURE)]
    for other in tree.body:
        if isinstance(other, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in other.targets):
            edits.append((other.lineno, other.end_lineno, ""))
    return edit(text, edits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true", help="Write planned edits after making backups")
    args = parser.parse_args()
    root = args.root.resolve()
    targets = {
        "src/ml_template/data/silver.py": patch_silver,
        "tests/data/test_silver.py": patch_silver_tests,
        "tests/data/test_eda_script.py": patch_eda_tests,
    }
    fixture = root / "tests/fixtures/ai4i_small.csv"
    if not fixture.is_file():
        raise FileNotFoundError(f"Required existing fixture: {fixture}")
    changes = []
    for relative, transform in targets.items():
        path = root / relative
        old = path.read_text(encoding="utf-8")
        new = transform(old)
        if old != new:
            changes.append((relative, path, old, new))
    if not changes:
        print("No changes required.")
        return
    for relative, *_ in changes:
        print(f"Would update: {relative}")
    if not args.apply:
        print("Preview only; rerun with --apply to write these changes.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root / ".policy_update_backups" / f"silver_eda_fix_{stamp}"
    for relative, path, old, new in changes:
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
        path.write_text(new, encoding="utf-8")
        print(f"Updated: {relative}")
    print(f"Original files backed up under {backup}")
    print("Run: uv run pytest tests/data/test_silver.py tests/data/test_eda_script.py -q")


if __name__ == "__main__":
    main()
