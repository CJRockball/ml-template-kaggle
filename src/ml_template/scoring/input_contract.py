"""Frozen schemas. V1 rejects missing values, extra columns and unseen categories."""

import numpy as np
import pandas as pd


def scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or type(value) not in (str, int, float, bool):
        raise ValueError("only scalar string/numeric/bool categories are supported")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite category")
    return value


def frame_schema(frame):
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("nonempty frame with unique columns required")
    schema = []
    for name in frame.columns:
        series = frame[name]
        if series.isna().any():
            raise ValueError(f"missing values are unsupported: {name}")
        if isinstance(series.dtype, pd.CategoricalDtype):
            item = {
                "kind": "category",
                "categories": [scalar(v) for v in series.cat.categories],
                "ordered": bool(series.cat.ordered),
            }
        elif pd.api.types.is_bool_dtype(series.dtype):
            item = {"kind": "bool", "dtype": str(series.dtype)}
        elif pd.api.types.is_integer_dtype(series.dtype):
            item = {"kind": "integer", "dtype": str(series.dtype)}
        elif pd.api.types.is_float_dtype(series.dtype):
            item = {"kind": "float", "dtype": str(series.dtype)}
        elif pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
            series.dtype
        ):
            if not series.map(lambda value: isinstance(value, str)).all():
                raise ValueError(f"mixed/non-string object column unsupported: {name}")
            item = {"kind": "string", "values": sorted(series.unique().tolist())}
        else:
            raise ValueError(f"unsupported input dtype: {name}: {series.dtype}")
        schema.append({"name": str(name), **item})
    return schema


def coerce_frame(frame, schema):
    names = [item["name"] for item in schema]
    if not names or len(set(names)) != len(names) or frame.columns.duplicated().any():
        raise ValueError("invalid/duplicate column schema")
    if set(frame.columns) != set(names):
        raise ValueError(
            f"column mismatch; missing={sorted(set(names) - set(frame.columns))}, extra={sorted(set(frame.columns) - set(names))}"
        )
    if frame.empty:
        raise ValueError("empty scoring frame")
    out = frame[names].copy()
    for item in schema:
        name, kind = item["name"], item["kind"]
        series = out[name]
        if series.isna().any():
            raise ValueError(f"missing values: {name}")
        if kind in {"category", "string"}:
            allowed = item["categories"] if kind == "category" else item["values"]
            if not series.isin(allowed).all():
                raise ValueError(f"unseen category/value: {name}")
            out[name] = (
                pd.Categorical(series, categories=allowed, ordered=item["ordered"])
                if kind == "category"
                else series.astype(object)
            )
        elif kind == "bool":
            if not series.map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ValueError(f"boolean values required: {name}")
            out[name] = series.astype(item["dtype"])
        elif kind in {"integer", "float"}:
            if pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(
                series.dtype
            ):
                raise ValueError(
                    f"numeric dtype required: {name}; implicit text parsing is disabled"
                )
            values = series.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite numeric input: {name}")
            if kind == "integer":
                if (series % 1 != 0).any():
                    raise ValueError(f"fractional integer input: {name}")
                bounds = np.iinfo(np.dtype(item["dtype"]))
                if ((series < bounds.min) | (series > bounds.max)).any():
                    raise ValueError(f"integer out of range: {name}")
            cast = series.astype(item["dtype"])
            if not np.isfinite(cast.to_numpy(dtype=float)).all():
                raise ValueError(f"numeric overflow: {name}")
            if kind == "float" and not np.array_equal(values, cast.to_numpy(dtype=float)):
                raise ValueError(f"lossy numeric conversion: {name}")
            out[name] = cast
        else:
            raise ValueError(f"unsupported schema kind: {kind}")
    return out


def validate_batch(frame, contract):
    key = contract["key_col"]
    if frame.columns.duplicated().any() or key not in frame:
        raise ValueError("missing key or duplicate column names")
    keys = frame[key]
    if keys.empty or keys.isna().any() or keys.duplicated().any():
        raise ValueError("keys must be nonempty, unique and non-null")
    kind = contract["key_kind"]
    if kind not in {"str", "int"}:
        raise ValueError("unsupported entity-key schema")
    ok = keys.map(
        lambda value: (
            isinstance(value, str)
            if kind == "str"
            else isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        )
    )
    if not ok.all():
        raise ValueError("entity key type differs from release")
    out = coerce_frame(frame.drop(columns=[key]), contract["source_schema"])
    out.index = pd.Index(keys.to_numpy(), name=key)
    return out
