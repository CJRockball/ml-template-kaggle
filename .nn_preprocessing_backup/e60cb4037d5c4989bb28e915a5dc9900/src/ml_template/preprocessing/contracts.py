"""Post-Gold validation utility for future adapters. Never fits or transforms."""

import numpy as np
import pandas as pd


def validate_estimator_frame(frame, *, allow_nan=False, require_numeric=True):
    """Validate the final DataFrame BEFORE estimator/tensor conversion.

    Future NN adapters must call this after saved Gold preprocessing. This utility
    is not inserted before existing fitted transformers and does not clean data.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("estimator input must be a nonempty DataFrame with unique columns")
    for name in frame.columns:
        series = frame[name]
        if not allow_nan and series.isna().any():
            raise ValueError(f"post-Gold missing values: {name}")
        numeric = pd.api.types.is_numeric_dtype(series.dtype)
        if require_numeric and not numeric:
            raise ValueError(f"post-Gold numeric input required: {name}")
        if numeric and not np.isfinite(series.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"post-Gold infinite numeric values: {name}")
    return frame
