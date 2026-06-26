"""Tests du préprocessing : pas de fuite, gestion des NaN."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.preprocessing import build_preprocessor  # noqa: E402


def _toy_df():
    return pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0],
        "cat1": ["a", "b", None, "a"],
    })


def test_preprocessor_handles_nan_and_outputs_dense():
    df = _toy_df()
    pre = build_preprocessor(numeric=["num1"], categorical=["cat1"])
    out = pre.fit_transform(df)
    # Sortie dense, sans NaN
    assert not np.isnan(out).any()
    assert out.shape[0] == 4


def test_preprocessor_fit_on_train_only():
    train = _toy_df()
    pre = build_preprocessor(numeric=["num1"], categorical=["cat1"])
    pre.fit(train)
    # Une modalité catégorielle inconnue au test ne doit pas faire échouer la transformation.
    test = pd.DataFrame({"num1": [3.0], "cat1": ["z_inconnu"]})
    out = pre.transform(test)
    assert out.shape[0] == 1
