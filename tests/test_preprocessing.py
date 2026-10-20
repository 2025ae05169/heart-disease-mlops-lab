import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.preprocess import FEATURES, build_pipeline, prepare_data


def example_data():
    row = {feature: 1.0 for feature in FEATURES}
    return pd.DataFrame([{**row, 'num': 0}, {**row, 'num': 3}])


def test_original_target_is_binary():
    X, y = prepare_data(example_data())
    assert X.shape == (2, 13)
    assert y.tolist() == [0, 1]


def test_missing_feature_rejected():
    with pytest.raises(ValueError, match='Missing features'):
        prepare_data(example_data().drop(columns=['age']))


def test_pipeline_contains_preprocessing():
    pipeline = build_pipeline(LogisticRegression())
    assert list(pipeline.named_steps) == ['preprocess', 'model']
