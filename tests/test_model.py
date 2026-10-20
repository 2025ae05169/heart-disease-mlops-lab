import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.preprocess import FEATURES, build_pipeline, prepare_data


def test_model_can_fit_predict_with_missing_values():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(30):
        row = {feature: float(rng.integers(1, 5)) for feature in FEATURES}
        row['age'] = np.nan if i == 0 else 40 + i
        row['target'] = i % 2
        rows.append(row)
    X, y = prepare_data(pd.DataFrame(rows))
    model = build_pipeline(LogisticRegression(max_iter=2000))
    model.fit(X, y)
    predictions = model.predict(X.iloc[:4])
    probabilities = model.predict_proba(X.iloc[:4])
    assert len(predictions) == 4
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
