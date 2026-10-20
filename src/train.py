"""Reproducible CLI training, aligned with the original notebook's model selection."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.preprocess import build_pipeline, prepare_data

SEED = 42


def train(data_path: str, output_dir: str):
    X, y = prepare_data(pd.read_csv(data_path))
    if y.nunique() != 2:
        raise ValueError('Both target classes must be present')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=SEED),
                                {'model__C': [0.1, 1.0, 10.0]}),
        'Random Forest': (RandomForestClassifier(random_state=SEED),
                          {'model__n_estimators': [100, 200], 'model__max_depth': [3, None]}),
    }
    rows, models = [], {}
    for name, (estimator, params) in candidates.items():
        search = GridSearchCV(build_pipeline(estimator), params, scoring='roc_auc', cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        rows.append({
            'model': name, 'cv_roc_auc': float(search.best_score_),
            'accuracy': accuracy_score(y_test, predicted),
            'precision': precision_score(y_test, predicted, zero_division=0),
            'recall': recall_score(y_test, predicted, zero_division=0),
            'f1': f1_score(y_test, predicted, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, probabilities),
            'best_params': str(search.best_params_),
        })
        models[name] = model
    comparison = pd.DataFrame(rows).sort_values('cv_roc_auc', ascending=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / 'model_comparison.csv', index=False)
    selected = comparison.iloc[0]['model']
    joblib.dump(models[selected], output / 'heart_disease_model.joblib')
    metadata = {'selected_model': selected, 'selection_criterion': 'highest training CV ROC-AUC',
                'feature_names': list(X.columns), 'metrics': comparison.to_dict(orient='records')}
    (output / 'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(comparison.to_string(index=False))
    print('Selected:', selected)
    return models[selected]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/heart_cleaned.csv')
    parser.add_argument('--output', default='ci_artifacts')
    args = parser.parse_args()
    train(args.data, args.output)
