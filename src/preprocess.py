"""Preprocessing shared by training and inference; mirrors the coursework notebook."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
FEATURES = NUMERIC + CATEGORICAL


def prepare_data(df: pd.DataFrame):
    """Accept original 'num' or notebook's binary 'target' column."""
    df = df.copy()
    if 'target' not in df and 'num' in df:
        df['target'] = (pd.to_numeric(df.pop('num'), errors='raise') > 0).astype(int)
    if 'target' not in df:
        raise ValueError('Missing target/num column')
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}')
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(df['target'], errors='raise').astype(int)
    if not y.isin([0, 1]).all():
        raise ValueError('Target must be binary')
    return X, y


def build_pipeline(estimator):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocess = ColumnTransformer([('numeric', numeric, NUMERIC),
                                    ('categorical', categorical, CATEGORICAL)])
    return Pipeline([('preprocess', preprocess), ('model', estimator)])
