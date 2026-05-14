import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "melhor_modelo.pkl"

MODEL_PATH.parent.mkdir(exist_ok=True)

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target #0=maligno, 1=benigno

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parametros_random = {'model__learning_rate': [0.01, 0.05, 0.1, 0.2], 'model__max_depth': [3, 4, 5, 6],
                     'model__n_estimators': [50, 100, 200, 300], 'model__subsample': [0.8, 0.9, 1.0], 'model__colsample_bytree': [0.8, 0.9, 1.0]}

xgb = Pipeline([('smote', SMOTE(random_state=0)), ('model', XGBClassifier(random_state=0))])

random_search = RandomizedSearchCV(xgb, param_distributions=parametros_random, n_iter=20,
                                   cv=5, scoring='accuracy', n_jobs=-1,  random_state=42)

features_reduzidas = ['mean concave points', 'worst radius', 'worst concavity', 'mean texture', 'worst perimeter', 'mean fractal dimension', 'area error']
X_treino_red = X_train[features_reduzidas]

random_search.fit(X_treino_red, y_train)
modelo_xgb = random_search.best_estimator_

joblib.dump(modelo_xgb, MODEL_PATH)