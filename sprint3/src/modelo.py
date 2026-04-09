# src/modelo.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

from processamento import load_and_prepare_data


class RandomForestTrainer:
    """Treinamento de Random Forest para predição de risco clínico."""

    def __init__(self) -> None:
        # Ajuste mais robusto (vc pode voltar para n_estimators=100 e max_depth=10 depois)
        self.model = RandomForestClassifier(
            n_estimators=200,           # mais árvores → melhor estabilidade
            max_depth=15,               # mais complexidade
            class_weight="balanced",    # prioriza classes menores
            random_state=42,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)


class LogisticRegressionBaseline:
    """Baseline com Regressão Logística (multiclasse automática)."""

    def __init__(self) -> None:
        # Removido multi_class; scikit-learn recente já lida com multiclasse
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, le = load_and_prepare_data()

    # Salva o scaler e encoder
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(le, "models/label_encoder.joblib")

    # Random Forest
    rf = RandomForestTrainer()
    rf.fit(X_train, y_train)
    rf.save("models/random_forest.joblib")

    y_pred_rf = rf.predict(X_test)
    print("Random Forest:")
    print(classification_report(y_test, y_pred_rf))

    # Logistic Regression
    lr = LogisticRegressionBaseline()
    lr.fit(X_train, y_train)
    lr.save("models/logistic_regression.joblib")

    y_pred_lr = lr.predict(X_test)
    print("\nLogistic Regression:")
    print(classification_report(y_test, y_pred_lr))