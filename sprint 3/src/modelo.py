# src/modelo.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np


class RandomForestTrainer:
    """Treinamento de Random Forest para predição de risco clínico."""

    def __init__(self) -> None:
        # Random Forest é adequado para dados tabulares clínicos porque:
        # - Capta não linearidades e interações entre variáveis (ex: Glicose × IMC).
        # - Lida bem com desbalanceamentos via ajuste de pesos ou threshold.
        # - Oferece importante feature_importance para interpretação clínica.
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
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
    """Baseline com Regressão Logística multinomial."""

    def __init__(self) -> None:
        # Regressão Logística serve como baseline:
        # - Simples, interpretable, boa para comparação de performance.
        # - Permite analisar pesos das features e significância clínica.
        self.model = LogisticRegression(
            multi_class="ovr",
            solver="lbfgs",
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
    from processamento import load_and_prepare_data

    X_train, X_test, y_train, y_test, _, _ = load_and_prepare_data()

    # Random Forest
    rf = RandomForestTrainer()
    rf.fit(X_train, y_train)
    rf.save("models/random_forest.joblib")

    y_pred_rf = rf.predict(X_test)
    print("Relatório de classificação (Random Forest):")
    print(classification_report(y_test, y_pred_rf))

    # Logistic Regression baseline
    lr = LogisticRegressionBaseline()
    lr.fit(X_train, y_train)
    lr.save("models/logistic_regression.joblib")

    y_pred_lr = lr.predict(X_test)
    print("\nRelatório de classificação (Logistic Regression baseline):")
    print(classification_report(y_test, y_pred_lr))
