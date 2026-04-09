# src/processamento.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import joblib

# depois de criar o scaler
joblib.dump(scaler, "models/scaler.joblib")

def load_and_prepare_data(
    data_path: str = "data/dataset_sintetico.csv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carrega, limpa e prepara o dataset para treino/teste.

    Args:
        data_path: caminho do CSV.

    Returns:
        X_train, X_test, y_train, y_test, scaler, le (para reverter labels).
    """
    df = pd.read_csv(data_path)

    # Colunas de features
    feature_cols = ["Idade", "Glicose", "Pressao_Arterial", "IMC", "Colesterol"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Risco"].values

    # Cleanup básico
    # Remove duplicatas (se houver)
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        X = df[feature_cols].values.astype(np.float32)
        y = df["Risco"].values

    # Codifica classes
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Normalização
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split 80/20 estratificado
    # Estratificação mantém proporções de risco entre treino/teste (crucial em saúde)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, scaler, le


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, le = load_and_prepare_data()
    print("X_train shape:", X_train.shape)
    print("Classes únicas em y_train:", np.unique(y_train))
