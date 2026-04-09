# src/processamento.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  # já importado aqui


def load_and_prepare_data(
    data_path: str = "data/dataset_sintetico.csv",
) -> tuple:
    df = pd.read_csv(data_path)

    feature_cols = ["Idade", "Glicose", "Pressao_Arterial", "IMC", "Colesterol"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Risco"].values

    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        X = df[feature_cols].values.astype(np.float32)
        y = df["Risco"].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, scaler, le