# src/app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd


@st.cache_resource
def load_model_and_scaler():
    """Carrega modelo treinado e scaler."""
    model = joblib.load("models/random_forest.joblib")
    scaler = joblib.load("models/scaler.joblib")  # Simulando salvamento de scaler
    # Se não tiver salvo, recarregue do processamento
    # Alternativa: refazer o scaler no ponto de entrada
    return model, scaler


def main():
    st.set_page_config(page_title="Predição de Risco Clínico", layout="centered")

    st.title("🏥 Predição de Risco Clínico")

    st.markdown(
        """
        Este modelo estima o risco clínico (Baixo, Médio, Alto) com base em:
        - Idade
        - Glicose
        - Pressão Arterial
        - IMC
        - Colesterol
        """
    )

    # Entrada de usuário
    idade = st.slider("Idade (anos)", 18, 90, 45)
    glicose = st.number_input("Glicose (mg/dL)", 70.0, 300.0, 100.0, step=1.0)
    pressao = st.number_input(
        "Pressão Arterial Sistólica (mmHg)", 90.0, 180.0, 120.0, step=1.0
    )
    imc = st.number_input("IMC", 15.0, 50.0, 25.0, step=0.1)
    colesterol = st.number_input("Colesterol total (mg/dL)", 130.0, 350.0, 200.0, step=1.0)

    if st.button("Estimar Risco"):
        # Carrega modelo (se não tiver salvo o scaler, recria ele)
        X = np.array([[idade, glicose, pressao, imc, colesterol]], dtype=np.float32)

        # Carrega o scaler usado no treino (supondo que foi salvo em processamento.py)
        # Se não tiver salvo, substitua pela lógica de fit_transform recarregando o dataset
        # Para este exemplo, usamos o mesmo padrão de StandardScaler que foi usado em processamento.py
        # Em produção, salvar o scaler junto com o modelo é obrigatório.

        # Simulando o scaler (na prática, salve o scaler em processamento.py)
        df = pd.read_csv("data/dataset_sintetico.csv")
        feature_cols = ["Idade", "Glicose", "Pressao_Arterial", "IMC", "Colesterol"]
        X_full = df[feature_cols].values.astype(np.float3
