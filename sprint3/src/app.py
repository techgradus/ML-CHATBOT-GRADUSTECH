# src/app.py
import streamlit as st
import requests
import json


def main():
    st.set_page_config(page_title="🏥 Predição de Risco Clínico", layout="centered")

    st.title("🏥 Predição de Risco Clínico")

    st.markdown(
        """
        Insira os dados do paciente e o sistema calculará o **risco de doenças crônicas** (Baixo, Médio ou Alto).
        """
    )

    nome = st.text_input("Nome do Paciente (fictício)", "Paciente Teste")
    idade = st.slider("Idade (anos)", 18, 90, 45)
    glicose = st.number_input("Glicose (mg/dL)", 70.0, 300.0, 100.0, step=1.0)
    pressao = st.number_input(
        "Pressão Arterial Sistólica (mmHg)", 90.0, 180.0, 120.0, step=1.0
    )
    imc = st.number_input("IMC", 15.0, 50.0, 25.0, step=0.1)
    colesterol = st.number_input(
        "Colesterol total (mg/dL)", 130.0, 350.0, 200.0, step=1.0
    )

    if st.button("Enviar para back‑end e calcular risco"):
        payload = {
            "nome": nome,
            "idade": idade,
            "glicose": glicose,
            "pressao": pressao,
            "imc": imc,
            "colesterol": colesterol,
        }

        try:
            # Chamada à API Flask rodando localmente
            response = requests.post("http://localhost:5000/predict", json=payload)
            if response.status_code == 200:
                result = response.json()

                st.subheader("Resultado de Predição")

                st.write(f"**Classificação final:** {result['risco_class']}")

                st.write("**Probabilidades porcentuais:**")
                for k, prob in result["probabilidades_porcentagem"].items():
                    st.metric(k.capitalize(), f"{prob}%")

                # Gráfico simples (medidor de risco)
                import plotly.express as px

                labels = list(result["probabilidades_porcentagem"].keys())
                values = list(result["probabilidades_porcentagem"].values())

                fig = px.bar(
                    x=labels,
                    y=values,
                    labels={"x": "Risco", "y": "Probabilidade (%)"},
                    title="Distribuição de Probabilidades de Risco",
                    text=values,
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"Erro do back‑end: {response.json().get('error')}")

        except Exception as e:
            st.error(f"Erro ao comunicar com o back‑end: {e}")


if __name__ == "__main__":
    main()