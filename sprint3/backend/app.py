# backend/app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Caminhos fixos (ajuste se necessário)
MODEL_PATH = "models/random_forest.joblib"
SCALER_PATH = "models/scaler.joblib"

# Carrega o modelo e o scaler
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler não encontrado em: {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Classes de risco (assumindo mesma ordem do treino)
CLASSES = ["Risco_Baixo", "Risco_Medio", "Risco_Alto"]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Validar campos
    required = ["nome", "idade", "glicose", "pressao", "imc", "colesterol"]
    for key in required:
        if key not in data:
            return jsonify({"error": f"Campo {key} é obrigatório."}), 400

    try:
        X = np.array(
            [
                [
                    float(data["idade"]),
                    float(data["glicose"]),
                    float(data["pressao"]),
                    float(data["imc"]),
                    float(data["colesterol"]),
                ]
            ]
        )

        X_scaled = scaler.transform(X)
        y_proba = model.predict_proba(X_scaled)[0]
        y_pred = model.predict(X_scaled)[0]

        risk = CLASSES[y_pred]
        proba_dict = dict(zip(CLASSES, y_proba))

        # (opcional no Flask, se quiser salvar)
        # from backend.database import save_paciente
        # save_paciente(
        #     nome=data["nome"],
        #     idade=data["idade"],
        #     glicose=data["glicose"],
        #     pressao=data["pressao"],
        #     imc=data["imc"],
        #     colesterol=data["colesterol"],
        #     risco=risk,
        #     prob_baixo=proba_dict["Risco_Baixo"],
        #     prob_medio=proba_dict["Risco_Medio"],
        #     prob_alto=proba_dict["Risco_Alto"],
        # )

        return jsonify(
            {
                "risco_class": risk,
                "probabilidade_baixo": proba_dict["Risco_Baixo"],
                "probabilidade_medio": proba_dict["Risco_Medio"],
                "probabilidade_alto": proba_dict["Risco_Alto"],
                "probabilidades_porcentagem": {
                    "baixo": round(proba_dict["Risco_Baixo"] * 100, 2),
                    "medio": round(proba_dict["Risco_Medio"] * 100, 2),
                    "alto": round(proba_dict["Risco_Alto"] * 100, 2),
                },
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Inicializa banco de dados
    from backend.database import init_db

    init_db()

    app.run(host="0.0.0.0", port=5000, debug=True)