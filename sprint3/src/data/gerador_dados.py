# src/data/gerador_dados.py
import pandas as pd
import numpy as np

np.random.seed(42)

N = 2000

# Idade: 18–90 anos
idade = np.random.randint(18, 91, size=N)

# Glicose (mg/dL): normal, pré-diabetes, diabetes leve
base_gluc = 80
disp_gluc = 30
glicose = np.random.normal(100, 30, size=N)
glicose = np.clip(glicose, 70, 300)

# Pressão arterial sistólica (mmHg): 90–180
pressao = np.random.normal(120, 20, size=N)
pressao = np.clip(pressao, 90, 180)

# IMC: 15–50
imc = np.random.normal(26, 6, size=N)
imc = np.clip(imc, 15, 50)

# Colesterol total (mg/dL): 130–350
colesterol = np.random.normal(200, 50, size=N)
colesterol = np.clip(colesterol, 130, 350)

# Nome fictício simples
nomes = [f"Paciente_{i+1}" for i in range(N)]

# Peso de cada fator no risco (coesão biológica)
z_gluc = (glicose - 100) / 50
z_pressao = (pressao - 120) / 20
z_imc = (imc - 25) / 5
z_col = (colesterol - 200) / 50
z_idade = (idade - 50) / 20

logodds = 0.8*z_gluc + 0.7*z_pressao + 0.9*z_imc + 0.6*z_col + 0.4*z_idade
proba_risco_alto = 1 / (1 + np.exp(-logodds))

# Risco em 3 classes
classes = []
for p in proba_risco_alto:
    r = np.random.rand()
    if r < 0.3:
        classes.append("Risco_Baixo")
    elif r < 0.8:
        classes.append("Risco_Medio")
    else:
        classes.append("Risco_Alto")

df = pd.DataFrame({
    "Nome_Ficticio": nomes,
    "Idade": idade,
    "Glicose": glicose.round(1),
    "Pressao_Arterial": pressao.round(1),
    "IMC": imc.round(2),
    "Colesterol": colesterol.round(1),
    "Risco": classes
})

df.to_csv("data/dataset_sintetico.csv", index=False)

print("Dataset gerado com 2.000+ registros.")
print("\nPrimeiras linhas:")
print(df.head())
print("\nEstatísticas:")
print(df.describe())
print("\nContagem de classes:")
print(df["Risco"].value_counts())
