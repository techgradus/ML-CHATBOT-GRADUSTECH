# src/avaliacao.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    class_names: list[str] = None,
) -> dict:
    """Avalia o modelo com múltiplas métricas e exibe matriz de confusão.

    Args:
        y_true: verdadeiros rótulos.
        y_pred: predições.
        y_proba: probabilidades por classe (opcional).
        class_names: nomes das classes para legenda.

    Returns:
        dicionário com métricas.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = ["Baixo", "Medio", "Alto"]

    # Matriz de confusão textual simples
    print("Matriz de Confusão:")
    print("Verdade \\ Predito\t", "\t".join(class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]}\t", "\t".join(map(str, row)))

    # Também exibe relatório completo
    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Risco_Baixo", "Risco_Medio", "Risco_Alto"],
        )
    )

    # Plot (opcional, se chamado com matplotlib aberto)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        cbar=False,
    )
    plt.title("Matriz de Confusão")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    plt.tight_layout()
    plt.show()

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }

    if y_proba is not None:
        results["y_proba"] = y_proba

    return results


if __name__ == "__main__":
    from modelo import RandomForestTrainer, LogisticRegressionBaseline
    from processamento import load_and_prepare_data
    import joblib

    X_train, X_test, y_train, y_test, _, le = load_and_prepare_data()

    # Carrega o modelo Random Forest
    rf_model = joblib.load("models/random_forest.joblib")
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)

    evaluate_model(
        y_test,
        y_pred_rf,
        y_proba_rf,
        class_names=["Risco_Baixo", "Risco_Medio", "Risco_Alto"],
    )

    # Interpretation técnica (simulada para a saída típica)
    # Suponha que a matriz de confusão mostra:
    #                       Predito
    #              Baixo  Medio  Alto
    # Verdadeato
    # Baixo          130      10      5
    # Medio           15      90     15
    # Alto             8      12     80

    print("\n" + "=" * 70)
    print("ANÁLISE TÉCNICA (SIMULADA) DA MATRIZ DE CONFUSÃO")
    print("=" * 70)
    print(
        "Em um cenário típico, observamos que há alguns Falsos Negativos na classe "
        "'Risco_Alto' (pessoas de alto risco classificadas como Médio ou Baixo). "
        "Isso é crítico em contexto médico, pois pacientes de alto risco podem não ser "
        "encaminhados a tempo, aumentando o risco de eventos adversos (ex: infarto, AVC)."
    )
    print(
        "Por outro lado, há também Falsos Positivos em 'Risco_Baixo/Medio' (pessoas "
        "classificadas como Alto ou Médio quando na verdade são Baixo). Isso aumenta "
        "a carga de encaminhamentos e exames, mas é menos perigoso do ponto de vista "
        "de vida imediata."
    )
    print(
        "Para mitigar Falsos Negativos em 'Risco_Alto', o modelo poderia priorizar "
        "Recall nessa classe (ajustando threshold ou pesos de classe), aceitando mais "
        "Falsos Positivos em troca de segurança do paciente."
    )
