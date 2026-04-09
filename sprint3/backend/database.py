# backend/database.py
import sqlite3
from datetime import datetime


def init_db():
    conn = sqlite3.connect("data/simulacoes_pacientes.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            idade INTEGER NOT NULL,
            glicose REAL NOT NULL,
            pressao REAL NOT NULL,
            imc REAL NOT NULL,
            colesterol REAL NOT NULL,
            risco_class TEXT NOT NULL,
            probabilidade_baixo REAL,
            probabilidade_medio REAL,
            probabilidade_alto REAL,
            created_at TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


def save_paciente(
    nome: str,
    idade: int,
    glicose: float,
    pressao: float,
    imc: float,
    colesterol: float,
    risco: str,
    prob_baixo: float,
    prob_medio: float,
    prob_alto: float,
):
    conn = sqlite3.connect("data/simulacoes_pacientes.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO pacientes
        (nome, idade, glicose, pressao, imc, colesterol, risco_class,
         probabilidade_baixo, probabilidade_medio, probabilidade_alto, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            nome,
            idade,
            glicose,
            pressao,
            imc,
            colesterol,
            risco,
            prob_baixo,
            prob_medio,
            prob_alto,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()