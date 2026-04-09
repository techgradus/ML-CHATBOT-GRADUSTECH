## Arquitetura (Back‑end + Front‑end)

- **Front‑end**: Streamlit (interface web para entrada de dados clínicos).
- **Back‑end**: Flask (API Python 3.10+ para processamento e modelo).
- **Banco de dados**: SQLite (registro de pacientes e resultados).
- **ML**: Scikit‑learn (Random Forest) com dataset sintético (≥2.000 registros).

## Como rodar

1. `pip install -r requirements.txt`
2. `python src/data/gerador_dados.py`
3. `python src/modelo.py`
4. Em um terminal: `python backend/app.py`
5. Em outro terminal: `streamlit run src/app.py`