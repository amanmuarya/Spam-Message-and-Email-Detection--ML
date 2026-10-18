# Simple Spam Message Checker (VS Code)

## Files
- `app.py` — simple interface with one message box and a Check Message button.
- `spam.csv` — training dataset.
- `requirements.txt` — required Python libraries.

## Run in Windows PowerShell / VS Code Terminal
Open this folder in VS Code, then run each command one at a time:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

If you already have a working `.venv`, skip the first command. If installing inside your existing `D:\AI-ML_class\hello_spam` folder, copy `app.py` there and keep `spam.csv` beside it, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app trains its classifier from `spam.csv`, so it does not need your existing `model.pkl`, `tfidf.pkl`, or `cv.pkl`.

Note: This dataset is SMS-style and predictions on email text may be less reliable. Percentages are model estimates, not guarantees.
