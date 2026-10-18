
#    .\.venv\Scripts\python.exe -m streamlit run app.py



from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"

st.set_page_config(page_title="Spam Message Checker", page_icon="✉️", layout="centered")


@st.cache_resource(show_spinner="Model tayyar ho raha hai...")
def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError("spam.csv file app.py ke same folder mein nahi mili.")

    data = pd.read_csv(DATA_PATH, encoding="latin-1")
    if not {"v1", "v2"}.issubset(data.columns):
        raise ValueError("spam.csv mein v1 aur v2 columns hone chahiye.")

    data = data[["v1", "v2"]].dropna().copy()
    data["v1"] = data["v1"].astype(str).str.strip().str.lower()
    data["v2"] = data["v2"].astype(str).str.strip()
    data = data[data["v1"].isin(["ham", "spam"]) & data["v2"].ne("")]

    if data.empty or data["v1"].nunique() < 2:
        raise ValueError("Dataset mein spam aur normal dono messages hone chahiye.")

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=30000)),
        ("classifier", MultinomialNB(alpha=0.35)),
    ])
    model.fit(data["v2"], data["v1"])
    spam_count = int((data["v1"] == "spam").sum())
    normal_count = int((data["v1"] == "ham").sum())
    return model, spam_count, normal_count


st.title("✉️ Spam Message Checker")
st.write("Apna email ya message neeche paste karo, phir **Check Message** button dabao.")

try:
    model, spam_count, normal_count = train_model()
except Exception as err:
    st.error(f"Model start nahi ho saka: {err}")
    st.info("Check karo ki app.py aur spam.csv ek hi folder mein hain.")
    st.stop()

with st.form("check_message_form"):
    message = st.text_area(
        "Email / Message",
        height=220,
        placeholder="Yahan apna email ya message likho ya paste karo...",
        help="Demo mein password, OTP ya doosri private information paste na karein.",
    )
    check_clicked = st.form_submit_button("Check Message", use_container_width=True)

if check_clicked:
    clean_message = message.strip()
    if not clean_message:
        st.warning("Pehle email ya message likho.")
    else:
        predictions = model.predict_proba([clean_message])[0]
        classes = list(model.named_steps["classifier"].classes_)
        spam_probability = float(predictions[classes.index("spam")])
        normal_probability = float(predictions[classes.index("ham")])

        st.divider()
        st.subheader("Result")
        if spam_probability >= normal_probability:
            st.error("🚨 Ye message SPAM ho sakta hai.")
        else:
            st.success("✅ Ye message NORMAL (Not Spam) ho sakta hai.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spam chance", f"{spam_probability * 100:.1f}%")
            st.progress(spam_probability)
        with col2:
            st.metric("Normal / Not Spam chance", f"{normal_probability * 100:.1f}%")
            st.progress(normal_probability)

        st.caption("Ye trained model ka estimate hai, 100% guarantee nahi. Dataset SMS-style messages par based hai; real emails par result alag ho sakta hai.")

with st.expander("Dataset ke baare mein"):
    col1, col2 = st.columns(2)
    col1.metric("Spam examples", f"{spam_count:,}")
    col2.metric("Normal examples", f"{normal_count:,}")
    st.caption("Ye counts training dataset ke hain. Upar diye gaye percentages tumhare likhe message ka model estimate hain.")
