# ============================================================
# ETAPE 6 : APPLICATION WEB STREAMLIT
# ============================================================
# Ce fichier cree l'application web Streamlit.
#
# Pour lancer l'app : streamlit run app.py
# (pas python app.py, mais streamlit run app.py)
# ============================================================

import streamlit as st
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Stress Detector - Social Media",
    page_icon="",
    layout="wide"
)

# ============================================================
# CHARGER LE MODELE ET LES OUTILS
# ============================================================
@st.cache_resource
def load_models():
    model = pickle.load(open('best_model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    numeric_features = pickle.load(open('numeric_features.pkl', 'rb'))
    return model, tfidf, scaler, numeric_features

model, tfidf, scaler, numeric_features = load_models()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ============================================================
# FONCTION DE NETTOYAGE
# ============================================================
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ============================================================
# INTERFACE UTILISATEUR
# ============================================================
st.title("Detecteur de Stress sur les Reseaux Sociaux")
st.markdown("**Prediction de la propagation du stress collectif a partir de posts**")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Entrez un texte a analyser")
    user_text = st.text_area(
        "Collez ici un post de reseau social :",
        height=200,
        placeholder="Ex: I can't stop worrying about everything. My anxiety is through the roof and I feel like I'm losing control..."
    )
    analyze_button = st.button("Analyser le Stress", type="primary", use_container_width=True)

# ============================================================
# PREDICTION
# ============================================================
if analyze_button and user_text:
    cleaned = clean_text(user_text)
    text_features = tfidf.transform([cleaned])
    num_features = np.zeros((1, len(numeric_features)))
    combined_features = hstack([text_features, num_features])

    prediction = model.predict(combined_features)[0]
    probability = model.predict_proba(combined_features)[0]

    prob_no_stress = probability[0] * 100
    prob_stress = probability[1] * 100

    with col2:
        st.subheader("Resultat")

        if prediction == 1:
            st.error("STRESS DETECTE")
            st.metric("Probabilite de Stress", "{:.1f}%".format(prob_stress))
        else:
            st.success("PAS DE STRESS")
            st.metric("Probabilite Non-Stress", "{:.1f}%".format(prob_no_stress))

        st.markdown("### Niveau de Stress")
        st.progress(int(prob_stress))

        st.markdown("### Details")
        st.write("Probabilite Stress : **{:.1f}%**".format(prob_stress))
        st.write("Probabilite Non-Stress : **{:.1f}%**".format(prob_no_stress))
        st.write("Mots analyses : **{}**".format(len(cleaned.split())))

    st.markdown("---")
    st.subheader("Analyse du texte")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Texte original :**")
        st.info(user_text)
    with col_b:
        st.write("**Texte nettoye :**")
        st.info(cleaned)

elif analyze_button and not user_text:
    st.warning("Veuillez entrer un texte a analyser.")

# ============================================================
# EXEMPLES PREDEFINIS
# ============================================================
st.markdown("---")
st.subheader("Exemples de textes a tester")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Post stressant :**")
    st.code("I can't sleep anymore. Every night I have panic attacks and I feel like the walls are closing in. I don't know what to do.", language=None)

with col2:
    st.markdown("**Post non-stressant :**")
    st.code("Had a great day at work today! Finished my project early and went for a nice walk in the park.", language=None)

with col3:
    st.markdown("**Post ambigu :**")
    st.code("Starting a new job next week. Feeling a mix of excitement and nervousness about the change.", language=None)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "**Projet ML** -- Prediction de la propagation du stress collectif sur reseaux sociaux | "
    "Dataset : Dreaddit (Reddit) | Polytechnique Sousse 2025-2026"
)
