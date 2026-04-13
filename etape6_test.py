# ============================================================
# ETAPE 6 : TEST RAPIDE DU MODELE
# ============================================================
# Ce fichier teste le modele directement dans le terminal
# AVANT de lancer l'app Streamlit.
#
# Pour lancer : python etape6_test.py
# Pour lancer l'app web : streamlit run app.py
# ============================================================

import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Charger le modele et les outils
print('Chargement du modele...')
model = pickle.load(open('best_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
numeric_features = pickle.load(open('numeric_features.pkl', 'rb'))
print('Modele charge.\n')

# Fonction de nettoyage
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# Fonction de prediction
def predict_stress(text):
    cleaned = clean_text(text)
    text_features = tfidf.transform([cleaned])
    num_features = np.zeros((1, len(numeric_features)))
    combined = hstack([text_features, num_features])
    pred = model.predict(combined)[0]
    prob = model.predict_proba(combined)[0]
    return pred, prob[0] * 100, prob[1] * 100

# ============================================================
# TESTS
# ============================================================
print('=' * 60)
print('TEST DU MODELE EN DIRECT')
print('=' * 60)

tests = [
    "I can't sleep anymore. Every night I have panic attacks and I feel like the walls are closing in.",
    "Had a great day at work! Finished my project and went for a nice walk in the park.",
    "I'm so worried about my exam tomorrow. I haven't studied enough and I feel overwhelmed.",
    "Just cooked a delicious pasta for dinner. Life is good!",
    "My relationship is falling apart. I don't know how to fix things and I cry every night."
]

for text in tests:
    pred, prob_no, prob_stress = predict_stress(text)
    label = 'STRESS' if pred == 1 else 'NON-STRESS'
    print('\n[{}] Stress: {:.1f}% | Non-Stress: {:.1f}%'.format(label, prob_stress, prob_no))
    print('   "{}"'.format(text[:80]))

# ============================================================
# RESUME FINAL
# ============================================================
results = pickle.load(open('results.pkl', 'rb'))
best_overall = max(results, key=lambda x: results[x]['f1_score'])

print('\n\n' + '=' * 60)
print('RESUME FINAL DU PROJET ML')
print('=' * 60)
print()
print('DATASET')
print('  Source  : Dreaddit (Reddit)')
print('  Taille  : 3553 posts')
print()
print('MODELES TESTES')
for name in results:
    r = results[name]
    flag = ' <-- MEILLEUR' if name == best_overall else ''
    print('  {:<30s} F1: {:.1f}%{}'.format(name, r['f1_score'] * 100, flag))
print()
print('MEILLEUR MODELE : {}'.format(best_overall))
print('  F1-Score  : {:.2f}%'.format(results[best_overall]['f1_score'] * 100))
print('  Accuracy  : {:.2f}%'.format(results[best_overall]['accuracy'] * 100))
print('  AUC-ROC   : {:.2f}%'.format(results[best_overall]['auc_roc'] * 100))
print()
print('APPLICATION WEB')
print('  Pour lancer : streamlit run app.py')
print()
print('LIVRABLES')
print('  [OK] Code source commente (etape1 a etape6)')
print('  [OK] Application web (app.py)')
print('  [OK] Modele sauvegarde (best_model.pkl)')
print('  [  ] Article scientifique (a rediger)')
print('  [  ] Presentation orale 5-7 min (a preparer)')
print()
print('=' * 60)
print('PROJET ML TERMINE !')
print('=' * 60)
