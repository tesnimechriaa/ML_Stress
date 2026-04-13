# ============================================================
# ETAPE 3 : PRETRAITEMENT DES DONNEES
# ============================================================
# But : Transformer le texte brut en chiffres que les modeles
#       ML peuvent comprendre.
#
# 3 choses a faire :
#   1. Nettoyer le texte (enlever liens, ponctuation, stopwords)
#   2. Transformer le texte en chiffres (TF-IDF)
#   3. Normaliser les features numeriques
#
# Pour lancer : python etape3_pretraitement.py
# ============================================================

import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Charger les donnees
train = pd.read_csv('dreaddit-train.csv', encoding='latin-1')
test = pd.read_csv('dreaddit-test.csv', encoding='latin-1')

print('Donnees chargees : train={}, test={}'.format(len(train), len(test)))

# ============================================================
# 3.1 - Fonction de nettoyage du texte
# ============================================================
# Pourquoi nettoyer ?
# Le texte brut contient du bruit : liens, ponctuation, majuscules
# Ce bruit perturbe le modele. On doit le supprimer.
#
# Exemple AVANT : "I'm SO anxious!!! https://reddit.com"
# Exemple APRES : "anxious"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Supprimer les liens
    text = re.sub(r'http\S+|www\S+', '', text)
    # 2. Garder uniquement les lettres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Minuscules
    text = text.lower()
    # 4. Supprimer stopwords + lemmatisation
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# Test sur un exemple
exemple = "I'm SO anxious!!! https://reddit.com I can't stop worrying about everything"
print('\nTest nettoyage :')
print('  AVANT :', exemple)
print('  APRES :', clean_text(exemple))

# ============================================================
# 3.2 - Appliquer le nettoyage sur tout le dataset
# ============================================================
print('\nNettoyage du texte en cours...')
train['clean_text'] = train['text'].apply(clean_text)
test['clean_text'] = test['text'].apply(clean_text)
print('Nettoyage termine.')

print('\nExemple :')
print('  ORIGINAL :', train['text'].iloc[0][:150], '...')
print('  NETTOYE  :', train['clean_text'].iloc[0][:150], '...')

# ============================================================
# 3.3 - TF-IDF : Transformer le texte en chiffres
# ============================================================
# TF = combien de fois un mot apparait dans un post
# IDF = le mot est-il rare ou commun dans tous les posts ?
# TF-IDF = TF x IDF
# Un mot frequent dans UN post mais rare dans les AUTRES = score eleve
#
# max_features=5000 : on garde les 5000 mots les plus importants
# ngram_range=(1,2) : mots seuls + paires de mots ("panic attack")

print('\nTransformation TF-IDF en cours...')
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95
)

X_text_train = tfidf.fit_transform(train['clean_text'])
X_text_test = tfidf.transform(test['clean_text'])

print('TF-IDF termine.')
print('  Matrice train :', X_text_train.shape)
print('  Matrice test  :', X_text_test.shape)
print('  Vocabulaire   :', len(tfidf.get_feature_names_out()), 'mots/bigrammes')

# Top 15 mots les plus importants
print('\nTop 15 mots du vocabulaire TF-IDF :')
feature_names = tfidf.get_feature_names_out()
mean_tfidf = X_text_train.mean(axis=0).A1
top_indices = mean_tfidf.argsort()[-15:][::-1]
for i in top_indices:
    print(f'  {feature_names[i]} : {mean_tfidf[i]:.4f}')

# ============================================================
# 3.4 - Preparer les features numeriques (LIWC + Social)
# ============================================================
# En plus du texte, on a des features numeriques deja calculees :
# - Features LIWC : 93 features psycholinguistiques
# - Features sociales : karma, upvotes, commentaires
# - Sentiment
#
# On les NORMALISE avec StandardScaler
# Normaliser = mettre toutes les features sur la meme echelle

liwc_cols = [c for c in train.columns if c.startswith('lex_liwc_')]
dal_cols = [c for c in train.columns if c.startswith('lex_dal_')]
social_cols = ['social_karma', 'social_upvote_ratio', 'social_num_comments']
other_cols = ['sentiment', 'syntax_fk_grade', 'syntax_ari']

numeric_features = liwc_cols + dal_cols + social_cols + other_cols

print('\nFeatures numeriques selectionnees :', len(numeric_features))
print('  LIWC    :', len(liwc_cols))
print('  DAL     :', len(dal_cols))
print('  Social  :', len(social_cols))
print('  Autres  :', len(other_cols))

scaler = StandardScaler()
X_num_train = scaler.fit_transform(train[numeric_features])
X_num_test = scaler.transform(test[numeric_features])

print('Normalisation terminee.')

# ============================================================
# 3.5 - Combiner TF-IDF + Features numeriques
# ============================================================
# On combine les 2 types de features en un seul grand tableau

X_train_combined = hstack([X_text_train, X_num_train])
X_test_combined = hstack([X_text_test, X_num_test])

y_train = train['label'].values
y_test = test['label'].values

print('\nFEATURES COMBINEES :')
print('  X_train :', X_train_combined.shape)
print('  X_test  :', X_test_combined.shape)
print('  y_train :', y_train.shape, '-- {} stress / {} non-stress'.format(sum(y_train), len(y_train)-sum(y_train)))
print('  y_test  :', y_test.shape, '-- {} stress / {} non-stress'.format(sum(y_test), len(y_test)-sum(y_test)))

# ============================================================
# 3.6 - Sauvegarder les donnees preparees
# ============================================================
# On sauvegarde tout pour les reutiliser dans etape4

pickle.dump(X_train_combined, open('X_train.pkl', 'wb'))
pickle.dump(X_test_combined, open('X_test.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(numeric_features, open('numeric_features.pkl', 'wb'))

print('\nFichiers sauvegardes :')
print('  X_train.pkl')
print('  X_test.pkl')
print('  y_train.pkl')
print('  y_test.pkl')
print('  tfidf_vectorizer.pkl')
print('  scaler.pkl')
print('  numeric_features.pkl')

# ============================================================
# RESUME
# ============================================================
print('\n' + '=' * 50)
print('RESUME DU PRETRAITEMENT')
print('=' * 50)
print('1. Nettoyage du texte : liens, ponctuation, stopwords, lemmatisation')
print('2. TF-IDF : {} features textuelles'.format(X_text_train.shape[1]))
print('3. Features numeriques : {} (LIWC + Social + Sentiment)'.format(X_num_train.shape[1]))
print('4. Dataset final : {} features au total'.format(X_train_combined.shape[1]))
print()
print('=' * 50)
print('ETAPE 3 TERMINEE -- Lance etape4_modeles.py pour la suite')
print('=' * 50)
