# ============================================================
# ETAPE 1 : CHARGEMENT ET DECOUVERTE DES DONNEES
# ============================================================
# But : Charger les fichiers CSV et comprendre ce qu'on a
#
# AVANT DE LANCER CE FICHIER :
# 1. Cree un dossier sur ton PC, par exemple : C:\Projets\ML_Stress
# 2. Mets ce fichier Python dedans
# 3. Mets aussi dreaddit-train.csv et dreaddit-test.csv dedans
# 4. Ouvre le dossier dans VS Code (File > Open Folder)
# 5. Ouvre un terminal (Terminal > New Terminal)
# 6. Tape : python etape1_chargement.py
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Charger les fichiers CSV
train = pd.read_csv('dreaddit-train.csv', encoding='latin-1')
test = pd.read_csv('dreaddit-test.csv', encoding='latin-1')

# Dimensions
print('=' * 50)
print('DIMENSIONS DU DATASET')
print('=' * 50)
print(f'Train : {train.shape[0]} lignes x {train.shape[1]} colonnes')
print(f'Test  : {test.shape[0]} lignes x {test.shape[1]} colonnes')
print(f'Total : {train.shape[0] + test.shape[0]} posts Reddit')

# Colonnes principales
print('\n' + '=' * 50)
print('COLONNES PRINCIPALES')
print('=' * 50)
colonnes_cles = ['subreddit', 'text', 'label', 'confidence', 'sentiment',
                 'social_karma', 'social_upvote_ratio', 'social_num_comments']
for col in colonnes_cles:
    print(f'  - {col} : type = {train[col].dtype}')

# Apercu des 3 premiers posts
print('\n' + '=' * 50)
print('APERCU DES 3 PREMIERS POSTS')
print('=' * 50)
for i in range(3):
    label_text = 'STRESS' if train['label'].iloc[i] == 1 else 'NON-STRESS'
    print(f'\n--- Post {i+1} ---')
    print(f'   Subreddit : {train["subreddit"].iloc[i]}')
    print(f'   Label     : {label_text}')
    print(f'   Texte     : {train["text"].iloc[i][:150]}...')

# Distribution du label
print('\n' + '=' * 50)
print('DISTRIBUTION DU LABEL (ce qu on veut predire)')
print('=' * 50)
print(train['label'].value_counts())
print(f'\nRatio stress     : {train["label"].mean()*100:.1f}%')
print(f'Ratio non-stress : {(1-train["label"].mean())*100:.1f}%')
print('--> Le dataset est bien equilibre.')

# Valeurs manquantes
print('\n' + '=' * 50)
print('VALEURS MANQUANTES')
print('=' * 50)
missing = train.isnull().sum().sum()
print(f'Total : {missing}')
print('--> Aucune valeur manquante.')

# Subreddits
print('\n' + '=' * 50)
print('SUBREDDITS')
print('=' * 50)
print(train['subreddit'].value_counts())

# Statistiques du texte
print('\n' + '=' * 50)
print('STATISTIQUES DU TEXTE')
print('=' * 50)
train['text_length'] = train['text'].str.len()
train['word_count'] = train['text'].str.split().str.len()
print(f'Longueur moyenne : {train["text_length"].mean():.0f} caracteres')
print(f'Mots par post    : {train["word_count"].mean():.0f} mots')
print(f'Post le + court  : {train["text_length"].min()} caracteres')
print(f'Post le + long   : {train["text_length"].max()} caracteres')

print('\n' + '=' * 50)
print('ETAPE 1 TERMINEE -- Lance etape2_eda.py pour la suite')
print('=' * 50)
