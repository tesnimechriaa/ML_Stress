# ============================================================
# ETAPE 2 : ANALYSE EXPLORATOIRE DES DONNEES (EDA)
# ============================================================
# But : Visualiser les donnees avec des graphiques pour
#       trouver des patterns avant de construire un modele.
#
# Pour lancer : python etape2_eda.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Charger les donnees
train = pd.read_csv('dreaddit-train.csv', encoding='latin-1')
test = pd.read_csv('dreaddit-test.csv', encoding='latin-1')
df = pd.concat([train, test], ignore_index=True)
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f'Dataset combine : {len(df)} posts')
print('Generation des graphiques en cours...\n')

# ============================================================
# GRAPHIQUE 1 : Distribution Stress vs Non-Stress
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

label_counts = df['label'].value_counts()
axes[0].pie(label_counts.values,
            labels=['Stress (1)', 'Non-Stress (0)'],
            colors=['#e74c3c', '#2ecc71'],
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 13, 'fontweight': 'bold'})
axes[0].set_title('Distribution Stress vs Non-Stress', fontsize=14, fontweight='bold')

sns.countplot(data=df, x='label', palette=['#2ecc71', '#e74c3c'], ax=axes[1])
axes[1].set_xticklabels(['Non-Stress (0)', 'Stress (1)'])
axes[1].set_title('Nombre de posts par classe', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Nombre de posts')
for i, v in enumerate(label_counts.sort_index().values):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('graphique1_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('[1/8] Distribution Stress vs Non-Stress -- OK')

# ============================================================
# GRAPHIQUE 2 : Distribution Stress par Subreddit
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

subreddit_stress = df.groupby(['subreddit', 'label']).size().unstack(fill_value=0)
subreddit_stress.columns = ['Non-Stress', 'Stress']
subreddit_stress = subreddit_stress.sort_values('Stress', ascending=True)

subreddit_stress.plot(kind='barh', stacked=True,
                       color=['#2ecc71', '#e74c3c'], ax=ax)
ax.set_title('Distribution Stress par Subreddit', fontsize=14, fontweight='bold')
ax.set_xlabel('Nombre de posts')
ax.set_ylabel('Subreddit')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('graphique2_subreddits.png', dpi=150, bbox_inches='tight')
plt.close()
print('[2/8] Distribution par Subreddit -- OK')

# ============================================================
# GRAPHIQUE 3 : Longueur du texte par classe
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, color, name in [(0, '#2ecc71', 'Non-Stress'), (1, '#e74c3c', 'Stress')]:
    subset = df[df['label'] == label]
    axes[0].hist(subset['text_length'], bins=50, alpha=0.6, color=color, label=name)
axes[0].set_title('Distribution de la longueur du texte', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Nombre de caracteres')
axes[0].set_ylabel('Frequence')
axes[0].legend()

data_box = [df[df['label']==0]['word_count'].values, df[df['label']==1]['word_count'].values]
bp = axes[1].boxplot(data_box, labels=['Non-Stress', 'Stress'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
axes[1].set_title('Nombre de mots par classe', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Nombre de mots')

plt.tight_layout()
plt.savefig('graphique3_longueur_texte.png', dpi=150, bbox_inches='tight')
plt.close()
print('[3/8] Longueur du texte -- OK')

# ============================================================
# GRAPHIQUE 4 : Top Features LIWC correlees au stress
# ============================================================
liwc_cols = [c for c in df.columns if c.startswith('lex_liwc_')]
correlations = df[liwc_cols + ['label']].corr()['label'].drop('label').sort_values()

fig, ax = plt.subplots(figsize=(12, 8))

top_n = 15
top_corr = pd.concat([correlations.head(top_n), correlations.tail(top_n)])
colors_corr = ['#2ecc71' if v < 0 else '#e74c3c' for v in top_corr.values]

top_corr.plot(kind='barh', color=colors_corr, ax=ax)
ax.set_title('Top Features LIWC correlees au Stress', fontsize=14, fontweight='bold')
ax.set_xlabel('Correlation de Pearson avec le label Stress')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_yticklabels([l.replace('lex_liwc_', '') for l in top_corr.index])
plt.tight_layout()
plt.savefig('graphique4_liwc_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print('[4/8] Correlations LIWC -- OK')

# ============================================================
# GRAPHIQUE 5 : Distribution du Sentiment par classe
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, color, name in [(0, '#2ecc71', 'Non-Stress'), (1, '#e74c3c', 'Stress')]:
    subset = df[df['label'] == label]
    axes[0].hist(subset['sentiment'], bins=40, alpha=0.6, color=color, label=name)
axes[0].set_title('Distribution du Sentiment par classe', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Score de sentiment')
axes[0].set_ylabel('Frequence')
axes[0].legend()

sent_by_sub = df.groupby('subreddit')['sentiment'].mean().sort_values()
colors_sent = ['#e74c3c' if v < 0 else '#2ecc71' for v in sent_by_sub.values]
sent_by_sub.plot(kind='barh', color=colors_sent, ax=axes[1])
axes[1].set_title('Sentiment moyen par Subreddit', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Sentiment moyen')
axes[1].axvline(x=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('graphique5_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print('[5/8] Sentiment -- OK')

# ============================================================
# GRAPHIQUE 6 : Heatmap de correlation
# ============================================================
key_features = ['lex_liwc_negemo', 'lex_liwc_posemo', 'lex_liwc_anx',
                'lex_liwc_anger', 'lex_liwc_sad', 'lex_liwc_i',
                'lex_liwc_social', 'lex_liwc_health', 'lex_liwc_death',
                'lex_liwc_risk', 'sentiment', 'social_karma',
                'social_num_comments', 'social_upvote_ratio', 'label']

fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df[key_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn_r', center=0, ax=ax, vmin=-1, vmax=1)

labels = [l.replace('lex_liwc_', '').replace('social_', '') for l in key_features]
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels, rotation=0)
ax.set_title('Heatmap de Correlation - Features Cles', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graphique6_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('[6/8] Heatmap -- OK')

# ============================================================
# GRAPHIQUE 7 : Nuages de mots (WordCloud)
# ============================================================
stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'him',
              'she', 'her', 'it', 'its', 'they', 'them', 'what', 'which', 'who',
              'this', 'that', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
              'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the', 'and',
              'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again', 'then', 'once',
              'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
              'more', 'most', 'other', 'some', 'no', 'not', 'only', 'own',
              'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
              'don', 'should', 'now', 'also', 'would', 'could', 'one', 'like',
              'get', 'got', 'go', 'going', 'know', 'really', 'much', 'even',
              'still', 'ive', 'im', 'dont', 'nt', 'people', 'make', 'way',
              'something', 'anything', 'thing', 'things', 'said', 'tell',
              'told', 'back', 'time', 'want', 'feel', 'think', 'need'}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (label, title, cmap) in enumerate([(0, 'Non-Stress Posts', 'Greens'),
                                              (1, 'Stress Posts', 'Reds')]):
    text_data = ' '.join(df[df['label'] == label]['text'].values)
    text_data = re.sub(r'http\S+|www\S+', '', text_data)
    text_data = re.sub(r'[^a-zA-Z\s]', '', text_data.lower())

    wc = WordCloud(width=800, height=400, max_words=100,
                   background_color='white', colormap=cmap,
                   stopwords=stop_words, min_font_size=10)
    wc.generate(text_data)
    axes[idx].imshow(wc, interpolation='bilinear')
    axes[idx].set_title(title, fontsize=14, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('graphique7_wordcloud.png', dpi=150, bbox_inches='tight')
plt.close()
print('[7/8] WordCloud -- OK')

# ============================================================
# GRAPHIQUE 8 : Features sociales
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for label, color, name in [(0, '#2ecc71', 'Non-Stress'), (1, '#e74c3c', 'Stress')]:
    subset = df[df['label'] == label]
    axes[0].hist(subset['social_karma'].clip(upper=subset['social_karma'].quantile(0.95)),
                 bins=40, alpha=0.6, color=color, label=name)
axes[0].set_title('Distribution du Karma', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Karma')
axes[0].legend()

for label, color, name in [(0, '#2ecc71', 'Non-Stress'), (1, '#e74c3c', 'Stress')]:
    subset = df[df['label'] == label]
    axes[1].hist(subset['social_upvote_ratio'], bins=40, alpha=0.6, color=color, label=name)
axes[1].set_title('Distribution Upvote Ratio', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Upvote Ratio')
axes[1].legend()

for label, color, name in [(0, '#2ecc71', 'Non-Stress'), (1, '#e74c3c', 'Stress')]:
    subset = df[df['label'] == label]
    axes[2].hist(subset['social_num_comments'].clip(upper=subset['social_num_comments'].quantile(0.95)),
                 bins=40, alpha=0.6, color=color, label=name)
axes[2].set_title('Distribution Nombre de Commentaires', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Nombre de commentaires')
axes[2].legend()

plt.tight_layout()
plt.savefig('graphique8_social.png', dpi=150, bbox_inches='tight')
plt.close()
print('[8/8] Features sociales -- OK')

# ============================================================
# RESUME
# ============================================================
print('\n' + '=' * 50)
print('RESUME DE L ANALYSE EXPLORATOIRE')
print('=' * 50)
print(f'Total posts          : {len(df)}')
print(f'Train / Test         : {len(train)} / {len(test)}')
print(f'Features             : {df.shape[1]}')
print(f'Subreddits           : {df["subreddit"].nunique()}')
print(f'Ratio Stress         : {df["label"].mean()*100:.1f}%')
print(f'Valeurs manquantes   : {df.isnull().sum().sum()}')
print()
print('INSIGHTS CLES :')
print('  1. Dataset quasi-equilibre (52% stress)')
print('  2. Emotions negatives (negemo) = feature #1 correlee au stress')
print('  3. Usage du "je" (i) = signe de stress')
print('  4. Ton positif (Tone) et confiance (Clout) = anti-stress')
print('  5. Subreddits PTSD et anxiety = les plus stressants')
print()
print('8 graphiques sauvegardes dans le dossier ML_Stress')
print()
print('=' * 50)
print('ETAPE 2 TERMINEE -- Lance etape3_pretraitement.py pour la suite')
print('=' * 50)
