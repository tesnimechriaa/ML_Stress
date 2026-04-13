# ============================================================
# ETAPE 4 : ENTRAINEMENT DES MODELES
# ============================================================
# But : Entrainer 4 modeles ML differents et les comparer.
#
# Les 4 modeles :
#   1. Logistic Regression -- le plus simple, la baseline
#   2. Random Forest -- un vote de plusieurs arbres de decision
#   3. SVM -- trouve la meilleure frontiere entre les 2 classes
#   4. XGBoost -- le champion des competitions ML
#
# Pour lancer : python etape4_modeles.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# 4.1 - Charger les donnees preparees (de l'etape 3)
# ============================================================
print('Chargement des donnees preparees...')
X_train = pickle.load(open('X_train.pkl', 'rb'))
X_test = pickle.load(open('X_test.pkl', 'rb'))
y_train = pickle.load(open('y_train.pkl', 'rb'))
y_test = pickle.load(open('y_test.pkl', 'rb'))

print('  X_train :', X_train.shape)
print('  X_test  :', X_test.shape)
print('  y_train :', y_train.shape)
print('  y_test  :', y_test.shape)

# ============================================================
# 4.2 - Definir les 4 modeles
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        C=1.0
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

print('\n{} modeles definis :'.format(len(models)))
for name in models:
    print('  -', name)

# ============================================================
# 4.3 - Entrainer et evaluer chaque modele
# ============================================================
results = {}

print('\n' + '=' * 70)
print('ENTRAINEMENT DES MODELES')
print('=' * 70)

for name, model in models.items():
    print('\n' + '-' * 70)
    print('Modele : {}'.format(name))
    print('-' * 70)

    # 1. Entrainer
    start_time = time.time()
    print('  Entrainement en cours...')
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print('  Entraine en {:.1f} secondes'.format(train_time))

    # 2. Predire
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 3. Calculer les metriques
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'accuracy': acc,
        'f1_score': f1,
        'precision': prec,
        'recall': rec,
        'auc_roc': auc,
        'train_time': train_time,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print('\n  RESULTATS :')
    print('  Accuracy  : {:.2f}%'.format(acc * 100))
    print('  F1-Score  : {:.2f}%'.format(f1 * 100))
    print('  Precision : {:.2f}%'.format(prec * 100))
    print('  Recall    : {:.2f}%'.format(rec * 100))
    print('  AUC-ROC   : {:.2f}%'.format(auc * 100))

# ============================================================
# 4.4 - Tableau comparatif
# ============================================================
print('\n' + '=' * 70)
print('TABLEAU COMPARATIF DES MODELES')
print('=' * 70)

model_names = list(results.keys())

print('\n{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
    'Modele', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC-ROC'))
print('-' * 75)
for name in model_names:
    r = results[name]
    print('{:<25s} {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>9.2f}%'.format(
        name, r['accuracy']*100, r['f1_score']*100, r['precision']*100,
        r['recall']*100, r['auc_roc']*100))

best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_f1 = results[best_model_name]['f1_score']
print('\nMEILLEUR MODELE : {} (F1 = {:.2f}%)'.format(best_model_name, best_f1 * 100))

# ============================================================
# 4.5 - GRAPHIQUE : Comparaison visuelle des modeles
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

metrics_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC-ROC']
x = np.arange(len(metrics_names))
width = 0.18
colors_models = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, (name, color) in enumerate(zip(model_names, colors_models)):
    values = [results[name]['accuracy'], results[name]['f1_score'],
              results[name]['precision'], results[name]['recall'],
              results[name]['auc_roc']]
    axes[0].bar(x + i * width, [v * 100 for v in values], width,
                label=name, color=color, alpha=0.85)

axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(metrics_names, fontsize=11)
axes[0].set_ylabel('Score (%)', fontsize=12)
axes[0].set_title('Comparaison des Modeles', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_ylim(50, 100)

f1_scores = [results[name]['f1_score'] * 100 for name in model_names]
bars = axes[1].bar(model_names, f1_scores, color=colors_models, alpha=0.85)
for bar, val in zip(bars, f1_scores):
    axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 '{:.1f}%'.format(val), ha='center', fontweight='bold', fontsize=12)
axes[1].set_title('F1-Score par Modele', fontsize=14, fontweight='bold')
axes[1].set_ylabel('F1-Score (%)', fontsize=12)
axes[1].set_ylim(50, 100)

plt.tight_layout()
plt.savefig('graphique9_comparaison_modeles.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n[1/3] Graphique comparaison -- OK')

# ============================================================
# 4.6 - Matrices de confusion
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for idx, (name, color) in enumerate(zip(model_names, colors_models)):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Non-Stress', 'Stress'],
                yticklabels=['Non-Stress', 'Stress'])
    axes[idx].set_title(name, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Reel' if idx == 0 else '')
    axes[idx].set_xlabel('Predit')

plt.suptitle('Matrices de Confusion', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphique10_matrices_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print('[2/3] Matrices de confusion -- OK')

# ============================================================
# 4.7 - Courbes ROC
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for name, color in zip(model_names, colors_models):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc = results[name]['auc_roc']
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label='{} (AUC = {:.3f})'.format(name, auc))

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Hasard (AUC = 0.500)')
ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
ax.set_title('Courbes ROC -- Comparaison des Modeles', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphique11_courbes_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print('[3/3] Courbes ROC -- OK')

# ============================================================
# 4.8 - Classification Report du meilleur modele
# ============================================================
print('\n' + '=' * 50)
print('RAPPORT DETAILLE -- {}'.format(best_model_name))
print('=' * 50)
print(classification_report(
    y_test,
    results[best_model_name]['y_pred'],
    target_names=['Non-Stress', 'Stress']
))

# ============================================================
# 4.9 - Feature Importance (Random Forest)
# ============================================================
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
numeric_features = pickle.load(open('numeric_features.pkl', 'rb'))

tfidf_names = list(tfidf.get_feature_names_out())
all_feature_names = tfidf_names + numeric_features

rf_model = models['Random Forest']
importances = rf_model.feature_importances_

top_n = 25
top_indices = importances.argsort()[-top_n:][::-1]

fig, ax = plt.subplots(figsize=(12, 8))

top_names = [all_feature_names[i].replace('lex_liwc_', 'LIWC:')
             for i in top_indices]
top_values = importances[top_indices]

colors_fi = ['#e74c3c' if 'LIWC:' in n or 'social_' in n or 'sentiment' in n
             else '#3498db' for n in top_names]

ax.barh(range(top_n), top_values[::-1], color=colors_fi[::-1])
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_names[::-1], fontsize=10)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 25 Features les plus Importantes (Random Forest)',
             fontsize=14, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', label='Feature Textuelle (TF-IDF)'),
                   Patch(facecolor='#e74c3c', label='Feature Numerique (LIWC/Social)')]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig('graphique12_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Feature Importance -- OK')

# ============================================================
# 4.10 - Sauvegarder les modeles
# ============================================================
pickle.dump(models, open('all_models.pkl', 'wb'))
pickle.dump(results, open('results.pkl', 'wb'))
print('\nModeles et resultats sauvegardes.')

# ============================================================
# RESUME
# ============================================================
print('\n' + '=' * 50)
print('RESUME -- ENTRAINEMENT DES MODELES')
print('=' * 50)
for name in model_names:
    r = results[name]
    flag = ' <-- MEILLEUR' if name == best_model_name else ''
    print('  {:<25s} F1: {:.1f}%  Acc: {:.1f}%  AUC: {:.1f}%{}'.format(
        name, r['f1_score']*100, r['accuracy']*100, r['auc_roc']*100, flag))

print('\n4 graphiques sauvegardes (9, 10, 11, 12)')
print()
print('=' * 50)
print('ETAPE 4 TERMINEE -- Lance etape5_optimisation.py pour la suite')
print('=' * 50)
