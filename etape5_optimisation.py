# ============================================================
# ETAPE 5 : OPTIMISATION DU MODELE (VERSION CORRIGEE WINDOWS)
# ============================================================
# Pour lancer : python etape5_optimisation.py
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

print('Chargement des donnees...')
X_train = pickle.load(open('X_train.pkl', 'rb'))
X_test = pickle.load(open('X_test.pkl', 'rb'))
y_train = pickle.load(open('y_train.pkl', 'rb'))
y_test = pickle.load(open('y_test.pkl', 'rb'))
results = pickle.load(open('results.pkl', 'rb'))
print('Donnees chargees.')

# ============================================================
# OPTIMISATION RANDOM FOREST
# ============================================================
print('\n' + '=' * 50)
print('OPTIMISATION RANDOM FOREST')
print('=' * 50)
print('Cela peut prendre 2-3 minutes...\n')

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_rf,
    cv=3,
    scoring='f1',
    n_jobs=1,
    verbose=1
)
grid_rf.fit(X_train, y_train)

print('\nRandom Forest optimise :')
print('  Meilleurs parametres :', grid_rf.best_params_)
print('  Meilleur F1 (CV)     : {:.2f}%'.format(grid_rf.best_score_ * 100))

# ============================================================
# OPTIMISATION XGBOOST (n_jobs=1 pour eviter le crash Windows)
# ============================================================
print('\n' + '=' * 50)
print('OPTIMISATION XGBOOST')
print('=' * 50)
print('Cela peut prendre 2-3 minutes...\n')

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1],
}

grid_xgb = GridSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False,
                  eval_metric='logloss', n_jobs=1),
    param_grid_xgb,
    cv=3,
    scoring='f1',
    n_jobs=1,
    verbose=1
)
grid_xgb.fit(X_train, y_train)

print('\nXGBoost optimise :')
print('  Meilleurs parametres :', grid_xgb.best_params_)
print('  Meilleur F1 (CV)     : {:.2f}%'.format(grid_xgb.best_score_ * 100))

# ============================================================
# EVALUER SUR LE TEST SET
# ============================================================
print('\n' + '=' * 70)
print('COMPARAISON AVANT / APRES OPTIMISATION')
print('=' * 70)

optimized_models = {
    'Random Forest (Optimise)': grid_rf.best_estimator_,
    'XGBoost (Optimise)': grid_xgb.best_estimator_
}

for name, model in optimized_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {
        'accuracy': acc, 'f1_score': f1, 'precision': prec,
        'recall': rec, 'auc_roc': auc, 'y_pred': y_pred, 'y_prob': y_prob
    }
    print('\n  {}'.format(name))
    print('  Accuracy  : {:.2f}%'.format(acc * 100))
    print('  F1-Score  : {:.2f}%'.format(f1 * 100))
    print('  Precision : {:.2f}%'.format(prec * 100))
    print('  Recall    : {:.2f}%'.format(rec * 100))
    print('  AUC-ROC   : {:.2f}%'.format(auc * 100))

best_overall = max(results, key=lambda x: results[x]['f1_score'])
print('\n' + '=' * 70)
print('MEILLEUR MODELE GLOBAL : {}'.format(best_overall))
print('  F1-Score : {:.2f}%'.format(results[best_overall]['f1_score'] * 100))
print('=' * 70)

# ============================================================
# GRAPHIQUE
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
compare_models = ['Random Forest', 'XGBoost',
                  'Random Forest (Optimise)', 'XGBoost (Optimise)']
compare_models = [m for m in compare_models if m in results]
x = np.arange(len(compare_models))
width = 0.25
colors_metrics = ['#3498db', '#2ecc71', '#e74c3c']
for i, (metric_name, key) in enumerate([('F1-Score', 'f1_score'),
                                          ('Accuracy', 'accuracy'),
                                          ('AUC-ROC', 'auc_roc')]):
    values = [results[m][key] * 100 for m in compare_models]
    bars = ax.bar(x + i * width, values, width, label=metric_name,
                  color=colors_metrics[i], alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                '{:.1f}'.format(val), ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(compare_models, fontsize=10, rotation=15)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Comparaison Avant / Apres Optimisation', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(50, 100)
plt.tight_layout()
plt.savefig('graphique13_avant_apres.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nGraphique sauvegarde.')

# ============================================================
# SAUVEGARDER
# ============================================================
if 'Random Forest' in best_overall:
    final_model = grid_rf.best_estimator_
else:
    final_model = grid_xgb.best_estimator_

pickle.dump(final_model, open('best_model.pkl', 'wb'))
pickle.dump(results, open('results.pkl', 'wb'))

print('\nFichiers sauvegardes :')
print('  best_model.pkl')
print('  results.pkl')
print('\n' + '=' * 50)
print('ETAPE 5 TERMINEE -- Lance etape6_test.py pour la suite')
print('=' * 50)
