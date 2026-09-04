"""
Mobility feature classification — age group
=============================================================================
Target  : age_group  (binned from continuous age using MATSim IVT 2015
          Switzerland age classes — Bösch et al. 2016, STRC)

          default 4-class (MATSim-aligned, MOBIS range 18-65):
              0_18-23   [18, 24)
              1_24-29   [24, 30)
              2_30-44   [30, 45)
              3_45-65   [45, 66)

          set use_3_class = True to collapse to 3 groups:
              0_Young   [18, 30)   (combines 18-24 + 24-30)
              1_Adult   [30, 45)
              2_Senior  [45, 66)

Models  : KNN · Random Forest · SVM (RBF) · XGBoost
CV      : 10-fold GridSearchCV (KNN / RF / SVM)
          12-candidate search + early stopping (XGBoost)
Split   : 80 % train  /  20 % test  (stratified)
Input   : feature_matrix_unnormalised.csv  

Reference: Bösch P., Becker F., Becker H., Axhausen K.W. (2016).
           The IVT 2015 Baseline Scenario. 16th Swiss Transport Research
           Conference. MATSim age classes: 18-24, 24-30, 30-45, 45-65.
=============================================================================
"""

from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import xgboost as xgb

warnings.filterwarnings('ignore')


# ============================================================================
# config
# ============================================================================
# feat_file   = Path('/data/baliu/thesis/02_merged_data/11_mobility_features/feature_matrix_unnormalised.csv')
feat_file = Path('/data/baliu/thesis/09_indicators/2_mobility_features_4weeks/feature_matrix_raw.csv')
survey_file = Path('/data/baliu/thesis/00data/sp_toponym_poi_purpose_demographics.csv')
out_dir     = Path('/data/baliu/thesis/00data/04_ml_classification/age')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'figures').mkdir(exist_ok=True)
(out_dir / 'models').mkdir(exist_ok=True)

user_col     = 'user_id'
random_state = 42
test_size    = 0.20

# set True to collapse 4 MATSim bins → 3 broader groups
use_3_class  = True

# ── MATSim IVT 2015 Switzerland age classes (Bösch et al. 2016) ──────────
# Full scheme: 6-15, 15-18, 18-24, 24-30, 30-45, 45-65, 65-80, 80+
# MOBIS covers 18-65, so we use the four bins that fall within that range.
# pd.cut uses right=False → [left, right)  i.e. 24 goes into 24-30, not 18-24

AGE_BINS_4 = {                     # default — MATSim IVT 2015 (Bösch et al. 2016)
    'bins'  : [18, 24, 30, 45, 66],
    'labels': ['0_18-24', '1_24-34', '2_30-44', '3_45-65'],
}

AGE_BINS_3 = {                     # collapsed 3-class (Young / Adult / Senior)
    'bins'  : [18, 25, 45, 66],
    'labels': ['0_18-24', '1_25-44', '2_45-65'],
}


# ============================================================================
# 1.  load & merge
# ============================================================================

def bin_age(age_series: pd.Series, cfg: dict) -> pd.Series:
    return pd.cut(age_series,
                  bins=cfg['bins'],
                  labels=cfg['labels'],
                  right=False)       # [left, right)


def load_and_merge():
    print("=" * 70)
    print(f"loading data  |  target: age_group  "
          f"({'3-class' if use_3_class else '4-class MATSim IVT 2015'})")
    print("=" * 70)

    feat_df = pd.read_csv(feat_file, dtype={user_col: str})
    feat_df[user_col] = feat_df[user_col].str.strip()
    print(f"  feature matrix : {feat_df.shape[0]} users x "
          f"{feat_df.shape[1]-1} features  [already in 0-1]")

    survey = pd.read_csv(survey_file, dtype={user_col: str})
    survey[user_col] = survey[user_col].str.strip()

    # one age value per user
    age_df = (survey.groupby(user_col)['age']
                     .first()
                     .reset_index())
    age_df['age'] = pd.to_numeric(age_df['age'], errors='coerce')
    age_df = age_df.dropna(subset=['age'])

    # distribution of raw age before binning
    print(f"\n  raw age — n={len(age_df)}  "
          f"min={age_df['age'].min():.0f}  "
          f"max={age_df['age'].max():.0f}  "
          f"mean={age_df['age'].mean():.1f}  "
          f"median={age_df['age'].median():.1f}")

    # bin into age groups
    cfg = AGE_BINS_3 if use_3_class else AGE_BINS_4
    age_df['age_group'] = bin_age(age_df['age'], cfg)
    age_df = age_df.dropna(subset=['age_group'])   # drops ages outside bin edges

    print(f"\n  age group distribution:")
    print(age_df['age_group'].value_counts().sort_index().to_string())

    merged = feat_df.merge(age_df[[user_col, 'age_group']],
                            on=user_col, how='inner')
    print(f"\n  merged : {len(merged)} users with features + age label")

    feature_cols = [c for c in feat_df.columns if c != user_col]
    X   = merged[feature_cols].values.astype(float)
    le  = LabelEncoder().fit(merged['age_group'].astype(str).values)
    y   = le.transform(merged['age_group'].astype(str).values)

    print(f"\n  classes : {list(le.classes_)}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, n in zip(le.classes_, counts):
        print(f"    {cls:<25} {n:>5}  ({n/len(y)*100:.1f}%)")

    return X, y, le, feature_cols


# ============================================================================
# 2.  split
# ============================================================================

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y)
    print(f"  train : {len(X_train):>5}  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  test  : {len(X_test):>5}  ({len(X_test)/len(X)*100:.1f}%)")
    return X_train, X_test, y_train, y_test


# ============================================================================
# shared helpers
# ============================================================================

def _print_cv(grid):
    print(f"\n  best params : {grid.best_params_}")
    std = grid.cv_results_['std_test_score'][grid.best_index_]
    print(f"  CV f1_macro : {grid.best_score_:.4f}  (+-{std:.4f})")
    cv_df = pd.DataFrame(grid.cv_results_)
    print("\n  top 3 configurations:")
    for _, row in cv_df.nsmallest(3, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score']].iterrows():
        print(f"    {row['params']}")
        print(f"      CV: {row['mean_test_score']:.4f} (+-{row['std_test_score']:.4f})")


def _print_test(y_train, y_pred_tr, y_test, y_pred_te,
                cv_score, cv_label='CV f1_macro'):
    tr_acc = accuracy_score(y_train, y_pred_tr)
    te_acc = accuracy_score(y_test,  y_pred_te)
    te_f1  = f1_score(y_test, y_pred_te, average='macro')
    gap    = tr_acc - te_acc
    print(f"\n  train accuracy : {tr_acc:.4f}")
    print(f"  test  accuracy : {te_acc:.4f}")
    print(f"  test  f1_macro : {te_f1:.4f}")
    print(f"  {cv_label:<18}: {cv_score:.4f}")
    if gap > 0.10:
        print(f"  WARNING overfitting  (gap: {gap:.4f})")
    else:
        print(f"  OK generalises well  (gap: {gap:.4f})")


# ============================================================================
# 3.  KNN
# ============================================================================

def train_knn(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 70)
    print("KNN -- 10-fold CV  (features already in [0,1], no scaling)")
    print("=" * 70)

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 19, 25],
        'weights'    : ['uniform', 'distance'],
        'metric'     : ['euclidean', 'manhattan', 'minkowski'],
    }
    n = (len(param_grid['n_neighbors']) *
         len(param_grid['weights']) *
         len(param_grid['metric']))
    print(f"  {n} combinations x 10 folds = {n*10} fits")

    grid = GridSearchCV(KNeighborsClassifier(), param_grid,
                        cv=10, scoring='f1_macro',
                        n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    _print_cv(grid)

    best   = grid.best_estimator_
    y_pred = best.predict(X_test)
    _print_test(y_train, best.predict(X_train),
                y_test, y_pred, grid.best_score_)
    return best, y_pred


# ============================================================================
# 4.  Random Forest
# ============================================================================

def train_rf(X_train, y_train, X_test, y_test, feature_cols):
    print("\n" + "=" * 70)
    print("Random Forest -- 10-fold CV  (RandomizedSearchCV n_iter=50)")
    print("=" * 70)

    param_grid = {
        'n_estimators'     : [100, 200, 300],
        'max_depth'        : [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
        'max_features'     : ['sqrt', 'log2'],
    }
    grid = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced',
                                random_state=random_state, n_jobs=-1),
        param_grid, n_iter=50, cv=10, scoring='f1_macro',
        n_jobs=-1, random_state=random_state, return_train_score=True)
    grid.fit(X_train, y_train)
    _print_cv(grid)

    best   = grid.best_estimator_
    y_pred = best.predict(X_test)
    _print_test(y_train, best.predict(X_train),
                y_test, y_pred, grid.best_score_)

    feat_imp = pd.DataFrame({
        'feature'   : feature_cols,
        'importance': best.feature_importances_,
    }).sort_values('importance', ascending=False)
    print("\n  top 15 features:")
    print(feat_imp.head(15).to_string(index=False))
    return best, feat_imp, y_pred


# ============================================================================
# 5.  SVM
# ============================================================================

def train_svm(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 70)
    print("SVM (RBF) -- 10-fold CV  (features already in [0,1], no scaling)")
    print("=" * 70)

    param_grid = {
        'C'     : [0.1, 1, 10, 100],
        'gamma' : ['scale', 'auto', 0.01, 0.001],
        'kernel': ['rbf'],
    }
    n = len(param_grid['C']) * len(param_grid['gamma'])
    print(f"  {n} combinations x 10 folds = {n*10} fits")

    grid = GridSearchCV(
        SVC(class_weight='balanced', probability=True,
            random_state=random_state),
        param_grid, cv=10, scoring='f1_macro',
        n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    _print_cv(grid)

    best   = grid.best_estimator_
    y_pred = best.predict(X_test)
    _print_test(y_train, best.predict(X_train),
                y_test, y_pred, grid.best_score_)
    return best, y_pred


# ============================================================================
# 6.  XGBoost
# ============================================================================

def _weights(y):
    classes, counts = np.unique(y, return_counts=True)
    wmap = dict(zip(classes, len(y) / (len(classes) * counts)))
    return np.array([wmap[yi] for yi in y])


def train_xgboost(X_train, y_train, X_test, y_test, feature_cols):
    print("\n" + "=" * 70)
    print("XGBoost -- 12-candidate search + early stopping")
    print("=" * 70)

    n_classes = len(np.unique(y_train))

    candidates = [
        (3, 0.05, 0.8, 0.8), (3, 0.1,  0.8, 0.8),
        (4, 0.05, 0.8, 0.8), (4, 0.1,  0.8, 0.8),
        (5, 0.05, 0.8, 0.8), (5, 0.1,  0.8, 0.8),
        (6, 0.05, 0.8, 0.8), (6, 0.1,  0.8, 0.8),
        (4, 0.05, 0.7, 0.7), (4, 0.1,  0.7, 0.7),
        (5, 0.05, 0.7, 0.7), (5, 0.1,  0.7, 0.7),
    ]

    # 15 % of training data as validation for early stopping (test stays locked)
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, _weights(y_train),
        test_size=0.15, random_state=random_state, stratify=y_train)
    print(f"  train: {len(X_tr)}  val: {len(X_val)}  test: {len(X_test)} (locked)")

    best_f1, best_model, best_params, log = -1.0, None, None, []

    for md, lr, ss, cs in candidates:
        m = xgb.XGBClassifier(
            objective='multi:softprob', num_class=n_classes,
            eval_metric='mlogloss',
            max_depth=md, learning_rate=lr, n_estimators=1000,
            subsample=ss, colsample_bytree=cs,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=random_state, tree_method='hist',
            early_stopping_rounds=20)
        m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              sample_weight_eval_set=[w_val],
              verbose=False)

        f1v = f1_score(y_val, m.predict(X_val), average='macro')
        log.append({'depth': md, 'lr': lr, 'sub': ss, 'col': cs,
                    'iter': m.best_iteration, 'f1_val': f1v})
        if f1v > best_f1:
            best_f1, best_model = f1v, m
            best_params = {'max_depth': md, 'learning_rate': lr,
                           'subsample': ss, 'colsample_bytree': cs}

    print(f"\n  best params : {best_params}")
    print(f"  best val f1 : {best_f1:.4f}  (iter: {best_model.best_iteration})")
    res = pd.DataFrame(log).sort_values('f1_val', ascending=False)
    print("\n  top 3:")
    for _, r in res.head(3).iterrows():
        print(f"    depth={r['depth']} lr={r['lr']} sub={r['sub']} col={r['col']}"
              f"  -->  val f1: {r['f1_val']:.4f}  iter: {r['iter']}")

    # retrain on full training set
    print("\n  retraining on full training set ...")
    final = xgb.XGBClassifier(
        objective='multi:softprob', num_class=n_classes,
        eval_metric='mlogloss',
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        n_estimators=best_model.best_iteration + 1,
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=random_state, tree_method='hist')
    final.fit(X_train, y_train,
              sample_weight=_weights(y_train), verbose=False)

    y_pred = final.predict(X_test)
    _print_test(y_train, final.predict(X_train),
                y_test, y_pred, best_f1, cv_label='val  f1_macro')

    feat_imp = pd.DataFrame({
        'feature'   : feature_cols,
        'importance': final.feature_importances_,
    }).sort_values('importance', ascending=False)
    print("\n  top 10 features:")
    print(feat_imp.head(10).to_string(index=False))
    return final, feat_imp, y_pred


# ============================================================================
# 7.  evaluation
# ============================================================================

def _confusion_plot(y_true, y_pred, labels, title, path):
    cm   = confusion_matrix(y_true, y_pred)
    cmap = LinearSegmentedColormap.from_list(
        'c', ['#2d1b3d', '#3a4f8f', '#1f968b', '#73d055', '#fde724'], N=100)
    plt.figure(figsize=(max(7, len(labels) + 3), max(6, len(labels) + 2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor='white', square=True,
                annot_kws={'size': 12, 'weight': 'bold'})
    acc = accuracy_score(y_true, y_pred)
    plt.title(f'{title}  (n={len(y_true)}, acc={acc:.3f})',
              fontsize=13, fontweight='bold', pad=14)
    plt.xlabel('predicted', fontsize=11, fontweight='bold')
    plt.ylabel('true',      fontsize=11, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def detailed_evaluation(models_dict, X_test, y_test, le):
    print("\n" + "=" * 70)
    print("detailed evaluation -- test set")
    print("=" * 70)
    for name, info in models_dict.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        y_pred = info['model'].predict(X_test)
        print(classification_report(y_test, y_pred,
                                    target_names=le.classes_, digits=4))
        cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred),
                              index=le.classes_, columns=le.classes_)
        print(cm_df.to_string())
        slug = name.lower().replace(' ', '_')
        _confusion_plot(y_test, y_pred, le.classes_, name,
                        out_dir / 'figures' / f'confusion_{slug}.png')
        cm_df.to_csv(out_dir / f'confusion_{slug}.csv')
        print(f"  saved: confusion_{slug}.png + .csv")


def print_comparison(models_dict, X_test, y_test, le):
    print("\n" + "=" * 70)
    print("model comparison summary")
    print("=" * 70)
    rows = []
    for name, info in models_dict.items():
        y_pred = info['model'].predict(X_test)
        # per-class F1
        per_class = f1_score(y_test, y_pred, average=None)
        row = {
            'model'      : name,
            'accuracy'   : round(accuracy_score(y_test, y_pred),               4),
            'f1_macro'   : round(f1_score(y_test, y_pred, average='macro'),    4),
            'f1_weighted': round(f1_score(y_test, y_pred, average='weighted'), 4),
        }
        for cls, f1 in zip(le.classes_, per_class):
            row[f'f1_{cls}'] = round(f1, 4)
        rows.append(row)

    comp = pd.DataFrame(rows).sort_values('f1_macro', ascending=False)
    print(comp.to_string(index=False))
    comp.to_csv(out_dir / 'model_comparison.csv', index=False)
    best = comp.iloc[0]
    print(f"\n  best: {best['model']}  (f1_macro: {best['f1_macro']:.4f})")
    return comp


def plot_comparison_bar(comp_df, le):
    base_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    class_metrics = [f'f1_{c}' for c in le.classes_]
    all_metrics   = base_metrics + class_metrics
    colors = ['#2196F3', '#4CAF50', '#9E9E9E',
              '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']

    x = np.arange(len(comp_df))
    w = 0.80 / len(all_metrics)

    fig, ax = plt.subplots(figsize=(max(11, len(comp_df) * 3), 5))
    for i, (metric, color) in enumerate(zip(all_metrics, colors)):
        if metric not in comp_df.columns:
            continue
        offset = (i - len(all_metrics) / 2 + 0.5) * w
        bars   = ax.bar(x + offset, comp_df[metric], w,
                        label=metric, color=color, alpha=0.85, edgecolor='white')
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.005,
                    f'{b.get_height():.3f}',
                    ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['model'], fontsize=11)
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.12)
    ax.set_title('model comparison — age group prediction',
                  fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'model_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: model_comparison.png")


def plot_feature_importance(rf_imp, xgb_imp, n_top=20):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, imp, title in zip(axes,
                               [rf_imp, xgb_imp],
                               ['Random Forest', 'XGBoost']):
        top = imp.head(n_top).sort_values('importance')
        ax.barh(top['feature'], top['importance'],
                color='#2196F3', alpha=0.85, edgecolor='white')
        ax.set_title(f'{title} — feature importance  (target: age group)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('importance')
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: feature_importance.png")


def plot_age_distribution(survey_file):
    """Extra plot: show raw age distribution with bin boundaries."""
    survey = pd.read_csv(survey_file, dtype={user_col: str})
    ages   = pd.to_numeric(
        survey.groupby(user_col)['age'].first(), errors='coerce').dropna()

    cfg    = AGE_BINS_3 if use_3_class else AGE_BINS_4
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E63946']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # histogram with bin boundaries
    ax = axes[0]
    ax.hist(ages, bins=30, color='#BBDEFB', edgecolor='white', alpha=0.9)
    for edge in cfg['bins'][1:-1]:
        ax.axvline(edge, color='#E63946', linestyle='--', lw=1.4,
                   label=f'split at {edge}')
    ax.set_xlabel('age', fontsize=10)
    ax.set_ylabel('users', fontsize=10)
    ax.set_title('raw age distribution with bin boundaries',
                  fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # bar chart of binned counts
    ax = axes[1]
    binned = bin_age(ages, cfg).value_counts().sort_index()
    bars = ax.bar(range(len(binned)), binned.values,
                  color=colors[:len(binned)], edgecolor='white', width=0.7)
    for bar, val in zip(bars, binned.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f'{val}\n({val/len(ages)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(binned)))
    ax.set_xticklabels(binned.index, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('users', fontsize=10)
    ax.set_title('users per age group', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Age binning overview', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'age_distribution.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: age_distribution.png")


# ============================================================================
# 8.  save models
# ============================================================================

def save_models(models_dict, le):
    for name, info in models_dict.items():
        slug = name.lower().replace(' ', '_')
        with open(out_dir / 'models' / f'{slug}_model.pkl', 'wb') as f:
            pickle.dump(info['model'], f)
        print(f"  saved: {slug}_model.pkl")
    with open(out_dir / 'models' / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("  saved: label_encoder.pkl")


# ============================================================================
# main
# ============================================================================

def main():
    print("=" * 70)
    print("mobility feature classification — age group")
    print(f"bins   : {'3-class (Young/Adult/Senior)' if use_3_class else '4-class MATSim IVT 2015 (18-23 · 24-29 · 30-44 · 45-65)'}")
    print("ref    : Bösch et al. 2016 — IVT 2015 MATSim Switzerland baseline")
    print("input  : features already MinMax normalised — no rescaling")
    print("=" * 70)

    X, y, le, feature_cols = load_and_merge()

    print("\n" + "=" * 70)
    print("train / test split  (80/20 stratified)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = split_data(X, y)

    knn_model,              knn_pred  = train_knn(X_train, y_train, X_test, y_test)
    rf_model,  rf_imp,      rf_pred   = train_rf( X_train, y_train, X_test, y_test, feature_cols)
    svm_model,              svm_pred  = train_svm(X_train, y_train, X_test, y_test)
    xgb_model, xgb_imp,     xgb_pred  = train_xgboost(X_train, y_train, X_test, y_test, feature_cols)

    models_dict = {
        'KNN'          : {'model': knn_model},
        'Random Forest': {'model': rf_model},
        'SVM'          : {'model': svm_model},
        'XGBoost'      : {'model': xgb_model},
    }

    detailed_evaluation(models_dict, X_test, y_test, le)
    comp_df = print_comparison(models_dict, X_test, y_test, le)

    print("\n" + "=" * 70)
    print("generating plots")
    print("=" * 70)
    plot_age_distribution(survey_file)
    plot_comparison_bar(comp_df, le)
    plot_feature_importance(rf_imp, xgb_imp)

    rf_imp.to_csv( out_dir / 'feature_importance_rf.csv',  index=False)
    xgb_imp.to_csv(out_dir / 'feature_importance_xgb.csv', index=False)
    print("  saved: feature_importance_rf.csv  &  feature_importance_xgb.csv")

    print("\n" + "=" * 70)
    print("saving models")
    print("=" * 70)
    save_models(models_dict, le)

    print("\n" + "=" * 70)
    print("done")
    print("=" * 70)
    print(f"  outputs: {out_dir}")


if __name__ == '__main__':
    main()
