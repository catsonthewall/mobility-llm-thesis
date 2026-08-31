"""
Age group correlation analysis — pre-built normalised feature matrix
=============================================================================
Input    : feature_matrix_normalised.csv  (already MinMax [0,1])
           sp_toponym_poi_purpose_demographics.csv  (age ground truth)

Target   : age_group — 4-class MATSim IVT 2015 (Bösch et al. 2016)
               0_18-24  [18, 24)
               1_24-30  [24, 30)
               2_30-45  [30, 45)
               3_45-65  [45, 66)
           Set USE_3_CLASS = True to collapse to 3 groups.

Step 1   : Spearman correlation  (every feature ↔ ordinal age_group label)
           Spearman is used instead of point-biserial because the age
           target is ordinal (ordered bins), not binary.
Step 2   : Correlation plots (bar, heatmap, violin distributions)
Step 3   : Run all 7 classifiers using
               KNN · Random Forest · SVM · XGBoost          (original mls 4)
               Logistic Regression · LDA · Ridge Classifier  (linear models)
               — full feature set       (baseline = age_group_prediction_pipeline.py)
               — top-1 correlated feature only
               — top-2 correlated features only
Step 4   : Comparison table + plot (full vs top-1 vs top-2)
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.stats import spearmanr

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score
)
import xgboost as xgb

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration 
# ============================================================================
feat_file    = Path('/data/baliu/thesis/00data/03_features_built_17may/feature_matrix_normalised.csv')
survey_file  = Path('/data/baliu/thesis/00data/sp_toponym_poi_purpose_demographics.csv')
out_dir      = Path('/data/baliu/thesis/04_method_corr/04_correlation_results/age_correlation')

user_col     = 'user_id'
random_state = 42
test_size    = 0.20

# MATSim IVT 2015 Switzerland age classes (Bösch et al. 2016)
AGE_BINS_4 = {
    'bins'  : [18, 24, 30, 45, 66],
    'labels': ['0_18-24', '1_24-30', '2_30-45', '3_45-65'],
}
AGE_BINS_3 = {
    'bins'  : [18, 30, 45, 66],
    'labels': ['0_18-29', '1_30-44', '2_45-65'],
}

# Set True to collapse 4 MATSim bins -> 3 broader groups
USE_3_CLASS  = False

# ============================================================================
# 1.  Load pre-built features + age ground truth
# ============================================================================

def bin_age(age_series, cfg):
    return pd.cut(age_series, bins=cfg['bins'],
                  labels=cfg['labels'], right=False)


def load_data():
    print("=" * 70)
    cfg_label = '3-class' if USE_3_CLASS else '4-class MATSim IVT 2015'
    print(f"Loading pre-built feature matrix  |  target: age_group ({cfg_label})")
    print("=" * 70)

    # Feature matrix (already MinMax normalised — no rescaling)
    feat_df = pd.read_csv(feat_file, dtype={user_col: str})
    feat_df[user_col] = feat_df[user_col].str.strip()
    feat_cols = [c for c in feat_df.columns if c != user_col]
    print(f"  Feature matrix : {feat_df.shape[0]} users x "
          f"{len(feat_cols)} features  [MinMax 0-1, no rescaling]")

    # Survey / age ground truth
    survey = pd.read_csv(survey_file, dtype={user_col: str})
    survey[user_col] = survey[user_col].str.strip()
    if user_col not in survey.columns and 'participant_ID' in survey.columns:
        survey = survey.rename(columns={'participant_ID': user_col})

    age_df = (survey.groupby(user_col)['age']
                     .first()
                     .reset_index())
    age_df['age'] = pd.to_numeric(age_df['age'], errors='coerce')
    age_df = age_df.dropna(subset=['age'])

    print(f"\n  Raw age — n={len(age_df)}  "
          f"min={age_df['age'].min():.0f}  "
          f"max={age_df['age'].max():.0f}  "
          f"mean={age_df['age'].mean():.1f}  "
          f"median={age_df['age'].median():.1f}")

    cfg = AGE_BINS_3 if USE_3_CLASS else AGE_BINS_4
    age_df['age_group'] = bin_age(age_df['age'], cfg)
    age_df = age_df.dropna(subset=['age_group'])

    print("\n  Age group distribution:")
    print(age_df['age_group'].value_counts().sort_index().to_string())

    # Merge
    merged = feat_df.merge(age_df[[user_col, 'age_group']],
                            on=user_col, how='inner')
    print(f"\n  Merged : {len(merged)} users with features + age label")

    le = LabelEncoder().fit(merged['age_group'].astype(str).values)
    merged['age_enc'] = le.transform(merged['age_group'].astype(str).values)

    print(f"\n  Classes : {list(le.classes_)}")
    unique, counts = np.unique(merged['age_enc'].values, return_counts=True)
    for cls, n in zip(le.classes_, counts):
        print(f"    {cls:<25} {n:>5}  ({n / len(merged) * 100:.1f}%)")

    return merged, feat_cols, le


# ============================================================================
# 2.  Spearman correlation  (feature <-> ordinal age_group index)
#     Spearman is appropriate here because age_group is ordered (0 < 1 < 2 < 3)
#     but the distances between bins are not equal — hence rank-based, not linear.
# ============================================================================

def compute_correlations(merged, feat_cols):
    print("\n" + "=" * 70)
    print("Spearman correlation  (feature <-> ordinal age_group)")
    print("=" * 70)

    y = merged['age_enc'].values
    records = []
    for col in feat_cols:
        x = merged[col].values
        if x.std() == 0:
            continue
        r, p = spearmanr(x, y)
        records.append({'feature': col,
                        'r'      : round(r, 6),
                        'abs_r'  : abs(r),
                        'p_value': round(p, 6)})

    corr_df = (pd.DataFrame(records)
                 .sort_values('abs_r', ascending=False)
                 .reset_index(drop=True))

    print(f"\n  Top 20 features by |r|:")
    print(corr_df.head(20)[['feature', 'r', 'p_value']].to_string(index=False))

    corr_df.to_csv(out_dir / 'age_feature_correlation.csv', index=False)
    print("\n  saved: age_feature_correlation.csv")
    return corr_df


# ============================================================================
# 3.  Correlation plots
# ============================================================================

def plot_correlation_bar(corr_df, n_top=20):
    """Bar chart — top-N features by |Spearman r|, coloured by sign."""
    top    = corr_df.head(n_top).sort_values('abs_r')
    # positive r -> older age groups, negative r -> younger age groups
    colors = ['#FF9800' if r > 0 else '#2196F3' for r in top['r']]

    fig, ax = plt.subplots(figsize=(9, max(6, n_top * 0.38)))
    bars = ax.barh(top['feature'], top['abs_r'],
                   color=colors, alpha=0.85, edgecolor='white')
    for bar, r_val in zip(bars, top['r']):
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f'{r_val:+.3f}', va='center', fontsize=8)

    ax.set_xlabel('|Spearman r|', fontsize=11)
    ax.set_title(
        f'Top {n_top} features correlated with age group  (Spearman)\n'
        f'(orange = r > 0 -> older bins,  blue = r < 0 -> younger bins)',
        fontsize=11, fontweight='bold')
    ax.set_xlim(0, corr_df['abs_r'].max() * 1.20)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'correlation_bar_age.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: correlation_bar_age.png")


def plot_correlation_heatmap(merged, feat_cols, corr_df, n_top=25):
    """Seaborn heatmap — top-N features + age_enc label."""
    top_feats = corr_df.head(n_top)['feature'].tolist()
    sub = (merged[top_feats + ['age_enc']]
           .rename(columns={'age_enc': 'age_group_ord'}))
    corr_matrix = sub.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(max(10, n_top * 0.5),
                                     max(9,  n_top * 0.45)))
    sns.heatmap(corr_matrix, ax=ax,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                center=0, annot=False,
                linewidths=0.4, linecolor='white')
    ax.set_title(f'Spearman correlation heatmap  '
                 f'(top {n_top} features by |r| with age group)',
                 fontsize=11, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'correlation_heatmap_age.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: correlation_heatmap_age.png")


def plot_top_feature_distributions(merged, corr_df, le, n=2):
    """Violin plots of top-N features split by age group."""
    top_feats = corr_df.head(n)['feature'].tolist()
    n_classes = len(le.classes_)
    palette   = dict(zip(le.classes_,
                         sns.color_palette('viridis', n_classes)))

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    merged_plot = merged.copy()
    merged_plot['age_group'] = le.inverse_transform(
        merged_plot['age_enc'].values)

    for ax, feat in zip(axes, top_feats):
        r_val = corr_df.loc[corr_df['feature'] == feat, 'r'].values[0]
        sns.violinplot(data=merged_plot, x='age_group', y=feat,
                       order=le.classes_,
                       palette=palette, inner='box', ax=ax, alpha=0.75)
        ax.set_title(f'{feat}\n(Spearman r = {r_val:+.3f})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('age group', fontsize=9)
        ax.set_ylabel('normalised value [0-1]', fontsize=9)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Top correlated feature distributions by age group',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'top_feature_distributions_age.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: top_feature_distributions_age.png")


# ============================================================================
# 4.  Shared helpers
# ============================================================================

def _print_result(y_train, y_pred_tr, y_test, y_pred_te,
                  cv_score, cv_label='CV f1_macro'):
    tr_acc = accuracy_score(y_train, y_pred_tr)
    te_acc = accuracy_score(y_test,  y_pred_te)
    te_f1  = f1_score(y_test, y_pred_te, average='macro')
    gap    = tr_acc - te_acc
    flag   = '  WARNING overfit' if gap > 0.10 else '  OK'
    print(f"    train={tr_acc:.4f}  test={te_acc:.4f}  "
          f"f1={te_f1:.4f}  {cv_label}={cv_score:.4f}  gap={gap:.3f}{flag}")


def _confusion_plot(y_true, y_pred, labels, title, path):
    cm   = confusion_matrix(y_true, y_pred)
    cmap = LinearSegmentedColormap.from_list(
        'c', ['#2d1b3d', '#3a4f8f', '#1f968b', '#73d055', '#fde724'], N=100)
    size = max(6, len(labels) + 2)
    plt.figure(figsize=(size, size - 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor='white', square=True,
                annot_kws={'size': 11, 'weight': 'bold'})
    acc = accuracy_score(y_true, y_pred)
    plt.title(f'{title}\n(n={len(y_true)}, acc={acc:.3f})',
              fontsize=10, fontweight='bold', pad=10)
    plt.xlabel('predicted', fontsize=9)
    plt.ylabel('true',      fontsize=9)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# 5.  Individual model trainers
#     Features already [0,1] — no extra scaling needed
# ============================================================================

def _knn(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 19, 25],
        'weights'    : ['uniform', 'distance'],
        'metric'     : ['euclidean', 'manhattan', 'minkowski'],
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid,
                        cv=10, scoring='f1_macro', n_jobs=-1,
                        return_train_score=True)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    yp   = best.predict(X_test)
    print(f"    best params : {grid.best_params_}")
    _print_result(y_train, best.predict(X_train), y_test, yp, grid.best_score_)
    return yp


def _rf(X_train, y_train, X_test, y_test):
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
    best = grid.best_estimator_
    yp   = best.predict(X_test)
    print(f"    best params : {grid.best_params_}")
    _print_result(y_train, best.predict(X_train), y_test, yp, grid.best_score_)
    return yp


def _svm(X_train, y_train, X_test, y_test):
    param_grid = {
        'C'     : [0.1, 1, 10, 100],
        'gamma' : ['scale', 'auto', 0.01, 0.001],
        'kernel': ['rbf'],
    }
    grid = GridSearchCV(
        SVC(class_weight='balanced', probability=True,
            random_state=random_state),
        param_grid, cv=10, scoring='f1_macro',
        n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    yp   = best.predict(X_test)
    print(f"    best params : {grid.best_params_}")
    _print_result(y_train, best.predict(X_train), y_test, yp, grid.best_score_)
    return yp


def _xgboost(X_train, y_train, X_test, y_test):
    n_classes   = len(np.unique(y_train))
    # always multi:softprob for age (4-class or 3-class)
    objective   = 'multi:softprob'
    eval_metric = 'mlogloss'

    classes, counts = np.unique(y_train, return_counts=True)
    wmap  = dict(zip(classes, len(y_train) / (len(classes) * counts)))
    w_all = np.array([wmap[yi] for yi in y_train])

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, w_all,
        test_size=0.15, random_state=random_state, stratify=y_train)

    candidates = [
        (3,0.05,0.8,0.8),(3,0.1,0.8,0.8),(4,0.05,0.8,0.8),(4,0.1,0.8,0.8),
        (5,0.05,0.8,0.8),(5,0.1,0.8,0.8),(6,0.05,0.8,0.8),(6,0.1,0.8,0.8),
        (4,0.05,0.7,0.7),(4,0.1,0.7,0.7),(5,0.05,0.7,0.7),(5,0.1,0.7,0.7),
    ]
    best_f1, best_m, best_p = -1.0, None, None
    for md, lr, ss, cs in candidates:
        kw = dict(objective=objective, eval_metric=eval_metric,
                  num_class=n_classes,
                  max_depth=md, learning_rate=lr, n_estimators=1000,
                  subsample=ss, colsample_bytree=cs,
                  reg_alpha=0.1, reg_lambda=1.0,
                  random_state=random_state, tree_method='hist',
                  early_stopping_rounds=20)
        m = xgb.XGBClassifier(**kw)
        m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              sample_weight_eval_set=[w_val], verbose=False)
        f1v = f1_score(y_val, m.predict(X_val), average='macro')
        if f1v > best_f1:
            best_f1, best_m, best_p = f1v, m, (md, lr, ss, cs)

    md, lr, ss, cs = best_p
    final = xgb.XGBClassifier(
        objective=objective, eval_metric=eval_metric,
        num_class=n_classes,
        max_depth=md, learning_rate=lr,
        n_estimators=best_m.best_iteration + 1,
        subsample=ss, colsample_bytree=cs,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=random_state, tree_method='hist')
    final.fit(X_train, y_train, sample_weight=w_all, verbose=False)

    yp = final.predict(X_test)
    print(f"    best params : depth={md} lr={lr} sub={ss} col={cs}  "
          f"iter={best_m.best_iteration}")
    _print_result(y_train, final.predict(X_train),
                  y_test, yp, best_f1, cv_label='val f1')
    return yp


def _logistic(X_train, y_train, X_test, y_test):
    """
    Logistic Regression — multinomial for multi-class age groups.
    GridSearch over C and solver.
    """
    param_grid = {
        'C'       : [0.01, 0.1, 1, 10, 100],
        'solver'  : ['lbfgs', 'saga'],
        'max_iter': [2000],
    }
    grid = GridSearchCV(
        LogisticRegression(class_weight='balanced',
                           random_state=random_state),
        param_grid, cv=10, scoring='f1_macro',
        n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    yp   = best.predict(X_test)
    print(f"    best params : {grid.best_params_}")
    _print_result(y_train, best.predict(X_train), y_test, yp, grid.best_score_)
    return yp


def _lda(X_train, y_train, X_test, y_test):
    """
    Linear Discriminant Analysis — finds the linear combinations of features
    that best separate the age group classes.
    No hyperparameters to tune; 10-fold CV reported for consistency.
    """
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    yp    = model.predict(X_test)
    cv_f1 = cross_val_score(
        LinearDiscriminantAnalysis(),
        X_train, y_train,
        cv=10, scoring='f1_macro', n_jobs=-1).mean()
    print(f"    no hyperparameters  (10-fold CV f1_macro reported)")
    _print_result(y_train, model.predict(X_train), y_test, yp, cv_f1)
    return yp


def _ridge(X_train, y_train, X_test, y_test):
    """
    Ridge Classifier — one-vs-rest for multi-class; GridSearch over alpha.
    """
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(
        RidgeClassifier(class_weight='balanced'),
        param_grid, cv=10, scoring='f1_macro',
        n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    yp   = best.predict(X_test)
    print(f"    best params : {grid.best_params_}")
    _print_result(y_train, best.predict(X_train), y_test, yp, grid.best_score_)
    return yp


# ============================================================================
# 6.  Run all 7 models on one feature subset
# ============================================================================

def run_all_models(X_train, y_train, X_test, y_test, tag, le):
    runners = [
        ('KNN',                 _knn),
        ('Random Forest',       _rf),
        ('SVM',                 _svm),
        ('XGBoost',             _xgboost),
        ('Logistic Regression', _logistic),
        ('LDA',                 _lda),
        ('Ridge Classifier',    _ridge),
    ]
    rows = []
    for name, fn in runners:
        print(f"\n  [{name}]")
        yp   = fn(X_train, y_train, X_test, y_test)
        slug = name.lower().replace(' ', '_')
        _confusion_plot(y_test, yp, le.classes_,
                        f'{name} — {tag}',
                        out_dir / 'figures' / f'confusion_{slug}_{tag}.png')
        rows.append({'model'   : name,
                     'tag'     : tag,
                     'accuracy': round(accuracy_score(y_test, yp), 4),
                     'f1_macro': round(f1_score(y_test, yp, average='macro'), 4)})
    return rows


# ============================================================================
# 7.  Comparison plot
# ============================================================================

def plot_comparison(results_df, top1_name, top2_name):
    tag_colors = {'full': '#2196F3', 'top1': '#FF9800', 'top2': '#4CAF50'}
    tag_labels = {
        'full': 'Full features (baseline)',
        'top1': f'Top-1: {top1_name}',
        'top2': f'Top-2: {top1_name} + {top2_name}',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric in zip(axes, ['accuracy', 'f1_macro']):
        pivot = results_df.pivot(index='model', columns='tag', values=metric)
        x     = np.arange(len(pivot))
        n     = len(pivot.columns)
        w     = 0.70 / n

        for i, tag in enumerate(['full', 'top1', 'top2']):
            if tag not in pivot.columns:
                continue
            offset = (i - n / 2 + 0.5) * w
            bars   = ax.bar(x + offset, pivot[tag], w,
                            label=tag_labels[tag],
                            color=tag_colors[tag],
                            alpha=0.85, edgecolor='white')
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.005,
                        f'{b.get_height():.3f}',
                        ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_ylim(0, 1.15)
        ax.set_title(f'{metric.replace("_", " ").title()} — full vs top-1 vs top-2',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        'Age group prediction — feature subset comparison\n'
        '(Full = same feature matrix as age_group_prediction_pipeline.py)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'feature_subset_comparison_age.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: feature_subset_comparison_age.png")


# ============================================================================
# main
# ============================================================================

def main():
    cfg_label = '3-class' if USE_3_CLASS else '4-class MATSim IVT 2015'
    print("=" * 70)
    print(f"Age group correlation analysis — pre-built normalised feature matrix")
    print(f"bins   : {cfg_label}")
    print(f"corr   : Spearman  (ordinal target, not binary)")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'figures').mkdir(exist_ok=True)

    # 1. Load
    merged, feat_cols, le = load_data()

    # 2. Spearman correlation
    corr_df = compute_correlations(merged, feat_cols)

    # 3. Plots
    print("\n" + "=" * 70)
    print("Generating correlation plots")
    print("=" * 70)
    plot_correlation_bar(corr_df, n_top=20)
    plot_correlation_heatmap(merged, feat_cols, corr_df, n_top=25)
    plot_top_feature_distributions(merged, corr_df, le, n=2)

    # 4. Top features
    top1_name = corr_df.iloc[0]['feature']
    top2_name = corr_df.iloc[1]['feature']
    top1_r    = corr_df.iloc[0]['r']
    top2_r    = corr_df.iloc[1]['r']

    print("\n" + "=" * 70)
    print("Top correlated features")
    print("=" * 70)
    print(f"  #1  {top1_name:<40}  Spearman r = {top1_r:+.4f}")
    print(f"  #2  {top2_name:<40}  Spearman r = {top2_r:+.4f}")

    # 5. Split
    X_full = merged[feat_cols].values.astype(float)
    y      = merged['age_enc'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size,
        random_state=random_state, stratify=y)
    print(f"\n  Split — train: {len(X_train)}  test: {len(X_test)}")

    idx1 = feat_cols.index(top1_name)
    idx2 = feat_cols.index(top2_name)
    X_train_top1 = X_train[:, [idx1]]
    X_test_top1  = X_test[:,  [idx1]]
    X_train_top2 = X_train[:, [idx1, idx2]]
    X_test_top2  = X_test[:,  [idx1, idx2]]

    all_rows = []

    # Full (baseline)
    print("\n" + "=" * 70)
    print(f"FULL features  ({len(feat_cols)})  <- baseline")
    print("=" * 70)
    all_rows += run_all_models(X_train, y_train, X_test, y_test, 'full', le)

    # Top-1
    print("\n" + "=" * 70)
    print(f"TOP-1 feature  [{top1_name}  r={top1_r:+.4f}]")
    print("=" * 70)
    all_rows += run_all_models(X_train_top1, y_train, X_test_top1, y_test, 'top1', le)

    # Top-2
    print("\n" + "=" * 70)
    print(f"TOP-2 features  [{top1_name}  +  {top2_name}]")
    print("=" * 70)
    all_rows += run_all_models(X_train_top2, y_train, X_test_top2, y_test, 'top2', le)

    # 6. Summary
    results_df = pd.DataFrame(all_rows)

    print("\n" + "=" * 70)
    print("SUMMARY — full vs top-1 vs top-2")
    print("=" * 70)
    pivot = results_df.pivot_table(
        index='model', columns='tag', values=['accuracy', 'f1_macro']
    ).reindex(columns=['full', 'top1', 'top2'], level='tag')
    print(pivot.round(4).to_string())

    results_df.to_csv(out_dir / 'feature_subset_comparison_age.csv', index=False)
    print("\n  saved: feature_subset_comparison_age.csv")

    print("\n" + "=" * 70)
    print("Generating comparison plot")
    print("=" * 70)
    plot_comparison(results_df, top1_name, top2_name)

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)
    print(f"  Outputs : {out_dir}")
    print(f"  Top-1   : {top1_name}  (r = {top1_r:+.4f})")
    print(f"  Top-2   : {top2_name}  (r = {top2_r:+.4f})")


if __name__ == '__main__':
    main()
