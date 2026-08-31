"""
Income level correlation analysis — pre-built normalised feature matrix
=============================================================================
Input    : feature_matrix_normalised.csv  (already MinMax [0,1])
           sp_toponym_poi_purpose_demographics.csv  (income ground truth)

Target   : income_level — 5-class ordinal
           The script auto-detects whether the income column is already
           categorical (string labels) or numeric (binned into 5 groups).

           If numeric, default 5-class CHF/month thresholds (Swiss MOBIS):
               0_low        < 4 000
               1_lower-mid  4 001 – 8 000
               2_mid        8 001 – 12 000
               3_upper-mid  12001 – 16 000
               4_high       ≥ 16 001

Step 1   : Spearman correlation  (every feature ↔ ordinal income label)
           Spearman is used because income classes are ordered but the
           distances between classes are not equal.
Step 2   : Correlation plots (bar, heatmap, violin distributions)
Step 3   : Run all 7 classifiers
               KNN · Random Forest · SVM · XGBoost          (original 4)
               Logistic Regression · LDA · Ridge Classifier  (linear models)
               — full feature set       (baseline)
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
# Config
# ============================================================================

feat_file   = Path('/data/baliu/thesis/00data/03_features_built_17may/feature_matrix_normalised.csv')
survey_file = Path('/data/baliu/thesis/00data/sp_toponym_poi_purpose_demographics.csv')
out_dir     = Path('/data/baliu/thesis/04_method_corr/04_correlation_results/income_correlation')

user_col     = 'user_id'
income_col   = 'income'          
random_state = 42
test_size    = 0.20

# ── Numeric income binning (CHF / month, Swiss MOBIS) ────────────────────────
# Used only when the income column contains numbers rather than string labels.
# pd.cut right=False  ->  [left, right)
INCOME_BINS = {
    'bins'  : [0, 4000, 8000, 12000, 16000, float('inf')],
    'labels': ['0_low', '1_lower-mid', '2_mid', '3_upper-mid', '4_high'],
}

# String labels considered valid (case-insensitive prefix match kept flexible).
# Leave empty [] to keep all non-null string values found in the column.
KEEP_LABELS: list = []


# ============================================================================
# 1.  Load pre-built features + income ground truth
# ============================================================================

def _parse_income(series: pd.Series) -> pd.Series:
    """
    Accept income as:
      - already ordered string labels  (returned as-is after stripping)
      - integers / floats              (binned using income bins defined above)
      - string representations of numbers ('6500' -> 6500.0 -> binned)
    Returns a string-label Series aligned to the input index.
    """
    stripped = series.astype(str).str.strip()

    # Try numeric conversion
    numeric = pd.to_numeric(stripped, errors='coerce')
    if numeric.notna().mean() > 0.5:          # majority are numbers -> bin them
        binned = pd.cut(numeric,
                        bins=INCOME_BINS['bins'],
                        labels=INCOME_BINS['labels'],
                        right=False)
        return binned.astype(str).where(numeric.notna(), other=np.nan)

    # Already categorical strings
    return stripped.replace({'nan': np.nan, 'None': np.nan, '': np.nan})

def load_data():
    print("=" * 70)
    print("Loading pre-built feature matrix  |  target: income_level (5-class)")
    print("=" * 70)

    # Feature matrix (already MinMax normalised — no rescaling)
    feat_df = pd.read_csv(feat_file, dtype={user_col: str})
    feat_df[user_col] = feat_df[user_col].str.strip()
    feat_cols = [c for c in feat_df.columns if c != user_col]
    print(f"  Feature matrix : {feat_df.shape[0]} users x "
          f"{len(feat_cols)} features  [MinMax 0-1, no rescaling]")

    # Survey / income ground truth
    survey = pd.read_csv(survey_file, dtype={user_col: str})
    survey[user_col] = survey[user_col].str.strip()

    if income_col not in survey.columns:
        available = [c for c in survey.columns if 'inc' in c.lower() or 'salary' in c.lower()]
        raise KeyError(
            f"Column '{income_col}' not found in survey. "
            f"Possible income columns: {available}. "
            f"Update income_col in CONFIG.")

    inc_df = (survey.groupby(user_col)[income_col]
                     .first()
                     .reset_index())

    print(f"\n  Raw '{income_col}' distribution (before parsing):")
    print(inc_df[income_col].value_counts(dropna=False).to_string())

    inc_df['income_level'] = _parse_income(inc_df[income_col])
    inc_df = inc_df.dropna(subset=['income_level'])
    inc_df = inc_df[inc_df['income_level'] != 'nan']

    # keep only specified labels
    if KEEP_LABELS:
        inc_df = inc_df[inc_df['income_level'].isin(KEEP_LABELS)]

    print(f"\n  Parsed income_level distribution:")
    print(inc_df['income_level'].value_counts().sort_index().to_string())

    # Merge
    merged = feat_df.merge(inc_df[[user_col, 'income_level']],
                            on=user_col, how='inner')
    print(f"\n  Merged : {len(merged)} users with features + income label")

    # LabelEncoder preserves the natural sort order of our 0_/1_/2_/3_/4_ prefixes
    le = LabelEncoder().fit(
        np.sort(merged['income_level'].unique()))
    merged['income_enc'] = le.transform(merged['income_level'].values)

    print(f"\n  Classes : {list(le.classes_)}")
    unique, counts = np.unique(merged['income_enc'].values, return_counts=True)
    for cls, n in zip(le.classes_, counts):
        print(f"    {cls:<20} {n:>5}  ({n / len(merged) * 100:.1f}%)")

    return merged, feat_cols, le

# ============================================================================
# 2.  Spearman correlation  (feature <-> ordinal income level)
# ============================================================================

def compute_correlations(merged, feat_cols):
    print("\n" + "=" * 70)
    print("Spearman correlation  (feature <-> income_level ordinal index)")
    print("=" * 70)

    y = merged['income_enc'].values
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

    corr_df.to_csv(out_dir / 'income_feature_correlation.csv', index=False)
    print("\n  saved: income_feature_correlation.csv")
    return corr_df


# ============================================================================
# 3.  Correlation plots
# ============================================================================

def plot_correlation_bar(corr_df, n_top=20):
    """Bar chart — top-N features by |Spearman r|, coloured by sign."""
    top    = corr_df.head(n_top).sort_values('abs_r')
    # positive r -> higher income, negative r -> lower income
    colors = ['#4CAF50' if r > 0 else '#9C27B0' for r in top['r']]

    fig, ax = plt.subplots(figsize=(9, max(6, n_top * 0.38)))
    bars = ax.barh(top['feature'], top['abs_r'],
                   color=colors, alpha=0.85, edgecolor='white')
    for bar, r_val in zip(bars, top['r']):
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f'{r_val:+.3f}', va='center', fontsize=8)

    ax.set_xlabel('|Spearman r|', fontsize=11)
    ax.set_title(
        f'Top {n_top} features correlated with income level  (Spearman)\n'
        f'(green = r > 0 -> higher income,  purple = r < 0 -> lower income)',
        fontsize=11, fontweight='bold')
    ax.set_xlim(0, corr_df['abs_r'].max() * 1.20)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'correlation_bar_income.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("saved: correlation_bar_income.png")


def plot_correlation_heatmap(merged, feat_cols, corr_df, n_top=25):
    """Spearman heatmap — top n features + income label."""
    top_feats = corr_df.head(n_top)['feature'].tolist()
    sub = (merged[top_feats + ['income_enc']]
           .rename(columns={'income_enc': 'income_level_ord'}))
    corr_matrix = sub.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(max(10, n_top * 0.5),
                                     max(9,  n_top * 0.45)))
    sns.heatmap(corr_matrix, ax=ax,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                center=0, annot=False,
                linewidths=0.4, linecolor='white')
    ax.set_title(f'Spearman correlation heatmap  '
                 f'(top {n_top} features by |r| with income level)',
                 fontsize=11, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'correlation_heatmap_income.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(" saved: correlation_heatmap_income.png")


def plot_top_feature_distributions(merged, corr_df, le, n=2):
    """Violin plots of top-N features split by income level."""
    top_feats = corr_df.head(n)['feature'].tolist()
    n_classes = len(le.classes_)
    palette   = dict(zip(le.classes_,
                         sns.color_palette('viridis', n_classes)))

    merged_plot = merged.copy()
    merged_plot['income_level'] = le.inverse_transform(
        merged_plot['income_enc'].values)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, top_feats):
        r_val = corr_df.loc[corr_df['feature'] == feat, 'r'].values[0]
        sns.violinplot(data=merged_plot, x='income_level', y=feat,
                       order=le.classes_,
                       palette=palette, inner='box', ax=ax, alpha=0.75)
        ax.set_title(f'{feat}\n(Spearman r = {r_val:+.3f})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('income level', fontsize=9)
        ax.set_ylabel('normalised value [0-1]', fontsize=9)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Top correlated feature distributions by income level',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'top_feature_distributions_income.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(" saved: top_feature_distributions_income.png")

# ============================================================================
# 4. functions for model training and evaluation 
# ============================================================================

def _print_result(y_train, y_pred_tr, y_test, y_pred_te,
                  cv_score, cv_label='CV f1_macro'):
    tr_acc = accuracy_score(y_train, y_pred_tr)
    te_acc = accuracy_score(y_test,  y_pred_te)
    te_f1  = f1_score(y_test, y_pred_te, average='macro')
    gap    = tr_acc - te_acc
    flag   = '  Warning overfit' if gap > 0.10 else '  OK'
    print(f"    train={tr_acc:.4f}  test={te_acc:.4f}  "
          f"f1={te_f1:.4f}  {cv_label}={cv_score:.4f}  gap={gap:.3f}{flag}")

def _confusion_plot(y_true, y_pred, labels, title, path):
    cm   = confusion_matrix(y_true, y_pred)
    cmap = LinearSegmentedColormap.from_list(
        'c', ['#2d1b3d', '#3a4f8f', '#1f968b', '#73d055', '#fde724'], N=100)
    size = max(7, len(labels) + 2)
    plt.figure(figsize=(size, size - 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor='white', square=True,
                annot_kws={'size': 10, 'weight': 'bold'})
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
    n_classes = len(np.unique(y_train))
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
        kw = dict(objective='multi:softprob', eval_metric='mlogloss',
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
        objective='multi:softprob', eval_metric='mlogloss',
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
    Logistic regression — sklearn >= 1.5 handles multi-class automatically.
    GridSearch over C and solver. multi_class argument removed (deprecated).
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
    print(f"best params : {grid.best_params_}")
    _print_result(y_train, best.predict(X_train), y_test, yp, grid.best_score_)
    return yp


def _lda(X_train, y_train, X_test, y_test):
    """
    Linear discriminant analysis — finds linear combinations that best
    separate the 5 income classes. No hyperparameters; 10-fold CV reported.
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
    Ridge classifier — one-vs-rest for 5-class income; GridSearch over alpha.
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

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
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
        'Income level prediction (5-class) — feature subset comparison',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'feature_subset_comparison_income.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: feature_subset_comparison_income.png")


# ============================================================================
# main
# ============================================================================

def main():
    print("=" * 70)
    print("Income level correlation analysis — pre-built normalised features")
    print("classes : 5  (0_low · 1_lower-mid · 2_mid · 3_upper-mid · 4_high)")
    print("corr    : Spearman  (ordinal target, unequal class widths)")
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
    y      = merged['income_enc'].values
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
    print("Summary — full vs top-1 vs top-2")
    print("=" * 70)
    pivot = results_df.pivot_table(
        index='model', columns='tag', values=['accuracy', 'f1_macro']
    ).reindex(columns=['full', 'top1', 'top2'], level='tag')
    print(pivot.round(4).to_string())

    results_df.to_csv(out_dir / 'feature_subset_comparison_income.csv', index=False)
    print("\n saved: feature_subset_comparison_income.csv")

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
