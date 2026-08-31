"""
Gender correlation analysis — pre-built normalised feature matrix
=============================================================================
Input    : feature_matrix_normalised.csv  (already MinMax [0,1])
           sp_toponym_poi_purpose_demographics.csv  (gender ground truth)

Step 1   : Point-biserial correlation  (every feature ↔ binary gender label)
Step 2   : Correlation plots (bar, heatmap, violin distributions)
Step 3   : Run all 7 classifiers using
               KNN · Random Forest · SVM · XGBoost          (original 4)
               Logistic Regression · LDA · Ridge Classifier  (new linear models)
               — full feature set       (baseline = 01_ml_classification_gender.py)
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
from scipy.stats import pointbiserialr

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import xgboost as xgb

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================
feat_file    = Path('/data/baliu/thesis/00data/03_features_built_17may/feature_matrix_normalised.csv')
survey_file  = Path('/data/baliu/thesis/00data/sp_toponym_poi_purpose_demographics.csv')
out_dir      = Path('/data/baliu/thesis/03_method/00_ml_classification/gender_correlation')

user_col     = 'user_id'
random_state = 42
test_size    = 0.20
keep_genders = {'Female', 'Male'}


# ============================================================================
# 1.  Load pre-built features + gender ground truth
# ============================================================================

def load_data():
    print("=" * 70)
    print("Loading pre-built feature matrix  |  target: gender")
    print("=" * 70)

    # Feature matrix (already MinMax normalised — no rescaling)
    feat_df = pd.read_csv(feat_file, dtype={user_col: str})
    feat_df[user_col] = feat_df[user_col].str.strip()
    feat_cols = [c for c in feat_df.columns if c != user_col]
    print(f"  Feature matrix : {feat_df.shape[0]} users x "
          f"{len(feat_cols)} features  [MinMax 0-1, no rescaling]")

    # Survey / gender ground truth
    survey = pd.read_csv(survey_file, dtype={user_col: str})
    survey[user_col] = survey[user_col].str.strip()
    if user_col not in survey.columns and 'participant_ID' in survey.columns:
        survey = survey.rename(columns={'participant_ID': user_col})

    gender_df = (survey.groupby(user_col)['gender']
                        .first()
                        .reset_index())
    gender_df['gender'] = gender_df['gender'].str.strip()

    print("\n  Raw gender distribution:")
    print(gender_df['gender'].value_counts().to_string())

    gender_df = gender_df[gender_df['gender'].isin(keep_genders)].copy()
    print(f"\n  After filtering to {keep_genders}:")
    print(gender_df['gender'].value_counts().to_string())

    merged = feat_df.merge(gender_df[[user_col, 'gender']], on=user_col, how='inner')
    print(f"\n  Merged : {len(merged)} users with features + gender label")

    le = LabelEncoder().fit(merged['gender'].values)   # Female=0, Male=1
    merged['gender_enc'] = le.transform(merged['gender'].values)
    print(f"  Encoding : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return merged, feat_cols, le


# ============================================================================
# 2.  Point-biserial correlation  (feature <-> binary gender)
# ============================================================================

def compute_correlations(merged, feat_cols):
    print("\n" + "=" * 70)
    print("Point-biserial correlation  (feature <-> gender)")
    print("=" * 70)

    y = merged['gender_enc'].values
    records = []
    for col in feat_cols:
        x = merged[col].values
        if x.std() == 0:
            continue
        r, p = pointbiserialr(y, x)
        records.append({'feature': col,
                        'r'      : round(r, 6),
                        'abs_r'  : abs(r),
                        'p_value': round(p, 6)})

    corr_df = (pd.DataFrame(records)
                 .sort_values('abs_r', ascending=False)
                 .reset_index(drop=True))

    print(f"\n  Top 20 features by |r|:")
    print(corr_df.head(20)[['feature', 'r', 'p_value']].to_string(index=False))

    corr_df.to_csv(out_dir / 'gender_feature_correlation.csv', index=False)
    print("\n  saved: gender_feature_correlation.csv")
    return corr_df


# ============================================================================
# 3.  Correlation plots
# ============================================================================

def plot_correlation_bar(corr_df, n_top=20):
    top    = corr_df.head(n_top).sort_values('abs_r')
    colors = ['#E91E63' if r > 0 else '#2196F3' for r in top['r']]

    fig, ax = plt.subplots(figsize=(9, max(6, n_top * 0.38)))
    bars = ax.barh(top['feature'], top['abs_r'],
                   color=colors, alpha=0.85, edgecolor='white')
    for bar, r_val in zip(bars, top['r']):
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f'{r_val:+.3f}', va='center', fontsize=8)
    ax.set_xlabel('|point-biserial r|', fontsize=11)
    ax.set_title(
        f'Top {n_top} features correlated with gender\n'
        f'(pink = r > 0 -> Male,  blue = r < 0 -> Female)',
        fontsize=11, fontweight='bold')
    ax.set_xlim(0, corr_df['abs_r'].max() * 1.20)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'correlation_bar_gender.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: correlation_bar_gender.png")


def plot_correlation_heatmap(merged, feat_cols, corr_df, n_top=25):
    top_feats = corr_df.head(n_top)['feature'].tolist()
    sub = (merged[top_feats + ['gender_enc']]
           .rename(columns={'gender_enc': 'gender(0=F 1=M)'}))
    corr_matrix = sub.corr()

    fig, ax = plt.subplots(figsize=(max(10, n_top * 0.5),
                                     max(9,  n_top * 0.45)))
    sns.heatmap(corr_matrix, ax=ax,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                center=0, annot=False,
                linewidths=0.4, linecolor='white')
    ax.set_title(f'Feature correlation heatmap  '
                 f'(top {n_top} by |r| with gender)',
                 fontsize=11, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'correlation_heatmap_gender.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: correlation_heatmap_gender.png")


def plot_top_feature_distributions(merged, corr_df, n=2):
    top_feats = corr_df.head(n)['feature'].tolist()
    palette   = {'Female': '#E91E63', 'Male': '#2196F3'}

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, top_feats):
        r_val = corr_df.loc[corr_df['feature'] == feat, 'r'].values[0]
        sns.violinplot(data=merged, x='gender', y=feat,
                       palette=palette, inner='box', ax=ax, alpha=0.75)
        ax.set_title(f'{feat}\n(r = {r_val:+.3f})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('normalised value [0-1]', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Top correlated feature distributions by gender',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'top_feature_distributions_gender.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: top_feature_distributions_gender.png")


# ============================================================================
# 4.  Individual model trainers
#     Features already in [0,1] — no extra scaling (same as gender script)
# ============================================================================

def _print_result(y_train, y_pred_tr, y_test, y_pred_te, cv_score,
                  cv_label='CV f1_macro'):
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
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor='white', square=True,
                annot_kws={'size': 13, 'weight': 'bold'})
    acc = accuracy_score(y_true, y_pred)
    plt.title(f'{title}\n(n={len(y_true)}, acc={acc:.3f})',
              fontsize=10, fontweight='bold', pad=10)
    plt.xlabel('predicted', fontsize=9)
    plt.ylabel('true',      fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


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
    _print_result(y_train, best.predict(X_train), y_test, yp, grid.best_score_)
    return yp


def _xgboost(X_train, y_train, X_test, y_test):
    n_classes   = len(np.unique(y_train))
    objective   = 'binary:logistic' if n_classes == 2 else 'multi:softprob'
    eval_metric = 'logloss'         if n_classes == 2 else 'mlogloss'

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
                  max_depth=md, learning_rate=lr, n_estimators=1000,
                  subsample=ss, colsample_bytree=cs,
                  reg_alpha=0.1, reg_lambda=1.0,
                  random_state=random_state, tree_method='hist',
                  early_stopping_rounds=20)
        if n_classes > 2:
            kw['num_class'] = n_classes
        m = xgb.XGBClassifier(**kw)
        m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              sample_weight_eval_set=[w_val], verbose=False)
        f1v = f1_score(y_val, m.predict(X_val), average='macro')
        if f1v > best_f1:
            best_f1, best_m, best_p = f1v, m, (md, lr, ss, cs)

    md, lr, ss, cs = best_p
    fkw = dict(objective=objective, eval_metric=eval_metric,
               max_depth=md, learning_rate=lr,
               n_estimators=best_m.best_iteration + 1,
               subsample=ss, colsample_bytree=cs,
               reg_alpha=0.1, reg_lambda=1.0,
               random_state=random_state, tree_method='hist')
    if n_classes > 2:
        fkw['num_class'] = n_classes
    final = xgb.XGBClassifier(**fkw)
    final.fit(X_train, y_train, sample_weight=w_all, verbose=False)
    yp = final.predict(X_test)
    _print_result(y_train, final.predict(X_train),
                  y_test, yp, best_f1, cv_label='val f1')
    return yp


# ============================================================================
# 5a.  Logistic Regression  — L2 penalty, GridSearch over C
#      Features already [0,1] — no extra scaling needed
# ============================================================================

def _logistic(X_train, y_train, X_test, y_test):
    param_grid = {
        'C'      : [0.01, 0.1, 1, 10, 100],
        'solver' : ['lbfgs', 'liblinear'],
        'max_iter': [1000],
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


# ============================================================================
# 5b.  Linear Discriminant Analysis  — no hyperparameters to tune
# ============================================================================

def _lda(X_train, y_train, X_test, y_test):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    yp  = model.predict(X_test)
    # Manual CV score for consistent reporting
    from sklearn.model_selection import cross_val_score
    cv_f1 = cross_val_score(LinearDiscriminantAnalysis(),
                             X_train, y_train,
                             cv=10, scoring='f1_macro',
                             n_jobs=-1).mean()
    print(f"    no hyperparameters  (10-fold CV f1_macro reported)")
    _print_result(y_train, model.predict(X_train), y_test, yp, cv_f1)
    return yp


# ============================================================================
# 5c.  Ridge Classifier  — GridSearch over alpha
# ============================================================================

def _ridge(X_train, y_train, X_test, y_test):
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
# 5.  Run all 7 models on one feature subset
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
        yp = fn(X_train, y_train, X_test, y_test)
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
# 6.  Comparison plot
# ============================================================================

def plot_comparison(results_df, top1_name, top2_name):
    tag_colors = {'full': '#2196F3', 'top1': '#FF9800', 'top2': '#4CAF50'}
    tag_labels = {
        'full': 'Full features (baseline)',
        'top1': f'Top-1: {top1_name}',
        'top2': f'Top-2: {top1_name} + {top2_name}',
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
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
                        ha='center', va='bottom', fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_ylim(0, 1.15)
        ax.set_title(f'{metric.replace("_", " ").title()} — full vs top-1 vs top-2',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        'Gender prediction — feature subset comparison\n'
        '(Full = same feature matrix as 01_ml_classification_gender.py)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'feature_subset_comparison_gender.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: feature_subset_comparison_gender.png")


# ============================================================================
# main
# ============================================================================

def main():
    print("=" * 70)
    print("Gender correlation analysis — pre-built normalised feature matrix")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'figures').mkdir(exist_ok=True)

    # 1. Load
    merged, feat_cols, le = load_data()

    # 2. Correlation
    corr_df = compute_correlations(merged, feat_cols)

    # 3. Plots
    print("\n" + "=" * 70)
    print("Generating correlation plots")
    print("=" * 70)
    plot_correlation_bar(corr_df, n_top=20)
    plot_correlation_heatmap(merged, feat_cols, corr_df, n_top=25)
    plot_top_feature_distributions(merged, corr_df, n=2)

    # 4. Top features
    top1_name = corr_df.iloc[0]['feature']
    top2_name = corr_df.iloc[1]['feature']
    top1_r    = corr_df.iloc[0]['r']
    top2_r    = corr_df.iloc[1]['r']

    print("\n" + "=" * 70)
    print("Top correlated features")
    print("=" * 70)
    print(f"  #1  {top1_name:<40}  r = {top1_r:+.4f}")
    print(f"  #2  {top2_name:<40}  r = {top2_r:+.4f}")

    # 5. Split
    X_full = merged[feat_cols].values.astype(float)
    y      = merged['gender_enc'].values
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

    results_df.to_csv(out_dir / 'feature_subset_comparison_gender.csv', index=False)
    print("\n  saved: feature_subset_comparison_gender.csv")

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
