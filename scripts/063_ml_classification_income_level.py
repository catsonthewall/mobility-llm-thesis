"""
Mobility feature classification pipeline
=============================================================================
Models  : KNN · Random Forest · SVM (RBF) · XGBoost
CV      : 10-fold GridSearchCV (KNN / RF / SVM)
          12-candidate search + early stopping (XGBoost)
Split   : 80 % train  /  20 % test  (stratified)
Target  : income_category  (5-class or 3-class, set USE_3_CLASS below)
Input   : feature_matrix_normalised.csv  from 03_features_built_17may
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
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
#feat_file   = Path('/data/baliu/thesis/02_merged_data/11_mobility_features/feature_matrix_normalised.csv')
feat_file = Path('/data/baliu/thesis/09_indicators/2_mobility_features_4weeks/feature_matrix_raw.csv')

#feature_path = Path("/data/baliu/thesis/02_merged_data/11_mobility_features/feature_matrix_normalised.csv")
#demo_path    = Path("/data/baliu/thesis/00data/sp_toponym_poi_purpose_demographics.csv")

survey_file = Path('/data/baliu/thesis/00data/sp_toponym_poi_purpose_demographics.csv')
out_dir     = Path('/data/baliu/thesis/03_method/07_outputs/00_ml_1508')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'figures').mkdir(exist_ok=True)
(out_dir / 'models').mkdir(exist_ok=True)

user_col       = 'user_id'
target_col     = 'income_category'   # final label column used for training
random_state   = 42
test_size      = 0.20                # 20 % held-out test set

# set true to collapse 5 income bins → 3 (Low / mid / high)
use_3_class    = False


# ============================================================================
# 1.  load & merge features with survey labels
# ============================================================================

def categorize_income(v) -> str | None:
    if pd.isna(v): return None
    s = str(v).strip().lower()
    if s == 'prefer not to say': return None
    if '4 000 chf or less'  in s or '4000 chf or less'  in s: return '0_<4000'
    if '4 001 - 8 000 chf'  in s: return '1_4001-8000'
    if '8 001 - 12 000 chf' in s: return '2_8001-12000'
    if '12 001 - 16 000 chf'in s: return '3_12001-16000'
    if 'more than 16 000 chf'in s: return '4_>16001'
    return None

def collapse_3_class(cat) -> str | None:
    if cat in ('0_<4000', '1_4001-8000'):     return '0_Low'
    if cat == '2_8001-12000':                 return '1_Mid'
    if cat in ('3_12001-16000', '4_>16001'):  return '2_High'
    return None

def load_and_merge() -> tuple[np.ndarray, np.ndarray, LabelEncoder, list]:
    print("=" * 70)
    print("loading data")
    print("=" * 70)

    # ── feature matrix (already normalised) ──────────────────────────────
    feat_df = pd.read_csv(feat_file, dtype={user_col: str})
    feat_df[user_col] = feat_df[user_col].str.strip()
    print(f"feature matrix : {feat_df.shape[0]} users × {feat_df.shape[1]-1} features")

    # ── survey labels ─────────────────────────────────────────────────────
    survey = pd.read_csv(survey_file, dtype={user_col: str})
    survey[user_col] = survey[user_col].str.strip()

    # one row per user — take first occurrence of income
    income_df = (survey.groupby(user_col)['income']
                        .first()
                        .reset_index())

    income_df['income_category'] = income_df['income'].apply(categorize_income)
    income_df = income_df.dropna(subset=['income_category'])

    if use_3_class:
        income_df['income_category'] = income_df['income_category'].apply(collapse_3_class)
        income_df = income_df.dropna(subset=['income_category'])
        print("  3-class mode: Low / Mid / High")

    print("  income distribution:")
    print(income_df['income_category'].value_counts().sort_index()
          .to_string(header=False))

    # ── merge ─────────────────────────────────────────────────────────────
    merged = feat_df.merge(income_df[[user_col, 'income_category']],
                            on=user_col, how='inner')
    print(f"  merged : {len(merged)} users with both features and income label")

    feature_cols = [c for c in feat_df.columns if c != user_col]
    X = merged[feature_cols].values.astype(float)
    y_raw = merged['income_category'].values

    le  = LabelEncoder()
    y   = le.fit_transform(y_raw)

    print(f"  classes: {list(le.classes_)}")
    return X, y, le, feature_cols


# ============================================================================
# 2.  train / validation / test split
# ============================================================================

def split_data(X, y):
    """
    80 % train  (used for 10-fold CV inside GridSearchCV)
    20 % test   (locked until final evaluation)
    Stratified to preserve class proportions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"\n  train: {len(X_train):>5}  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  test : {len(X_test):>5}  ({len(X_test)/len(X)*100:.1f}%)")
    return X_train, X_test, y_train, y_test


# ============================================================================
# 3.  Model 1: KNN  — 10-fold GridSearchCV
# ============================================================================

def train_knn(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 70)
    print("KNN — 10-fold cross-validation")
    print("=" * 70)

    # KNN is distance-based: StandardScaler applied before search
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 19, 25],
        'weights'    : ['uniform', 'distance'],
        'metric'     : ['euclidean', 'manhattan', 'minkowski'],
    }
    n_combos = (len(param_grid['n_neighbors']) *
                len(param_grid['weights']) *
                len(param_grid['metric']))
    print(f"  {n_combos} combinations × 10 folds = {n_combos * 10} fits")

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1,
        return_train_score=True,
    )
    grid.fit(X_tr_s, y_train)

    # ── validation report ─────────────────────────────────────────────────
    print(f"\n  best params : {grid.best_params_}")
    print(f"  CV f1_macro : {grid.best_score_:.4f}  "
          f"(±{grid.cv_results_['std_test_score'][grid.best_index_]:.4f})")

    cv_df = pd.DataFrame(grid.cv_results_)
    print("\n  top 3 configurations:")
    for _, row in cv_df.nsmallest(3, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score']].iterrows():
        print(f"    {row['params']}")
        print(f"      CV: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")

    # ── final evaluation ──────────────────────────────────────────────────
    best = grid.best_estimator_
    y_pred_tr = best.predict(X_tr_s)
    y_pred_te = best.predict(X_te_s)

    tr_acc = accuracy_score(y_train, y_pred_tr)
    te_acc = accuracy_score(y_test,  y_pred_te)
    te_f1  = f1_score(y_test, y_pred_te, average='macro')

    print(f"\n  train accuracy : {tr_acc:.4f}")
    print(f"  test  accuracy : {te_acc:.4f}")
    print(f"  test  f1_macro : {te_f1:.4f}")
    print(f"  CV    f1_macro : {grid.best_score_:.4f}")
    _overfit_check(tr_acc, te_acc)

    return best, scaler, y_pred_te


# ============================================================================
# 4.  Model 2: Random Forest  — 10-fold GridSearchCV
# ============================================================================

def train_rf(X_train, y_train, X_test, y_test, feature_cols):
    print("\n" + "=" * 70)
    print("Random Forest — 10-fold cross-validation")
    print("=" * 70)

    param_grid = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
        'max_features'    : ['sqrt', 'log2'],
    }
    n_combos = (len(param_grid['n_estimators']) *
                len(param_grid['max_depth']) *
                len(param_grid['min_samples_split']) *
                len(param_grid['min_samples_leaf']) *
                len(param_grid['max_features']))
    print(f"  {n_combos} combinations × 10 folds = {n_combos * 10} fits")
    print("  note: RandomizedSearchCV used (n_iter=50) to keep runtime feasible")

    from sklearn.model_selection import RandomizedSearchCV
    grid = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced',
                                random_state=random_state, n_jobs=-1),
        param_grid,
        n_iter=50,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    # ── validation report ─────────────────────────────────────────────────
    print(f"\n  best params:")
    for k, v in grid.best_params_.items():
        print(f"    {k}: {v}")
    print(f"\n  CV f1_macro : {grid.best_score_:.4f}  "
          f"(±{grid.cv_results_['std_test_score'][grid.best_index_]:.4f})")

    cv_df = pd.DataFrame(grid.cv_results_)
    print("\n  top 3 configurations:")
    for _, row in cv_df.nsmallest(3, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score']].iterrows():
        print(f"    CV: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")

    # ── final evaluation ──────────────────────────────────────────────────
    best = grid.best_estimator_
    y_pred_tr = best.predict(X_train)
    y_pred_te = best.predict(X_test)

    tr_acc = accuracy_score(y_train, y_pred_tr)
    te_acc = accuracy_score(y_test,  y_pred_te)
    te_f1  = f1_score(y_test, y_pred_te, average='macro')

    print(f"\n  train accuracy : {tr_acc:.4f}")
    print(f"  test  accuracy : {te_acc:.4f}")
    print(f"  test  f1_macro : {te_f1:.4f}")
    print(f"  CV    f1_macro : {grid.best_score_:.4f}")
    _overfit_check(tr_acc, te_acc)

    # ── feature importance ────────────────────────────────────────────────
    feat_imp = pd.DataFrame({
        'feature'   : feature_cols,
        'importance': best.feature_importances_,
    }).sort_values('importance', ascending=False)
    print("\n  top 15 features:")
    print(feat_imp.head(15).to_string(index=False))

    return best, feat_imp, y_pred_te


# ============================================================================
# 5.  Model 3: SVM (RBF)  — 10-fold GridSearchCV
# ============================================================================

def train_svm(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 70)
    print("SVM (RBF kernel) — 10-fold cross-validation")
    print("=" * 70)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    param_grid = {
        'C'     : [0.1, 1, 10, 100],
        'gamma' : ['scale', 'auto', 0.01, 0.001],
        'kernel': ['rbf'],
    }
    n_combos = len(param_grid['C']) * len(param_grid['gamma'])
    print(f"  {n_combos} combinations × 10 folds = {n_combos * 10} fits")

    grid = GridSearchCV(
        SVC(class_weight='balanced', probability=True, random_state=random_state),
        param_grid,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1,
        return_train_score=True,
    )
    grid.fit(X_tr_s, y_train)

    # ── validation report ─────────────────────────────────────────────────
    print(f"\n  best params : {grid.best_params_}")
    print(f"  CV f1_macro : {grid.best_score_:.4f}  "
          f"(±{grid.cv_results_['std_test_score'][grid.best_index_]:.4f})")

    cv_df = pd.DataFrame(grid.cv_results_)
    print("\n  top 3 configurations:")
    for _, row in cv_df.nsmallest(3, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score']].iterrows():
        print(f"    {row['params']}")
        print(f"      CV: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")

    # ── final evaluation ──────────────────────────────────────────────────
    best = grid.best_estimator_
    y_pred_tr = best.predict(X_tr_s)
    y_pred_te = best.predict(X_te_s)

    tr_acc = accuracy_score(y_train, y_pred_tr)
    te_acc = accuracy_score(y_test,  y_pred_te)
    te_f1  = f1_score(y_test, y_pred_te, average='macro')

    print(f"\n  train accuracy : {tr_acc:.4f}")
    print(f"  test  accuracy : {te_acc:.4f}")
    print(f"  test  f1_macro : {te_f1:.4f}")
    print(f"  CV    f1_macro : {grid.best_score_:.4f}")
    _overfit_check(tr_acc, te_acc)

    return best, scaler, y_pred_te


# ============================================================================
# 6. model 4: XGBoost  — 12-candidate search + early stopping
# ============================================================================

def _compute_sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    wmap = dict(zip(classes, n_samples / (n_classes * counts)))
    return np.array([wmap[yi] for yi in y])


def train_xgboost(X_train, y_train, X_test, y_test, feature_cols):
    print("\n" + "=" * 70)
    print("XGBoost — 12-candidate search + early stopping")
    print("=" * 70)

    n_classes = len(np.unique(y_train))

    param_candidates = [
        # (max_depth, learning_rate, subsample, colsample_bytree)
        (3, 0.05, 0.8, 0.8), (3, 0.1,  0.8, 0.8),
        (4, 0.05, 0.8, 0.8), (4, 0.1,  0.8, 0.8),
        (5, 0.05, 0.8, 0.8), (5, 0.1,  0.8, 0.8),
        (6, 0.05, 0.8, 0.8), (6, 0.1,  0.8, 0.8),
        (4, 0.05, 0.7, 0.7), (4, 0.1,  0.7, 0.7),
        (5, 0.05, 0.7, 0.7), (5, 0.1,  0.7, 0.7),
    ]

    # 15 % of training data used as validation (for early stopping only)
    # test set remains locked
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, _compute_sample_weights(y_train),
        test_size=0.15, random_state=random_state, stratify=y_train
    )

    print(f"  {len(param_candidates)} candidates")
    print(f"  train: {len(X_tr)}  |  val: {len(X_val)}  |  "
          f"test: {len(X_test)} (locked)")

    best_f1, best_model, best_params = -1.0, None, None
    results_log = []

    for md, lr, ss, cs in param_candidates:
        m = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            max_depth=md, learning_rate=lr,
            n_estimators=1000,
            subsample=ss, colsample_bytree=cs,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=random_state,
            tree_method='hist',
            early_stopping_rounds=20,
        )
        m.fit(X_tr, y_tr,
              sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              sample_weight_eval_set=[w_val],
              verbose=False)

        f1_val = f1_score(y_val, m.predict(X_val), average='macro')
        results_log.append({'max_depth': md, 'lr': lr,
                             'subsample': ss, 'colsample': cs,
                             'best_iter': m.best_iteration,
                             'f1_val': f1_val})
        if f1_val > best_f1:
            best_f1 = f1_val
            best_model = m
            best_params = {'max_depth': md, 'learning_rate': lr,
                           'subsample': ss, 'colsample_bytree': cs}

    # ── validation report ─────────────────────────────────────────────────
    print(f"\n  best params: {best_params}")
    print(f"  best val f1_macro: {best_f1:.4f}")
    print(f"  best iteration:    {best_model.best_iteration}")

    res_df = pd.DataFrame(results_log).sort_values('f1_val', ascending=False)
    print("\n  top 3 configurations:")
    for _, row in res_df.head(3).iterrows():
        print(f"    depth={row['max_depth']} lr={row['lr']}  "
              f"sub={row['subsample']} col={row['colsample']}")
        print(f"      val f1: {row['f1_val']:.4f}  "
              f"iterations: {row['best_iter']}")

    # retrain best config on FULL training set
    print("\n  retraining best config on full training set ...")
    full_w = _compute_sample_weights(y_train)
    final = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        n_estimators=best_model.best_iteration + 1,
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=random_state,
        tree_method='hist',
    )
    final.fit(X_train, y_train, sample_weight=full_w, verbose=False)

    # ── final evaluation ──────────────────────────────────────────────────
    y_pred_tr = final.predict(X_train)
    y_pred_te = final.predict(X_test)

    tr_acc = accuracy_score(y_train, y_pred_tr)
    te_acc = accuracy_score(y_test,  y_pred_te)
    te_f1  = f1_score(y_test, y_pred_te, average='macro')

    print(f"\n  train accuracy : {tr_acc:.4f}")
    print(f"  test  accuracy : {te_acc:.4f}")
    print(f"  test  f1_macro : {te_f1:.4f}")
    print(f"  val   f1_macro : {best_f1:.4f}")
    _overfit_check(tr_acc, te_acc)

    feat_imp = pd.DataFrame({
        'feature'   : feature_cols,
        'importance': final.feature_importances_,
    }).sort_values('importance', ascending=False)
    print("\n  top 10 features:")
    print(feat_imp.head(10).to_string(index=False))

    return final, feat_imp, y_pred_te


# ============================================================================
# overfit check function
# ============================================================================

def _overfit_check(tr_acc, te_acc):
    gap = tr_acc - te_acc
    if gap > 0.10:
        print(f"  ⚠  overfitting detected (gap: {gap:.4f})")
    else:
        print(f"  ✓  model generalises well (gap: {gap:.4f})")


# ============================================================================
# 7.  detailed evaluation + confusion matrices
# ============================================================================

def _plot_confusion_matrix(y_true, y_pred, labels, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    colors = ['#2d1b3d', '#3a4f8f', '#1f968b', '#73d055', '#fde724']
    cmap   = LinearSegmentedColormap.from_list('custom', colors, N=100)

    plt.figure(figsize=(max(8, len(labels)), max(7, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor='white', square=True,
                annot_kws={'size': 11, 'weight': 'bold'})

    n_test = len(y_true)
    acc    = accuracy_score(y_true, y_pred)
    plt.title(f'{model_name} — confusion matrix  '
              f'(n={n_test}, accuracy={acc:.3f})',
              fontsize=13, fontweight='bold', pad=16)
    plt.xlabel('Predicted', fontsize=11, fontweight='bold')
    plt.ylabel('True',      fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def detailed_evaluation(models_dict, X_test, y_test, le):
    print("\n" + "=" * 70)
    print("detailed model evaluation — test set")
    print("=" * 70)

    for name, info in models_dict.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        model  = info['model']
        X_proc = info['scaler'].transform(X_test) if 'scaler' in info else X_test
        y_pred = model.predict(X_proc)

        print("\n  classification report:")
        print(classification_report(y_test, y_pred,
                                    target_names=le.classes_, digits=4))

        print("  confusion matrix (counts):")
        cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred),
                              index=le.classes_, columns=le.classes_)
        print(cm_df.to_string())

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro')
        print(f"\n  accuracy   : {acc:.4f}")
        print(f"  f1 macro   : {f1m:.4f}")

        # save confusion matrix figure + csv
        slug = name.lower().replace(' ', '_')
        _plot_confusion_matrix(y_test, y_pred, le.classes_, name,
                               out_dir / 'figures' / f'confusion_{slug}.png')
        cm_df.to_csv(out_dir / f'confusion_{slug}.csv')
        print(f"  saved: confusion_{slug}.png + .csv")


def print_comparison(models_dict, X_test, y_test, le):
    print("\n" + "=" * 70)
    print("model comparison summary")
    print("=" * 70)

    rows = []
    for name, info in models_dict.items():
        X_proc = info['scaler'].transform(X_test) if 'scaler' in info else X_test
        y_pred = info['model'].predict(X_proc)
        rows.append({
            'model'      : name,
            'accuracy'   : round(accuracy_score(y_test, y_pred),    4),
            'f1_macro'   : round(f1_score(y_test, y_pred, average='macro'),    4),
        })

    comp = pd.DataFrame(rows).sort_values('f1_macro', ascending=False)
    print(comp.to_string(index=False))
    comp.to_csv(out_dir / 'model_comparison.csv', index=False)

    best = comp.iloc[0]
    print(f"\n  best model: {best['model']}  "
          f"(f1_macro: {best['f1_macro']:.4f})")
    return comp


def plot_comparison_bar(comp_df):
    candidates = ['accuracy', 'f1_macro', 'f1_weighted']
    metrics = [m for m in candidates if m in comp_df.columns]
    if not metrics:
        print("  skipped model_comparison.png: no metric columns found")
        return
    x      = np.arange(len(comp_df))
    width  = 0.8 / len(metrics)          # bars fill the slot regardless of count
    colors = ['#2196F3', '#4CAF50', '#FF9800'][:len(metrics)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, comp_df[metric], width,
                      label=metric, color=color, alpha=0.85,
                      edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(comp_df['model'], fontsize=11)
    ax.set_ylabel('score', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title('model comparison — accuracy & F1 scores',
                  fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'model_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: model_comparison.png")


def plot_feature_importance(rf_imp, xgb_imp, n_top=20):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, imp, title in zip(
            axes,
            [rf_imp, xgb_imp],
            ['Random Forest feature importance', 'XGBoost feature importance']):
        top = imp.head(n_top).sort_values('importance')
        ax.barh(top['feature'], top['importance'],
                color='#2196F3', alpha=0.85, edgecolor='white')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('importance', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved: feature_importance.png")


# ============================================================================
# 8.  save models
# ============================================================================

def save_models(models_dict, le):
    for name, info in models_dict.items():
        slug = name.lower().replace(' ', '_')
        with open(out_dir / 'models' / f'{slug}_model.pkl', 'wb') as f:
            pickle.dump(info, f)
        print(f"  saved: models/{slug}_model.pkl")

    with open(out_dir / 'models' / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("  saved: models/label_encoder.pkl")


# ============================================================================
# main
# ============================================================================

def main():
    print("=" * 70)
    print("mobility feature classification pipeline")
    print(f"target: income_category  ({'3-class' if use_3_class else '5-class'})")
    print("=" * 70)

    # ── 1. load data ──────────────────────────────────────────────────────
    X, y, le, feature_cols = load_and_merge()

    # ── 2. split ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("train / test split  (80 / 20 stratified)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── 3. train models ───────────────────────────────────────────────────
    knn_model,  knn_scaler,  _ = train_knn(X_train, y_train, X_test, y_test)
    rf_model,   rf_imp,      _ = train_rf(X_train, y_train, X_test, y_test, feature_cols)
    svm_model,  svm_scaler,  _ = train_svm(X_train, y_train, X_test, y_test)
    xgb_model,  xgb_imp,     _ = train_xgboost(X_train, y_train, X_test, y_test, feature_cols)

    # ── 4. evaluate ───────────────────────────────────────────────────────
    models_dict = {
        'KNN'          : {'model': knn_model,  'scaler': knn_scaler},
        'Random Forest': {'model': rf_model},
        'SVM'          : {'model': svm_model,  'scaler': svm_scaler},
        'XGBoost'      : {'model': xgb_model},
    }

    detailed_evaluation(models_dict, X_test, y_test, le)
    comp_df = print_comparison(models_dict, X_test, y_test, le)

    # ── 5. plots ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("generating plots")
    print("=" * 70)
    plot_comparison_bar(comp_df)
    plot_feature_importance(rf_imp, xgb_imp)

    # ── 6. save feature importances ───────────────────────────────────────
    rf_imp.to_csv( out_dir / 'feature_importance_rf.csv',  index=False)
    xgb_imp.to_csv(out_dir / 'feature_importance_xgb.csv', index=False)
    print("  saved: feature_importance_rf.csv  &  feature_importance_xgb.csv")

    # ── 7. save models ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("saving models")
    print("=" * 70)
    save_models(models_dict, le)

    # ── done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("done")
    print("=" * 70)
    print(f"\n  outputs: {out_dir}")
    print(f"""
  files:
    model_comparison.csv
    feature_importance_rf.csv
    feature_importance_xgb.csv
    confusion_knn.png/csv
    confusion_random_forest.png/csv
    confusion_svm.png/csv
    confusion_xgboost.png/csv
    figures/model_comparison.png
    figures/feature_importance.png
    models/knn_model.pkl
    models/random_forest_model.pkl
    models/svm_model.pkl
    models/xgboost_model.pkl
    models/label_encoder.pkl
    """)


if __name__ == '__main__':
    main()
