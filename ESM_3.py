# =============================================================
# Article : Mechanism-Guided Synthetic Tax Data Generation:
#            A Computational Framework for Tax Evasion Detection
# Authors : Xiaojing Fan
# Affiliation: Business School, University of Shanghai for
#              Science and Technology, Shanghai, China
# Corresponding author: fanxiaojing@usst.edu.cn
# =============================================================

import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# Set a random seed to ensure reproducibility
np.random.seed(42)

# ============================================================================
# 1. Core utility functions
# ============================================================================
def calculate_audit_metrics(y_true, y_prob):
   
    if len(y_true) == 0: 
        return 0.0, 0.0, 0.0
    
    df = pd.DataFrame({'true': y_true.values if hasattr(y_true, 'values') else y_true, 
                       'prob': y_prob})
    
    # Sort by predicted probability in descending order
    df_sorted = df.sort_values('prob', ascending=False)
    n_audit = int(len(df) * 0.20)  # Top 20%
    top_k = df_sorted.head(n_audit)
    
    # Precision@Top20% (audit hit rate) - Paper's core metric
   
    precision_top20 = top_k['true'].sum() / len(top_k) if len(top_k) > 0 else 0.0
    
    # Recall@Top20% 
    
    total_positive = df['true'].sum()
    recall_top20 = top_k['true'].sum() / total_positive if total_positive > 0 else 0.0
    
    # Recall @ 5% FPR (auxiliary metric)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= 0.05)[0]
    recall_fpr5 = tpr[valid_indices[-1]] if len(valid_indices) > 0 else 0.0
    
    return precision_top20, recall_top20, recall_fpr5


def add_measurement_noise(X, noise_level=0.15):
    
    np.random.seed(42)  # Fixed noise seed
    noise = np.random.normal(0, noise_level, X.shape)
    return X * (1 + noise)


def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, rng_seed=42):
    
    rng = np.random.RandomState(rng_seed)
    indices = np.arange(len(y_pred))
    scores = []
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    for _ in range(n_bootstraps):
        indices_boot = rng.choice(indices, size=len(indices), replace=True)
        y_true_boot = y_true_np[indices_boot]
        y_pred_boot = y_pred_np[indices_boot]
        if len(np.unique(y_true_boot)) < 2: 
            continue
        score = roc_auc_score(y_true_boot, y_pred_boot)
        scores.append(score)
    
    ci_lower = np.percentile(scores, 2.5)
    ci_upper = np.percentile(scores, 97.5)
    return ci_lower, ci_upper


# ============================================================================
# 2. Main experiment program
# ============================================================================
def run_experiments():
    # --- A. Data loading ---
    csv_path = 'ESM_9.csv'
    if not os.path.exists(csv_path):
        print(f" Error: {csv_path} not found.")
        print("   Please run ESM_1.py to generate data first.")
        return

    print(f"Loading data: {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"   Records: {len(df):,} records")
    print(f"   Risky companies: {df['is_risky'].sum():,} ({df['is_risky'].mean()*100:.1f}%)")

    # --- B. Feature definition --- 
    LEVEL1_FEATURES = [
        'revenue', 'cost', 'profit', 'tax', 'asset', 'liability', 'equity',
        'profit_margin', 'tax_burden', 'asset_turnover','age'
    ]
    LEVEL2_FEATURES = [
        'vat_sales', 'income_gap_ratio', 'high_risk_invoice_ratio',
        'related_party_revenue_ratio', 'transfer_pricing_anomaly',
        'electricity_consumption', 'operating_expenses', 'rd_expenses_ratio'
    ]
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['industry', 'size'], drop_first=True)
    meta_cols = [c for c in df_encoded.columns if c.startswith('industry_') or c.startswith('size_')]
    
    cols_full = [f for f in LEVEL1_FEATURES + LEVEL2_FEATURES if f in df.columns] + meta_cols
    cols_priv = [f for f in LEVEL1_FEATURES if f in df.columns] + meta_cols
    
    X = df_encoded[cols_full].fillna(0)
    y = df_encoded['is_risky'].fillna(0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training set: {len(X_train):,}, Test set: {len(X_test):,}")

    # ========================================================================
    # Table 5: Model performance comparison
    # ========================================================================
    print("\n" + "="*100)
    print(" Table 5: Model performance comparison")
    print("="*100)
    print(f"{'Model':<22} | {'AUC':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'Prec@20%':<10} | {'Rec@20%':<10}")
    print("-"*100)
    
    baseline_metrics = {}
    table5_data = []

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(16,), alpha=10.0, max_iter=500, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                      eval_metric='logloss', random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        # Standardization
        if name in ['Logistic Regression', 'Neural Network', 'Naive Bayes']:
            scaler = StandardScaler()
            Xt_train = scaler.fit_transform(X_train)
            Xt_test = scaler.transform(X_test)
        else:
            Xt_train, Xt_test = X_train.values, X_test.values
            
        model.fit(Xt_train, y_train)
        trained_models[name] = (model, scaler if name in ['Logistic Regression', 'Neural Network', 'Naive Bayes'] else None)
        
        # Prediction
        y_prob = model.predict_proba(Xt_test)[:, 1]
        y_pred = model.predict(Xt_test)
        
        # Compute metrics
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        # Audit-specific metrics 
        prec_top20, rec_top20, rec_fpr5 = calculate_audit_metrics(y_test, y_prob)
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_auc(y_test, y_prob)
        
        if name == 'XGBoost':
            baseline_metrics = {'AUC': auc, 'Prec20': prec_top20, 'Rec20': rec_top20}
        
        print(f"{name:<22} | {auc:.4f}  | {f1:.4f}  | {prec:.4f}  | {rec:.4f}  | {prec_top20:.2%}     | {rec_top20:.2%}")
        
        table5_data.append({
            'Model': name,
            'AUC-ROC': round(auc, 4),
            'F1-Score': round(f1, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'Precision@Top20%': round(prec_top20, 4),  #  Audit hit rate
            'Recall@Top20%': round(rec_top20, 4),
            'CI_Lower': round(ci_lower, 4), 
            'CI_Upper': round(ci_upper, 4)
        })

    # save Table 5
    df_table5 = pd.DataFrame(table5_data)
    df_table5.to_csv('ESM_13.csv', index=False)
    print("-"*100)
    print(f" Save: ESM_13.csv")

    # ========================================================================
    # Table 6:Robustness Analysis
    # ========================================================================
    print("\n" + "="*100)
    print("Table 6: Robustness Analysis")
    print("="*100)
    print(f"{'Scenario':<25} | {'Features':<12} | {'Noise':<8} | {'AUC':<10} | {'Retention':<10}")
    print("-"*100)
    
    table6_data = []
    
    # Obtain XGBoost model
    xgb_model = trained_models['XGBoost'][0]
    
    # 1. Baseline (Full features, 0% noise)
    scenario = 'Baseline'
    print(f"{scenario:<25} | {'Full':<12} | {'0%':<8} | {baseline_metrics['AUC']:.4f}    | {'100.0%'}")
    table6_data.append({
        'Scenario': scenario, 
        'Features': 'Full', 
        'Noise': '0%', 
        'AUC': round(baseline_metrics['AUC'], 4), 
        'Retention': 1.0
    })
    
    # 2. Privacy Mode (L1 Only, 0% noise)
    scenario = 'Privacy Mode'
    X_train_p = X_train[cols_priv]
    X_test_p = X_test[cols_priv]
    
    model_priv = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 eval_metric='logloss', random_state=42)
    model_priv.fit(X_train_p, y_train)
    y_prob_p = model_priv.predict_proba(X_test_p)[:, 1]
    auc_p = roc_auc_score(y_test, y_prob_p)
    
    print(f"{scenario:<25} | {'L1 Only':<12} | {'0%':<8} | {auc_p:.4f}    | {auc_p/baseline_metrics['AUC']:.1%}")
    table6_data.append({
        'Scenario': scenario, 
        'Features': 'L1 Only', 
        'Noise': '0%', 
        'AUC': round(auc_p, 4), 
        'Retention': round(auc_p/baseline_metrics['AUC'], 4)
    })
    
    # 3. Data Noise (Full features, 15% noise)
    scenario = 'Data Noise'
    X_test_noisy = add_measurement_noise(X_test.values, 0.15)
    y_prob_n = xgb_model.predict_proba(X_test_noisy)[:, 1]
    auc_n = roc_auc_score(y_test, y_prob_n)
    
    print(f"{scenario:<25} | {'Full':<12} | {'15%':<8} | {auc_n:.4f}    | {auc_n/baseline_metrics['AUC']:.1%}")
    table6_data.append({
        'Scenario': scenario, 
        'Features': 'Full', 
        'Noise': '15%', 
        'AUC': round(auc_n, 4), 
        'Retention': round(auc_n/baseline_metrics['AUC'], 4)
    })
    
    # 4. Worst Case (L1 Only, 15% noise)
    scenario = 'Worst Case'
    X_test_p_noisy = add_measurement_noise(X_test_p.values, 0.15)
    y_prob_w = model_priv.predict_proba(X_test_p_noisy)[:, 1]
    auc_w = roc_auc_score(y_test, y_prob_w)
    
    print(f"{scenario:<25} | {'L1 Only':<12} | {'15%':<8} | {auc_w:.4f}    | {auc_w/baseline_metrics['AUC']:.1%}")
    table6_data.append({
        'Scenario': scenario, 
        'Features': 'L1 Only', 
        'Noise': '15%', 
        'AUC': round(auc_w, 4), 
        'Retention': round(auc_w/baseline_metrics['AUC'], 4)
    })
    
    print("-"*100)
    
    # Save Table 6
    df_table6 = pd.DataFrame(table6_data)
    df_table6.to_csv('ESM_14.csv', index=False)
    print(f" Save: ESM_14.csv")

    # ========================================================================
    # Print key data summary of the paper
    # ========================================================================
    print("\n" + "="*100)
    print(" Key Results Summary")
    print("="*100)
    
    lr_row = df_table5[df_table5['Model'] == 'Logistic Regression'].iloc[0]
    xgb_row = df_table5[df_table5['Model'] == 'XGBoost'].iloc[0]
    nn_row = df_table5[df_table5['Model'] == 'Neural Network'].iloc[0]
    
    print(f"""
[Table 5 key data]
- Logistic Regression AUC: {lr_row['AUC-ROC']:.4f}
- Neural Network AUC: {nn_row['AUC-ROC']:.4f}  
- XGBoost AUC: {xgb_row['AUC-ROC']:.4f}
- Performance Gap: {(xgb_row['AUC-ROC'] - lr_row['AUC-ROC'])*100:.2f} percentage points

[Precision@Top 20%]
- Logistic Regression: {lr_row['Precision@Top20%']*100:.2f}%
- XGBoost: {xgb_row['Precision@Top20%']*100:.2f}%
- Improvement: {(xgb_row['Precision@Top20%'] - lr_row['Precision@Top20%'])*100:.2f} percentage points

[Table 6 key data]
- Baseline AUC: {table6_data[0]['AUC']:.4f}
- Privacy Mode AUC: {table6_data[1]['AUC']:.4f}
- Data Noise AUC: {table6_data[2]['AUC']:.4f}
- Worst Case AUC: {table6_data[3]['AUC']:.4f}
""")
    print("="*100)


if __name__ == "__main__":
    run_experiments()
