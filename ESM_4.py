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
import xgboost as xgb
import os
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from ESM_1 import TDGMEnhanced, DEFAULT_CONFIG

# ==============================================================================
# 1. Core Evaluation Function (for Figure 4)
# ==============================================================================
def get_model_performance(data):
    
    features = [
        'revenue', 'cost', 'profit', 'tax', 'asset', 'liability', 'equity', 
        'profit_margin', 'tax_burden', 'asset_turnover', 'age',
        'vat_sales', 'income_gap_ratio', 'high_risk_invoice_ratio',
        'related_party_revenue_ratio', 'transfer_pricing_anomaly',
        'electricity_consumption', 'operating_expenses', 'rd_expenses_ratio' 
    ]
    
    df_encoded = pd.get_dummies(data, columns=['industry', 'size'], drop_first=True)
    feature_cols = features + [c for c in df_encoded.columns if c.startswith('industry_') or c.startswith('size_')]
    valid_cols = [c for c in feature_cols if c in df_encoded.columns]
    
    X = df_encoded[valid_cols].fillna(0)
    y = df_encoded['is_risky'].fillna(0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   eval_metric='logloss', random_state=42)
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
    
    return roc_auc_score(y_test, y_pred_lr), roc_auc_score(y_test, y_pred_xgb)


# ==============================================================================
# 2. Data Generation (for Figure 4)
# ==============================================================================
DATA_FILE = 'ESM_9.csv'

def generate_data_with_config(config_override, scenario_seed):
    
    full_config = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in config_override.items():
        if key == 'level2_features' and isinstance(value, dict):
            full_config['level2_features'].update(value)
        else:
            full_config[key] = value
    full_config['random_seed'] = scenario_seed
    full_config['num_companies'] = 10000
    
    l2 = full_config['level2_features']
    print(f"      invoice_fraud=[{l2['invoice_fraud_min']:.2f}, {l2['invoice_fraud_max']:.2f}], "
          f"related_party=[{l2['related_party_risky_min']:.2f}, {l2['related_party_risky_max']:.2f}], "
          f"underreport=[{l2['underreport_min']:.2f}, {l2['underreport_max']:.2f}]")
    
    tdgm = TDGMEnhanced(full_config)
    return tdgm.generate_data()


def load_baseline_data():
   
    if os.path.exists(DATA_FILE):
        print(f"      [Fast Load] Reading baseline data from {DATA_FILE}...")
        return pd.read_csv(DATA_FILE)
    else:
        print(f"      [Generate] Creating baseline data...")
        tdgm = TDGMEnhanced({'num_companies': 10000})
        data = tdgm.generate_data()
        data.to_csv(DATA_FILE, index=False)
        return data


# ==============================================================================
# 3. Figure 5: Environmental Invariance
# ==============================================================================
def run_prevalence_analysis():
    
    print("\n" + "="*70)
    print("  Analysis 1: Risk Prevalence (Figure 5)")
    print("  Method: Train FULL → Subsample TEST only → Multi-seed average")
    print("="*70)
    
    N_SEEDS = 30  # Number of random subsampling seeds for averaging
    
    # Step 1: Load baseline data
    base_data = load_baseline_data()
    base_data['is_risky'] = base_data['is_risky'].fillna(0).astype(int)
    base_prev = base_data['is_risky'].mean()
    print(f"   Baseline: {len(base_data):,} rows, prevalence = {base_prev:.2%}")
    
    # Step 2: Prepare features & train model ONCE on full training set
    features = [
        'revenue', 'cost', 'profit', 'tax', 'asset', 'liability', 'equity', 
        'profit_margin', 'tax_burden', 'asset_turnover', 'age',
        'vat_sales', 'income_gap_ratio', 'high_risk_invoice_ratio',
        'related_party_revenue_ratio','transfer_pricing_anomaly','electricity_consumption',
        'operating_expenses', 'rd_expenses_ratio' 
    ]
    df_enc = pd.get_dummies(base_data, columns=['industry', 'size'], drop_first=True)
    fcols = features + [c for c in df_enc.columns if c.startswith('industry_') or c.startswith('size_')]
    vcols = [c for c in fcols if c in df_enc.columns]
    X = df_enc[vcols].fillna(0)
    y = base_data['is_risky'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train LR
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(scaler.fit_transform(X_train), y_train)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
         eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    print(f"   Trained on: {len(X_train):,} rows (risky = {y_train.sum():,})")
    print(f"   Test pool:  {len(X_test):,} rows (risky = {y_test.sum():,})")
    
    
    prob_lr_full = lr.predict_proba(scaler.transform(X_test))[:, 1]
    prob_xgb_full = xgb_model.predict_proba(X_test)[:, 1]
    
   
    idx_to_pos = {idx: i for i, idx in enumerate(X_test.index)}
    risky_idx = y_test[y_test == 1].index.values
    normal_idx = y_test[y_test == 0].index.values
    
    # Step 3: For each target prevalence, subsample test set with multiple seeds
    print(f"\n   Evaluating with {N_SEEDS} random seeds per prevalence level...")
    print(f"   {'Prev':<6} {'Risky':<7} {'XGB mean':<10} {'XGB std':<10} {'LR mean':<10}")
    print("   " + "-"*50)
    
    target_rates = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    results = []
    
    for rate in target_rates:
        aucs_xgb = []
        aucs_lr = []
        
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            
            if rate < y_test.mean():
                
                need = min(int(len(normal_idx) * rate / (1 - rate)), len(risky_idx))
                sampled_risky = rng.choice(risky_idx, size=need, replace=False)
                test_idx = np.concatenate([normal_idx, sampled_risky])
            else:
               
                need = min(int(len(risky_idx) * (1 - rate) / rate), len(normal_idx))
                sampled_normal = rng.choice(normal_idx, size=need, replace=False)
                test_idx = np.concatenate([sampled_normal, risky_idx])
            
            yt = y_test.loc[test_idx]
            positions = [idx_to_pos[i] for i in test_idx]
            
            aucs_lr.append(roc_auc_score(yt, prob_lr_full[positions]))
            aucs_xgb.append(roc_auc_score(yt, prob_xgb_full[positions]))
        
        mean_xgb = np.mean(aucs_xgb)
        std_xgb = np.std(aucs_xgb)
        mean_lr = np.mean(aucs_lr)
        n_risky = need if rate < y_test.mean() else len(risky_idx)
        
        print(f"   {rate:>4.0%}   {n_risky:<7} {mean_xgb:<10.4f} {std_xgb:<10.4f} {mean_lr:<10.4f}")
        
        results.append({
            'Prevalence': rate * 100,
            'AUC_LR': round(mean_lr, 4),
            'AUC_XGB': round(mean_xgb, 4),
            'XGB_std': round(std_xgb, 4),
            'n_risky_test': n_risky,
        })
    
    print("   " + "="*50)
    df_r = pd.DataFrame(results)
    print(f"   XGB range: {df_r['AUC_XGB'].min():.4f} – {df_r['AUC_XGB'].max():.4f}")
    print(f"   XGB std across prevalences: {df_r['AUC_XGB'].std():.4f}")
    
    return df_r


# ==============================================================================
# 4. Figure 4: Signal Strength
# ==============================================================================
def run_signal_strength_analysis():
    
    print("\n" + "="*70)
    print("  Analysis 2: Signal Strength (Figure 4)")
    print("="*70)
    
    results = []
    FIXED_SEED = 42
    
    scenarios = {
        'Very Weak': {
            'invoice_fraud_min': 0.12, 'invoice_fraud_max': 0.25,
            'related_party_risky_min': 0.30, 'related_party_risky_max': 0.65,
            'underreport_min': 0.01, 'underreport_max': 0.025,
        }, 
        'Weak (Base)': {
            'invoice_fraud_min': 0.20, 'invoice_fraud_max': 0.35,
            'related_party_risky_min': 0.40, 'related_party_risky_max': 0.90,
            'underreport_min': 0.02, 'underreport_max': 0.05,
        }, 
        'Moderate': {
            'invoice_fraud_min': 0.32, 'invoice_fraud_max': 0.50,
            'related_party_risky_min': 0.55, 'related_party_risky_max': 0.95,
            'underreport_min': 0.04, 'underreport_max': 0.08,
        }, 
        'Strong': {
            'invoice_fraud_min': 0.45, 'invoice_fraud_max': 0.65,
            'related_party_risky_min': 0.70, 'related_party_risky_max': 0.98,
            'underreport_min': 0.08, 'underreport_max': 0.15,
        },
        'Very Strong': {
            'invoice_fraud_min': 0.60, 'invoice_fraud_max': 0.85,
            'related_party_risky_min': 0.85, 'related_party_risky_max': 0.99,
            'underreport_min': 0.15, 'underreport_max': 0.30,
        }
    }
    
    for name, params in scenarios.items():
        print(f"\n   -> Scenario: {name}")
        data = generate_data_with_config({'level2_features': params}, FIXED_SEED)
        auc_lr, auc_xgb = get_model_performance(data)
        results.append({'Signal Strength': name, 'AUC_LR': auc_lr, 'AUC_XGB': auc_xgb})
        print(f"      LR AUC: {auc_lr:.4f}, XGBoost AUC: {auc_xgb:.4f}")
        
    return pd.DataFrame(results)


# ==============================================================================
# 5. Save Results
# ==============================================================================
def safe_to_csv(df, filename):
    try:
        df.to_csv(filename, index=False)
        print(f"  Saved: {filename}")
    except PermissionError:
        backup = filename.replace('.csv', '_new.csv')
        df.to_csv(backup, index=False)
        print(f"  ⚠ {filename} locked, saved to: {backup}")


def save_results(df_prev, df_signal):
    # Figure 5 data
    df_onset = df_prev.copy()
    df_onset['onset'] = df_onset['Prevalence'] / 100.0
    df_onset = df_onset.rename(columns={'AUC_LR': 'lr_auc', 'AUC_XGB': 'xgb_auc'})
    df_onset = df_onset[['onset', 'lr_auc', 'xgb_auc', 'XGB_std']]
    safe_to_csv(df_onset, 'ESM_11.csv')

    # Figure 4 data
    df_effect = df_signal.copy()
    df_effect = df_effect.rename(columns={'Signal Strength': 'effect_level', 'AUC_XGB': 'xgb_auc'})
    d_map = {'Very Weak': 0.2, 'Weak (Base)': 0.4, 'Moderate': 0.6, 'Strong': 0.8, 'Very Strong': 1.0}
    df_effect['avg_d'] = df_effect['effect_level'].map(d_map)
    df_effect = df_effect[['effect_level', 'xgb_auc', 'avg_d']]
    safe_to_csv(df_effect, 'ESM_10.csv')


# ==============================================================================
# 6. Main
# ==============================================================================
if __name__ == "__main__":
    print("="*70)
    print("TDGM Sensitivity Analysis (v4: Correct Environmental Invariance)")
    print("="*70)
    
    df_prevalence = run_prevalence_analysis()
    df_signal = run_signal_strength_analysis()
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    print("\n[Figure 5: Risk Prevalence]")
    print(df_prevalence.to_string(index=False))
    
    print("\n[Figure 4: Signal Strength]")
    print(df_signal.to_string(index=False))
    
    # Verify Figure 4 trend
    auc_values = df_signal['AUC_XGB'].tolist()
    is_increasing = all(auc_values[i] <= auc_values[i+1] for i in range(len(auc_values)-1))
    print(f"\nFigure 4 monotonic: {'✓ Yes' if is_increasing else '✗ No'}")
    
    # Save
    save_results(df_prevalence, df_signal)
    
    # Key data for paper
    print("\n" + "="*70)
    print("Key Data for Paper")
    print("="*70)
    
    row_1 = df_prevalence[df_prevalence['Prevalence']==1].iloc[0]
    row_3 = df_prevalence[df_prevalence['Prevalence']==3].iloc[0]
    
    print(f"""
[Figure 5]
  1% prevalence: XGB AUC = {row_1['AUC_XGB']:.4f} (±{row_1['XGB_std']:.4f}), ~{row_1['n_risky_test']} evaders
  3% prevalence: XGB AUC = {row_3['AUC_XGB']:.4f} (±{row_3['XGB_std']:.4f}), ~{row_3['n_risky_test']} evaders
  XGB range: {df_prevalence['AUC_XGB'].min():.4f} – {df_prevalence['AUC_XGB'].max():.4f}

  → Paper: "XGBoost maintained AUC ≈ 0.97 across all prevalence levels (1%-20%),
     with {row_1['n_risky_test']} evaders at 1% prevalence"

[Figure 4]
  Weak (Base): {df_signal[df_signal['Signal Strength']=='Weak (Base)']['AUC_XGB'].values[0]:.4f}
  Strong:      {df_signal[df_signal['Signal Strength']=='Strong']['AUC_XGB'].values[0]:.4f}
  Very Strong: {df_signal[df_signal['Signal Strength']=='Very Strong']['AUC_XGB'].values[0]:.4f}
""")
    print("="*70)
