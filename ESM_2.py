# =============================================================
# Article : Mechanism-Guided Synthetic Tax Data Generation:
#            A Computational Framework for Tax Evasion Detection
# Authors : Xiaojing Fan
# Affiliation: Business School, University of Shanghai for
#              Science and Technology, Shanghai, China
# Corresponding author: fanxiaojing@usst.edu.cn
# =============================================================

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
#  Temporal validation module
# ============================================================================
class TDGMTemporalValidator:
    
    
    def __init__(self, num_companies=2000, num_years=30, ar_coefficient=0.85, noise_std=0.12):
        self.num_companies = num_companies
        self.num_years = num_years
        self.rho = ar_coefficient
        self.sigma = noise_std
        np.random.seed(42)
        
    def generate(self):
        
        
        # 1. Generate initial company size (Pareto distribution)
        a = 1.05
        min_revenue = 100000
        initial_revenues = (np.random.pareto(a, self.num_companies) + 1) * min_revenue
        
        records = []
        true_mus = {}  # Store true long-term mean μ_i for each firm
        
        for company_id in range(self.num_companies):
            # Long-term mean (log domain)
            mu = np.log(initial_revenues[company_id])
            true_mus[company_id] = mu
            current_log_rev = mu
            
            for year in range(self.num_years):
                # Mean-Reverting AR(1): ln(R_t) = ρ·ln(R_{t-1}) + (1-ρ)·μ + ε
                epsilon = np.random.normal(0, self.sigma)
                current_log_rev = self.rho * current_log_rev + (1 - self.rho) * mu + epsilon
                revenue = np.exp(current_log_rev)
                
                # Generate supporting financial data (simplified version)
                cost_ratio = 0.65 + np.random.normal(0, 0.05)
                cost = revenue * np.clip(cost_ratio, 0.4, 0.85)
                profit = revenue - cost
                
                asset = revenue * (1.5 + np.random.normal(0, 0.3))
                debt_ratio = np.clip(0.60 + np.random.normal(0, 0.15), 0.3, 0.8)
                liability = asset * debt_ratio
                equity = asset - liability 
                
                records.append({
                    'FirmID': company_id,
                    'Year': year,
                    'Revenue': revenue,
                    'Assets': asset,
                    'Liabilities': liability,
                    'Equity': equity
                })
        
        return pd.DataFrame(records), true_mus


# ============================================================================
# Evaluation Function
# ============================================================================
def check_constraints(df):
    
    diff = np.abs(df['Assets'] - (df['Liabilities'] + df['Equity']))
    violation_rate = (diff > 1.0).mean()
    return (1 - violation_rate) * 100


def check_temporal_corr_true_mu(df, true_mus):
    
    df_sorted = df.sort_values(['FirmID', 'Year']).copy()
    avg_T = df_sorted.groupby('FirmID')['Year'].count().mean()
    
    # Transform to log domain
    df_sorted['Log_Revenue'] = np.log(df_sorted['Revenue'].clip(lower=1))
    
    # Center using true μ_i (no Nickell bias)
    df_sorted['True_Mu'] = df_sorted['FirmID'].map(true_mus)
    df_sorted['Log_Rev_Centered'] = df_sorted['Log_Revenue'] - df_sorted['True_Mu']
    df_sorted['Prev_Log_Rev_Centered'] = df_sorted.groupby('FirmID')['Log_Rev_Centered'].shift(1)
    
    valid = df_sorted[['Log_Rev_Centered', 'Prev_Log_Rev_Centered']].dropna()
    if len(valid) < 2:
        return 0.0, avg_T
    
    rho_est = valid.corr().iloc[0, 1]
    return rho_est, avg_T


def check_temporal_corr_within_firm(df, min_T_for_correction=10):
   
    df_sorted = df.sort_values(['FirmID', 'Year']).copy()
    
    # Check average T
    avg_T = df_sorted.groupby('FirmID')['Year'].count().mean()
    
    # Transform to Log domain
    df_sorted['Log_Revenue'] = np.log(df_sorted['Revenue'].clip(lower=1))
    
    # Center by FirmID (sample mean)
    df_sorted['Log_Rev_Centered'] = df_sorted.groupby('FirmID')['Log_Revenue'].transform(
        lambda x: x - x.mean()
    )
    df_sorted['Prev_Log_Rev_Centered'] = df_sorted.groupby('FirmID')['Log_Rev_Centered'].shift(1)
    
    valid = df_sorted[['Log_Rev_Centered', 'Prev_Log_Rev_Centered']].dropna()
    if len(valid) < 2:
        return 0.0, avg_T
    
    rho_obs = valid.corr().iloc[0, 1]
    
    # Apply Nickell bias correction
    # Nickell (1981): E[ρ̂] ≈ ρ - (1+ρ)/(T-1)
    if avg_T >= min_T_for_correction:
        rho_corrected = rho_obs + (1 + rho_obs) / (avg_T - 1)
        rho_corrected = min(max(rho_corrected, 0), 0.99)
        return rho_corrected, avg_T
    
    return rho_obs, avg_T


def check_temporal_corr_global(df):
    
    df_sorted = df.sort_values(['FirmID', 'Year']).copy()
    df_sorted['Log_Revenue'] = np.log(df_sorted['Revenue'].clip(lower=1))
    df_sorted['Prev_Log_Revenue'] = df_sorted.groupby('FirmID')['Log_Revenue'].shift(1)
    corr = df_sorted[['Log_Revenue', 'Prev_Log_Revenue']].dropna().corr().iloc[0, 1]
    return corr if not np.isnan(corr) else 0.0


# ============================================================================
# Main program
# ============================================================================
def main():
    print("=" * 75)
    print("TDGM vs CTGAN: Mechanism-guided vs. Pure Data-driven Comparison Experiment")
    print("( Long time-series for AR(1) coefficient validation)")
    print("=" * 75)

    # ==========================================
    # 1. Generate long-term TDGM time series data (for temporal validation)
    # ==========================================
    print("\n[1/5] Generating TDGM long time-series data (T=30)...")
    
    tdgm_validator = TDGMTemporalValidator(
        num_companies=10000, 
        num_years=30,        # Key: Use 30 years to avoid Nickell bias
        ar_coefficient=0.85,
        noise_std=0.12
    )
    tdgm_long_data, tdgm_true_mus = tdgm_validator.generate()
    print(f"      Generated {len(tdgm_long_data):,} rows ({tdgm_validator.num_companies} firms × {tdgm_validator.num_years} years)")

    # ==========================================
    # 2. Read original TDGM data (for accounting equation verification)
    # ==========================================
    print("\n[2/5] Loading original TDGM data (T=3)...")
    
    filename = 'ESM_9.csv'
    try:
        real_df = pd.read_csv(filename)
        tdgm_short_data = pd.DataFrame({
            'FirmID': real_df['company_id'],
            'Year': real_df['year'],
            'Assets': real_df['asset'],
            'Liabilities': real_df['liability'],
            'Equity': real_df['equity'],
            'Revenue': real_df['revenue']
        })
        print(f"      Loaded {len(tdgm_short_data):,} rows")
    except FileNotFoundError:
        print(f"      Warning: {filename} not found, using generated data for all tests")
        tdgm_short_data = tdgm_long_data[tdgm_long_data['Year'] < 3].copy()

    # ==========================================
    # 3.Train CTGAN (using long time series data)
    # ==========================================
    print("\n[3/5] Training CTGAN on long time-series data...")
    
    # Train CTGAN using long sequence data identical to TDGM
    training_data = tdgm_long_data.copy()
    print(f"      Training on {len(training_data):,} rows...")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(training_data)

    synthesizer = CTGANSynthesizer(metadata, epochs=50, verbose=False)
    synthesizer.fit(training_data)

    print("      Generating synthetic samples...")
    ctgan_long_data = synthesizer.sample(num_rows=len(training_data))

    # ==========================================
    # 4. Evaluate Comparison Metrics
    # ==========================================
    print("\n[4/5] Evaluating Metrics...")

    # 4.1 Accounting Equation Validity (using original T=3 data, more strict)
    tdgm_validity = check_constraints(tdgm_short_data)
    
    # Test constraint validity on CTGAN short-horizon data    
    ctgan_short_training = tdgm_short_data[tdgm_short_data['FirmID'] <= 10000].copy()
    metadata_short = SingleTableMetadata()
    metadata_short.detect_from_dataframe(ctgan_short_training)
    synthesizer_short = CTGANSynthesizer(metadata_short, epochs=50, verbose=False)
    synthesizer_short.fit(ctgan_short_training)
    ctgan_short_data = synthesizer_short.sample(num_rows=len(ctgan_short_training))
    ctgan_validity = check_constraints(ctgan_short_data)

    # 4.2 Temporal Autocorrelation (using long time series data)
    # TDGM: use true μ_i centering (no Nickell bias)
    tdgm_rho, tdgm_T = check_temporal_corr_true_mu(tdgm_long_data, tdgm_true_mus)
    # CTGAN: use sample-mean centering + Nickell correction (true μ unknown)
    ctgan_rho, ctgan_T = check_temporal_corr_within_firm(ctgan_long_data)
    
    tdgm_global = check_temporal_corr_global(tdgm_long_data)
    ctgan_global = check_temporal_corr_global(ctgan_long_data)

    # ==========================================
    # 5. Output Results (Table 4)
    # ==========================================
    print("\n[5/5] Results")

    print("\n" + "=" * 80)
    print("Main Results")
    print("=" * 80)
    print(f"{'Metric':<45} | {'TDGM (Ours)':<15} | {'CTGAN':<15}")
    print("-" * 80)
    print(f"{'Constraint Validity (% on T=3 data)':<45} | {tdgm_validity:>6.1f}%          | {ctgan_validity:>6.1f}%")
    print(f"{'Temporal AR(1) Coefficient (on T=30 data)':<45} | {tdgm_rho:>8.4f}         | {ctgan_rho:>8.4f}")
    print(f"{'Global Log-Correlation (on T=30 data)':<45} | {tdgm_global:>8.4f}         | {ctgan_global:>8.4f}")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("Parameter validation")
    print("-" * 80)
    print(f"AR(1) coefficient set by TDGM ρ = 0.85")
    print(f"AR(1) coefficient estimated ρ̂ = {tdgm_rho:.4f} (true-μ centering, T={tdgm_T:.0f} years)")
    print(f"Validation Result: {'✓ Pass' if abs(tdgm_rho - 0.85) < 0.05 else '✗ Deviation is large'}")
    print(f"Note: Since true μ_i is known in synthetic data, we center using μ_i directly,")
    print(f"      avoiding Nickell bias from within-group demeaning.")
    print("-" * 80)

       
    # ==========================================
    # 6. Save Table 4 results to CSV
    # ==========================================
    table4_data = [
        {
            'Metric': 'Constraint Validity (Assets ≡ L + E)',
            'Data_Source': 'T=3 data',
            'TDGM': round(tdgm_validity, 4),
            'CTGAN': round(ctgan_validity, 4),
        },
        {
            'Metric': 'Temporal AR(1) Coefficient',
            'Data_Source': 'T=30 data',
            'TDGM': round(tdgm_rho, 4),
            'CTGAN': round(ctgan_rho, 4),
        },
        {
            'Metric': 'Global Log-Correlation',
            'Data_Source': 'T=30 data',
            'TDGM': round(tdgm_global, 4),
            'CTGAN': round(ctgan_global, 4),
        },
    ]

    df_table4 = pd.DataFrame(table4_data)
    df_table4.to_csv('ESM_12.csv', index=False)
    print("\nSaved: ESM_12.csv")

if __name__ == "__main__":
    main()
