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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Configuration parameters
# ============================================================================

DEFAULT_CONFIG = {
    # Basic Configuration (Keep Unchanged)
    'num_companies': 10000, 
    'num_years': 3,
    'random_seed': 42,
    'industries': ['Manufacturing', 'Retail', 'Services', 'Construction', 'Technology'],
    'industry_probs': [0.25, 0.20, 0.30, 0.15, 0.10],
    'company_sizes': ['Small', 'Medium', 'Large'],
    'size_probs': [0.60, 0.30, 0.10],
    'revenue_ranges': {'Small': (1e5, 1e6), 'Medium': (1e6, 1e7), 'Large': (1e7, 1e8)},
    'ar_coefficient': 0.85, 
    'noise_std': 0.12,
    
    # Risk parameters (unchanged)
    'risk_onset': 0.03, 
    'risk_persistence': 0.80,
    'risk_types': ['Revenue Suppression', 'Cost Inflation', 'Transfer Manipulation', 'Shell Company'],
    'risk_type_probs': [0.40, 0.30, 0.20, 0.10],
    'size_risk_ratios': {'Small': 0.18, 'Medium': 0.15, 'Large': 0.12},

    # Level 2 parameters
        'level2_features': {
        # 1. Measurement noise regresses normally 
        'vat_measurement_noise': 0.10,      

        # 2.The Grey Zone
        # Normal companies
        'invoice_baseline_min': 0.05, 
        'invoice_baseline_max': 0.30,
        
        # Risk enterprises
        'invoice_fraud_min': 0.20,            
        'invoice_fraud_max': 0.35,            

        # 3. Related party transactions
        'related_party_normal_max': 0.75,
        'related_party_risky_min': 0.40,     
        'related_party_risky_max': 0.90,
        
        # 4. Auxiliary Feature
        'electricity_noise': 0.30,
        'underreport_min': 0.02,             
        'underreport_max': 0.05,
        'operating_expenses': 0.2
    },
    'use_burnin': True, 
    'burnin_years': 7,
}
# ============================================================================
# TDGM main class
# ============================================================================

class TDGMEnhanced:
       
    def __init__(self, config: Optional[Dict] = None):
        
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        np.random.seed(self.config['random_seed'])
        
        
        print()
        print(f"Configuration:")
        print(f"  - Number of enterprises:{self.config['num_companies']:,}")
        print(f"  - Number of years:{self.config['num_years']}")
        print(f"  - Whether to use burn-in:{self.config['use_burnin']}")
        if self.config['use_burnin']:
            print(f"  - Burn-in years:{self.config['burnin_years']}")
        
        print()
    
    def generate_data(self) -> pd.DataFrame:
       
        print("[1/4] Step 1: Initialize enterprise attributes...")
        companies = self._initialize_companies()
        print(f"      ✓ Completed, generated {len(companies):,} enterprises")
        
        print("[2/4] Step 2: Temporal evolution...")
        panel_data = self._temporal_evolution(companies)
        print(f"      ✓ Completed, generated {len(panel_data):,} observations")
        
        print("[3/4] Step 3: Risk injection...")
        panel_data = self._risk_injection(panel_data)
        risky_count = panel_data['is_risky'].sum()
        risky_pct = risky_count / len(panel_data) * 100
        print(f"      ✓ Completed, risk samples:{risky_count:,} ({risky_pct:.2f}%)")
        
        print("[4/4] Step 4: Generate observation features (Level 2)...")
        panel_data = self._observation_generation(panel_data)
        print(f"      ✓ Completed, generated Level 2 features")
        
        print()
        print("="*80)
        print("Data generation completed")
        print("="*80)
        print(f"Total number of observations:{len(panel_data):,}")
        print(f"Number of features:{len(panel_data.columns)}")
        print(f"  - Level 1 features:13(Basic Finance)")
        print(f"  - Level 2 features:8(Audit Signals)")
        print(f"  - Label features:2(is_risky, risk_type)")
        print()
        
        return panel_data
    
    # ========================================================================
    # Step 1: Initialize enterprise attributes
    # ========================================================================
    
    def _initialize_companies(self) -> pd.DataFrame:
        """Initialize enterprise attributes """
        num_companies = self.config['num_companies']
        
        # 1.Generate revenue directly using the Pareto distribution
        a = 1.05 
        min_revenue = 100000
        revenues = (np.random.pareto(a, num_companies) + 1) * min_revenue
        
        # 2. Define scale labels based on income (Small/Medium/Large)
        thresholds = np.percentile(revenues, [60, 90])
        
        def get_size(r):
            if r <= thresholds[0]: return 'Small'
            elif r <= thresholds[1]: return 'Medium'
            else: return 'Large'
            
        sizes = [get_size(r) for r in revenues]
        
        companies = pd.DataFrame({
            'company_id': range(1, num_companies + 1),
            'industry': np.random.choice(
                self.config['industries'],
                size=num_companies,
                p=self.config['industry_probs']
            ),
            'size': sizes,  
            'age': np.random.randint(1, 31, size=num_companies),
            'initial_revenue': revenues 
        })
        
        return companies
           
    
    # ========================================================================
    # Step 2: Temporal evolution
    # ========================================================================

    def _temporal_evolution(self, companies: pd.DataFrame) -> pd.DataFrame:
        """AR(1) time series evolution """
        records = []

        use_burnin = self.config['use_burnin']
        burnin_years = self.config['burnin_years']
        observation_years = self.config['num_years']
        total_years = (burnin_years if use_burnin else 0) + observation_years

        # 1. Get parameters
        rho = self.config['ar_coefficient']  
        sigma = self.config['noise_std']  # 0.12

        for _, company in companies.iterrows():
            # 2. Set long-term mean mu (i.e., initial generated revenue)
            # We assume the initial generated revenue is the long-term equilibrium level
            long_term_mean_log = np.log(max(company['initial_revenue'], 1000))

            # 3. Initialize current state (starting from long-term mean)    
            current_log_revenue = long_term_mean_log

            for year in range(total_years):
               
                epsilon = np.random.normal(0, sigma)
                current_log_revenue = (
                        rho * current_log_revenue +
                        (1 - rho) * long_term_mean_log +
                        epsilon
                )

                # 4. Transform back to absolute value
                # Transform back to absolute value
                revenue = np.exp(current_log_revenue)
                
                # 5. Ensure non-negative/ logical lower bound
                revenue = max(revenue, 1000)

                
                if use_burnin and year < burnin_years:
                    continue

                actual_year = year - (burnin_years if use_burnin else 0)
                record = self._generate_basic_financials(company, revenue, actual_year)
                records.append(record)

        return pd.DataFrame(records)
    
    def _generate_basic_financials(
        self, 
        company: pd.Series, 
        revenue: float, 
        year: int
    ) -> Dict:
        """Generate basic financial metrics (Level 1 features' true values)"""
        
        # Industry benchmark parameters
        industry_params = {
            'Manufacturing': {'cost_ratio': 0.65, 'profit_margin': 0.08},
            'Retail': {'cost_ratio': 0.70, 'profit_margin': 0.05},
            'Services': {'cost_ratio': 0.55, 'profit_margin': 0.12},
            'Construction': {'cost_ratio': 0.68, 'profit_margin': 0.07},
            'Technology': {'cost_ratio': 0.50, 'profit_margin': 0.15},
        }
        
        params = industry_params[company['industry']]
        
        # Cost
        cost_ratio = params['cost_ratio'] + np.random.normal(0, 0.05)
        cost_ratio = np.clip(cost_ratio, 0.4, 0.85)
        cost = revenue * cost_ratio
        
        # 6. Profit
        profit = revenue - cost
        
        # 7. Tax (based on profit)
        tax_rate = 0.25 + np.random.normal(0, 0.03)
        tax_rate = np.clip(tax_rate, 0.15, 0.35)
        tax = max(profit * tax_rate, 0)
        
        # 8.Asset and Liability Statement
        # Industry capital intensity
        industry_capital_intensity = {
            'Manufacturing': 1.5,
            'Retail': 1.2,
            'Services': 0.8,
            'Construction': 1.3,
            'Technology': 1.0,
        }
        phi_k = industry_capital_intensity[company['industry']]
        asset = revenue * (phi_k + np.random.normal(0, 0.2))    

        
        debt_ratio = 0.60 + np.random.normal(0, 0.15)
        debt_ratio = np.clip(debt_ratio, 0.3, 0.8)
        liability = asset * debt_ratio
        equity = asset - liability
        
        # Financial Ratio
        profit_margin = profit / revenue if revenue > 0 else 0
        tax_burden = tax / revenue if revenue > 0 else 0
        asset_turnover = revenue / asset if asset > 0 else 0
        
        return {
            'company_id': company['company_id'],
            'industry': company['industry'],
            'size': company['size'],
            'age': company['age'],
            'year': year,
            
            # 9. True Financial Data
            'true_revenue': revenue,
            'true_cost': cost,
            'true_profit': profit,
            'true_tax': tax,
            
            # Level 1 Features
            'revenue': revenue,
            'cost': cost,
            'profit': profit,
            'tax': tax,
            'asset': asset,
            'liability': liability,
            'equity': equity,
            'profit_margin': profit_margin,
            'tax_burden': tax_burden,
            'asset_turnover': asset_turnover,
        }
    
    # ========================================================================
    # Step 3: Risk injection (Markov chain-driven)
    # ========================================================================
    
    def _risk_injection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Markov chain-driven risk injection"""
        
        # 1. Initialize risk states
        df = df.sort_values(['company_id', 'year']).reset_index(drop=True)
        df['is_risky'] = 0
        df['risk_type'] = None
        
        # 2. For each company, independently evolve risk states
        for company_id in df['company_id'].unique():
            mask = df['company_id'] == company_id
            company_data = df[mask].copy()
            
            # 3. Initial risk probability
            size = company_data.iloc[0]['size']
            initial_risk_prob = self.config['size_risk_ratios'][size]
            
            # 4. Initial state  
            is_risky = np.random.random() < initial_risk_prob
            risk_type = None
            
            if is_risky:
                risk_type = np.random.choice(
                    self.config['risk_types'],
                    p=self.config['risk_type_probs']
                )
            
            # 5. Record the status each year
            risk_states = []
            risk_types = []
            
            for _ in range(len(company_data)):
                risk_states.append(is_risky)
                risk_types.append(risk_type if is_risky else None)
                
                # 6. Markov transition  
                if is_risky:
                    #Risk status persists
                    is_risky = np.random.random() < self.config['risk_persistence']
                    if not is_risky:
                        risk_type = None
                else:
                    # Enter risk state
                    is_risky = np.random.random() < self.config['risk_onset']
                    if is_risky:
                        risk_type = np.random.choice(
                            self.config['risk_types'],
                            p=self.config['risk_type_probs']
                        )
            
            # 7. Update DataFrame
            df.loc[mask, 'is_risky'] = risk_states
            df.loc[mask, 'risk_type'] = risk_types
        
        # 8. Apply risk manipulation logic
        df = self._apply_risk_manipulations(df)
        
        return df
    
    def _apply_risk_manipulations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply specific risk manipulation behaviors"""    
        
        risky_mask = df['is_risky'] == 1
        
       
        rev_manip = risky_mask & (df['risk_type'] == 'Revenue Suppression')
        if rev_manip.any():
            # Get the underreport parameter
            underreport_min = self.config['level2_features'].get('underreport_min', 0.02)
            underreport_max = self.config['level2_features'].get('underreport_max', 0.05)
            underreport_rate = np.random.uniform(underreport_min, underreport_max, rev_manip.sum()) 
            df.loc[rev_manip, 'revenue'] = (
                df.loc[rev_manip, 'true_revenue'] * (1 - underreport_rate)
            )
            
            
            df.loc[rev_manip, 'profit'] = df.loc[rev_manip, 'revenue'] - df.loc[rev_manip, 'cost']
            df.loc[rev_manip, 'tax'] = np.maximum(df.loc[rev_manip, 'profit'] * 0.25, 0)
        
        # Cost Inflation - Increasing costs 
        cost_manip = risky_mask & (df['risk_type'] == 'Cost Inflation')
        if cost_manip.any():
            inflation_rate = np.random.uniform(0.25, 0.40, cost_manip.sum())  
            df.loc[cost_manip, 'cost'] = (
                df.loc[cost_manip, 'true_cost'] * (1 + inflation_rate)
            )
            df.loc[cost_manip, 'profit'] = df.loc[cost_manip, 'revenue'] - df.loc[cost_manip, 'cost']
            df.loc[cost_manip, 'tax'] = np.maximum(df.loc[cost_manip, 'profit'] * 0.25, 0)
        
        # Transfer Manipulation
        
        
        agg_acct = risky_mask & (df['risk_type'] == 'Transfer Manipulation')
        if agg_acct.any():
            shift_rate = np.random.uniform(0.20, 0.35, agg_acct.sum())  
            # Shift profit by inflating cost 
            shifted_amount = df.loc[agg_acct, 'profit'] * shift_rate
            df.loc[agg_acct, 'cost'] = df.loc[agg_acct, 'cost'] + shifted_amount
            # Profit re-derived from identity
            df.loc[agg_acct, 'profit'] = df.loc[agg_acct, 'revenue'] - df.loc[agg_acct, 'cost']
            df.loc[agg_acct, 'tax'] = np.maximum(df.loc[agg_acct, 'profit'] * 0.25, 0)
        
        # Shell Company -Shell company 
        shell = risky_mask & (df['risk_type'] == 'Shell Company')
        if shell.any():
            # An abnormally high balance sheet 
            df.loc[shell, 'asset'] = df.loc[shell, 'liability'] * 100
            df.loc[shell, 'equity'] = df.loc[shell, 'asset'] - df.loc[shell, 'liability']

            # An abnormally low cost structure
            df.loc[shell, 'cost'] = df.loc[shell, 'revenue'] * 0.95
            df.loc[shell, 'profit'] = df.loc[shell, 'revenue'] * 0.05
            df.loc[shell, 'tax'] = df.loc[shell, 'profit'] * 0.10  # Abnormally low tax rate
        
        # Recalculate financial ratios
        df['profit_margin'] = df['profit'] / df['revenue']
        df['tax_burden'] = df['tax'] / df['revenue']
        df['asset_turnover'] = df['revenue'] / df['asset']
        
        return df
    
    # ========================================================================
    # Step 4: Observation generation (Level 2 features) 
    # ========================================================================
    
    def _observation_generation(self, df: pd.DataFrame) -> pd.DataFrame:
                
        print("      VAT sales...", end=" ")
        df = self._generate_vat_sales(df)
        print("✓")
        
        print("      Revenue gap ratio...", end=" ")
        df = self._calculate_income_gap_ratio(df)
        print("✓")
        
        print("      Invoice system signals...", end=" ")
        df = self._generate_invoice_signals(df)
        print("✓")
        
        print("      Related party transactions...", end=" ")
        df = self._generate_related_party_transactions(df)
        print("✓")
        
        print("      Other audit characteristics...", end=" ")
        df = self._generate_additional_features(df)
        print("✓")
        
        return df
    
    def _generate_vat_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        
        noise_std = self.config['level2_features']['vat_measurement_noise']
        
        # Benchmark: VAT sales = actual revenue + noise
        measurement_noise = np.random.normal(1.0, noise_std, len(df))
        df['vat_sales'] = df['true_revenue'] * measurement_noise
        
        # Ensure non-negative values
        df['vat_sales'] = df['vat_sales'].clip(lower=0)
        
        return df
    
    def _calculate_income_gap_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df['income_gap_ratio'] = (
            (df['vat_sales'] - df['revenue']) / df['revenue']
        )
        
        # Handle extreme values
        df['income_gap_ratio'] = df['income_gap_ratio'].clip(-0.5, 1.0)
        
        return df
    
    def _generate_invoice_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        
        baseline_min = self.config['level2_features']['invoice_baseline_min']
        baseline_max = self.config['level2_features']['invoice_baseline_max']
        fraud_min = self.config['level2_features']['invoice_fraud_min']
        fraud_max = self.config['level2_features']['invoice_fraud_max']
        
        # Default: Normal level 
        df['high_risk_invoice_ratio'] = np.random.uniform(
            baseline_min, baseline_max, len(df)
        )
        
        # Companies with inflated costs
        cost_manip = (df['is_risky'] == 1) & (df['risk_type'] == 'Cost Inflation')
        if cost_manip.any():
            # Proportion of fraudulent invoices
            df.loc[cost_manip, 'high_risk_invoice_ratio'] = np.random.uniform(
                fraud_min, fraud_max, cost_manip.sum()
            )
        
        # Other risky companies
        other_risky = (df['is_risky'] == 1) & (df['risk_type'] != 'Cost Inflation')
        if other_risky.any():
            df.loc[other_risky, 'high_risk_invoice_ratio'] = np.random.uniform(
                0.05, 0.15, other_risky.sum()
            )
        
        # Ensure the scope is reasonable
        df['high_risk_invoice_ratio'] = df['high_risk_invoice_ratio'].clip(0, 0.95)
        
        return df
    
    def _generate_related_party_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
       
        normal_max = self.config['level2_features']['related_party_normal_max']
        risky_min = self.config['level2_features']['related_party_risky_min']
        risky_max = self.config['level2_features']['related_party_risky_max']
        
        # Default: Normal level
        df['related_party_revenue_ratio'] = np.random.uniform(
            0.05, normal_max, len(df)
        )
        
        # Transfer Manipulation company
        agg_acct = (df['is_risky'] == 1) & (df['risk_type'] == 'Transfer Manipulation')
        if agg_acct.any():
            df.loc[agg_acct, 'related_party_revenue_ratio'] = np.random.uniform(
                risky_min, risky_max, agg_acct.sum()  
            )
            
            # Transfer Pricing Anomalies: Degree of Deviation from Market Price 
            df.loc[agg_acct, 'transfer_pricing_anomaly'] = np.random.uniform(
                0.15, 0.35, agg_acct.sum()  
            )
        
        # Non-Transfer Manipulation firm
        df['transfer_pricing_anomaly'] = df.get('transfer_pricing_anomaly', 0)
        normal_mask = ~agg_acct
        if normal_mask.any():
            df.loc[normal_mask, 'transfer_pricing_anomaly'] = np.random.uniform(
                0, 0.08, normal_mask.sum()
            )
        
        return df
    
    def _generate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
       
        
        # 1. Electricity consumption 
        # Estimate normal electricity consumption based on industry and income
        industry_electricity_intensity = {
            'Manufacturing': 0.15,   # High energy consumption
            'Retail': 0.03,          # Low energy consumption
            'Services': 0.05,        # Moderate
            'Construction': 0.08,    # Medium-high
            'Technology': 0.06,      # Moderate
        }
        
        df['expected_electricity'] = df.apply(
            lambda row: row['true_revenue'] * industry_electricity_intensity[row['industry']],
            axis=1
        )
        
        noise_std = self.config['level2_features']['electricity_noise']
        measurement_noise = np.random.normal(1.0, noise_std, len(df))
        df['electricity_consumption'] = df['expected_electricity'] * measurement_noise
        
        # Shell company: electricity consumption significantly lower than expected
        shell = (df['is_risky'] == 1) & (df['risk_type'] == 'Shell Company')
        if shell.any():
            df.loc[shell, 'electricity_consumption'] = (
                df.loc[shell, 'expected_electricity'] * np.random.uniform(0.1, 0.3, shell.sum())
            )
        
        df['electricity_consumption'] = df['electricity_consumption'].clip(lower=0)
        
        # 2. Operating expenses (cross-checked with output)
    
        
        op_exp_ratio = self.config['level2_features'].get('operating_expenses', 0.2)
        low = max(0.05, op_exp_ratio - 0.10)
        high = min(0.45, op_exp_ratio + 0.05)
        df['operating_expenses'] = df['revenue'] * np.random.uniform(low, high, len(df))
        
        # 3. R&D expense ratio (mainly for Technology industry)
        df['rd_expenses_ratio'] = 0.0
        
        tech_mask = df['industry'] == 'Technology'
        if tech_mask.any():
            df.loc[tech_mask, 'rd_expenses_ratio'] = np.random.uniform(
                0.05, 0.20, tech_mask.sum()
            )
        
        # Other industries: little or no research and development
        other_mask = ~tech_mask
        if other_mask.any():
            df.loc[other_mask, 'rd_expenses_ratio'] = np.random.uniform(
                0, 0.05, other_mask.sum()
            )
        
        return df
    
    # ========================================================================
    # Data Export
    # ========================================================================
    
    def export_data(
        self, 
        df: pd.DataFrame, 
        filename: str = 'ESM_9.csv'
    ) -> None:
        """Export data to CSV"""
        
        
        level1_features = [
            'company_id', 'industry', 'size', 'age', 'year',
            'revenue', 'cost', 'profit', 'tax',
            'asset', 'liability', 'equity',
            'profit_margin', 'tax_burden', 'asset_turnover'
        ]
        
        level2_features = [
            'vat_sales', 'income_gap_ratio',
            'high_risk_invoice_ratio',
            'related_party_revenue_ratio', 'transfer_pricing_anomaly',
            'electricity_consumption', 'operating_expenses', 'rd_expenses_ratio'
        ]
        
        labels = ['is_risky', 'risk_type']
        
        export_columns = level1_features + level2_features + labels
        
        
        available_columns = [col for col in export_columns if col in df.columns]
        
        df[available_columns].to_csv(filename, index=False)
        print(f"✓ Data exported to: {filename}")
        print(f"  - Sample size: {len(df):,}")
        print(f"  - Feature count: {len(available_columns) - 2}")  
        print(f"  - Level 1: 13 (10 numeric + industry, size, age)")
        print(f"  - Level 2: {len(level2_features)}")



def main():
      
    
    tdgm = TDGMEnhanced()
    data = tdgm.generate_data()
     
           
    # -----------------------------------------------------------
    # Save data to CSV 
    # -----------------------------------------------------------
    output_file = 'ESM_9.csv'
    print(f"Saving data to {output_file} ...", end=" ")
    data.to_csv(output_file, index=False)
    print("Completed!")
    # -----------------------------------------------------------

    # Print statistics 
    print(f"Total number of observations: {len(data):,}")
    # ...

if __name__ == "__main__":
    main()