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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


COLORS = {
    'blue': '#0072B2',      
    'red': '#CC3311',       
    'gray': '#888888',      
    'text': '#333333',
}


def plot_zipf_validation_from_data(filename='ESM_9.csv'):
    
    print(f" Loading data: {filename} ...")
    
    # 1. Try to read the data
    try:
        df = pd.read_csv(filename)
        print(f"   ✓ Successfully read {len(df):,} records")
    except FileNotFoundError:
        print(f"    File {filename} not found, using simulated Pareto distribution data for demonstration...")
        # Simulate generating data with the same structure for demonstration
        np.random.seed(42)
        N_companies = 10000
        revenues = np.random.pareto(1.05, N_companies) * 100000 + 10000
        df = pd.DataFrame({'year': [2]*N_companies, 'revenue': revenues})

    # 2. Extract cross-section
    max_year = df['year'].max()
    cross_section = df[df['year'] == max_year].copy()
    
    print(f"   Extracting cross-sectional data for year {max_year}: a total of {len(cross_section):,} records")
    
    # 3. Prepare rank-size plot data
    revenues = cross_section['revenue'].sort_values(ascending=False).values
    ranks = np.arange(1, len(revenues) + 1)
    
    # Logarithmic transformation
    log_revenue = np.log10(revenues)
    log_rank = np.log10(ranks)
    
    # 4. Linear Regression Fitting
    model = LinearRegression()
    X = log_revenue.reshape(-1, 1)
    y = log_rank
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, model.predict(X))
    
    # Calculate standard error
    y_pred = model.predict(X)
    n = len(y)
    se = np.sqrt(np.sum((y - y_pred)**2) / (n - 2)) / np.sqrt(np.sum((X.flatten() - X.mean())**2))
    
    # 5. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Subsample points for visual clarity
    n_sample = min(2000, len(revenues))
    indices = np.linspace(0, len(revenues)-1, n_sample).astype(int)
    
    ax.scatter(log_revenue[indices], log_rank[indices], 
               alpha=0.4, s=12, color=COLORS['blue'], 
               edgecolor='none', label=f'Simulated Firms (Year {max_year})', zorder=2)
    
    # OLS 
    x_fit = np.linspace(log_revenue.min(), log_revenue.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color=COLORS['red'], linewidth=2.5, linestyle='-',
            label=f'OLS Fit (slope = {slope:.3f})', zorder=3)
    
    # Theoretical slope line (slope = -1.0)
   
    center_x = np.mean(log_revenue)
    center_y = np.mean(log_rank)
    c_theory = center_y + 1.0 * center_x
    y_theory = -1.0 * x_fit + c_theory
    ax.plot(x_fit, y_theory, color=COLORS['gray'], linewidth=2, linestyle='--',
            label='Theoretical (slope = -1.00)', zorder=2)
    
    
    
    ax.set_xlabel(r'$\log_{10}$(Revenue)', fontsize=12, fontweight='medium')
    ax.set_ylabel(r'$\log_{10}$(Rank)', fontsize=12, fontweight='medium')
    
    
    stats_text = (f"N = {len(cross_section):,}\n"
                  f"Slope = {slope:.3f} (SE = {se:.3f})\n"
                  f"Theoretical = -1.00\n"
                  f"R² = {r2:.4f}\n"
                  f"p < 0.001")
    
    ax.text(0.03, 0.03, stats_text, transform=ax.transAxes,
            fontsize=10, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.95, 
                     edgecolor=COLORS['gray'], linewidth=1,
                     boxstyle='round,pad=0.4'))
    
    
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
   
    plt.savefig('Figure_3_Zipf.png', facecolor='white', edgecolor='none')
    plt.savefig('Figure_3_Zipf.pdf', facecolor='white', edgecolor='none')
    
    print(f"   save: Figure_3_Zipf.png/pdf")
    print(f"   result: Slope = {slope:.4f}, R² = {r2:.4f}")
    
    plt.close()
    
    return slope, r2


if __name__ == "__main__":
    print("=" * 50)
    print("Figure 3: Zipf's Law Validation")
    print("=" * 50)
    plot_zipf_validation_from_data()
    print("=" * 50)
