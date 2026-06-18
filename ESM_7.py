# =============================================================
# Article : Mechanism-Guided Synthetic Tax Data Generation:
#            A Computational Framework for Tax Evasion Detection
# Authors : Xiaojing Fan
# Affiliation: Business School, University of Shanghai for
#              Science and Technology, Shanghai, China
# Corresponding author: fanxiaojing@usst.edu.cn
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Global Drawing Style Settings
# ==========================================
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

# Color palette
COLORS = {
    'blue': '#0072B2',
    'red': '#CC3311',
    'gray': '#888888',
}


def plot_signal_strength():
   
    print("Figure 4: Signal Strength Sensitivity...")
    
    # 1. Load data
    csv_file = 'ESM_10.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        print("   Please run ESM_4.py first to generate data")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"   ✓ Loaded {csv_file}")
    
    # 2. Sort by signal strength level
    order_map = {'Very Weak': 1, 'Weak (Base)': 2, 'Moderate': 3, 'Strong': 4, 'Very Strong': 5}
    df['order'] = df['effect_level'].map(order_map)
    df = df.sort_values('order').reset_index(drop=True)
    
    # 3. Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(df))
    
    # Draw line chart
    ax.plot(x_pos, df['xgb_auc'], 
            marker='o', markersize=10, linewidth=3,
            color=COLORS['blue'], markerfacecolor='white', 
            markeredgewidth=2.5, label='XGBoost AUC-ROC', zorder=3)
    
    # Y-axis settings - dynamic range adjustment
   
   
    y_min = max(0.94, df['xgb_auc'].min() - 0.01)
    y_max = min(1.003, df['xgb_auc'].max() + 0.013)
    ax.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='medium')
    ax.set_ylim(y_min, y_max)
    
    # X-axis settings
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['effect_level'], fontsize=11)
    ax.set_xlabel('Signal Strength Scenario', fontsize=12, fontweight='medium')
    
    # Add data labels above each point
    for i, (x, y) in enumerate(zip(x_pos, df['xgb_auc'])):
        ax.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=9,
                    fontweight='medium', color=COLORS['blue'])
    
    # Identify the performance breakthrough interval
    
    
    auc_diff = df['xgb_auc'].diff()
    if auc_diff.max() > 0.003:
        threshold_cross_idx = (df['xgb_auc'] > 0.975).idxmax()
        x_start = threshold_cross_idx - 1  
        x_end = threshold_cross_idx        
        x_mid = (x_start + x_end) / 2
        y_mid = (df['xgb_auc'].iloc[x_start] + df['xgb_auc'].iloc[x_end]) / 2

        
        ax.annotate('Performance\nBreakthrough',
                    xy=(x_mid, y_mid),
                    xytext=(x_mid + 1.2, y_mid - 0.008),
                    fontsize=9, color=COLORS['red'], fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5,
                                   connectionstyle='arc3,rad=0'),
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=COLORS['red'],
                             boxstyle='round,pad=0.3'))

       
        ax.plot([x_start, x_end],
                [df['xgb_auc'].iloc[x_start], df['xgb_auc'].iloc[x_end]],
                color=COLORS['red'], linewidth=4, alpha=0.3, zorder=2)
    
   
    ax.axhline(y=0.975, color=COLORS['gray'], linestyle=':', alpha=0.5, zorder=0)
    ax.text(len(df)-0.5, 0.9755, 'Grey Zone Threshold', fontsize=8, color=COLORS['gray'], ha='right')
    
    # Legend
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4. Save figures
    plt.tight_layout()
    plt.savefig('Figure4_SignalStrength.png', facecolor='white', edgecolor='none')
    plt.savefig('Figure4_SignalStrength.pdf', facecolor='white', edgecolor='none')
    
    print("   Saved: Figure4_SignalStrength.png/pdf")
    
    # 5. Print key statistics
    print("\n" + "="*50)
    print("Figure 4 Key Data Summary")
    print("="*50)
    for _, row in df.iterrows():
        print(f"  {row['effect_level']:<15}: AUC = {row['xgb_auc']:.4f}")
    
    
    strong_auc = df[df['effect_level'] == 'Strong']['xgb_auc'].values[0]
    print("-"*50)
    print(f"  'AUC increasing to {strong_auc:.3f}'")
    print("="*50)
    
    plt.close()
    return df


if __name__ == "__main__":
    print("="*50)
    plot_signal_strength()
