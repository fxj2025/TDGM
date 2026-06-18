# =============================================================
# Article : Mechanism-Guided Synthetic Tax Data Generation:
#            A Computational Framework for Tax Evasion Detection
# Authors : Xiaojing Fan
# Affiliation: Business School, University of Shanghai for
#              Science and Technology, Shanghai, China
# Corresponding author: fanxiaojing@usst.edu.cn
# =============================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def make_xlabel(v, x_min, x_max):
    if v == x_min:
        return f'{v}%\n(Rare)'
    elif v == x_max:
        return f'{v}%\n(High)'
    else:
        return f'{v}%'


def plot_figure5(csv_filename='ESM_11.csv'):
    # ===========================================
    # 1. Load Data from CSV
    # ===========================================
    if os.path.exists(csv_filename):
        df_raw = pd.read_csv(csv_filename)
        data = {
            'Risk Prevalence (%)': (df_raw['onset'] * 100).round().astype(int).tolist(),
            'XGBoost (Non-Linear)': df_raw['xgb_auc'].tolist(),
            'Logistic Regression (Linear)': df_raw['lr_auc'].tolist(),
            'XGB_std': df_raw['XGB_std'].tolist() if 'XGB_std' in df_raw.columns else [0]*len(df_raw)
        }
        print(f"✓ Loaded data from {csv_filename}")
    else:
        raise FileNotFoundError(
            f"{csv_filename} not found. "
            "Please run ESM_4.py first to generate this file."
        )

    df = pd.DataFrame(data)

    # ===========================================
    # 2. Plot Configuration (Academic Style)
    # ===========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(10, 6))

    # ===========================================
    # 3. Plot Lines
    # ===========================================
    # XGBoost (Red, solid line with circles)
    ax.plot(df['Risk Prevalence (%)'], df['XGBoost (Non-Linear)'],
            marker='o', markersize=9, linewidth=2.5,
            color='#d62728', markerfacecolor='white', markeredgewidth=2,
            label='XGBoost (Non-Linear)', zorder=3)

    xgb_arr = np.array(df['XGBoost (Non-Linear)'])
    std_arr  = np.array(df['XGB_std'])
    x_arr    = np.array(df['Risk Prevalence (%)'])
    ax.fill_between(x_arr, xgb_arr - std_arr, xgb_arr + std_arr,
                    color='#d62728', alpha=0.12, zorder=2, label='XGBoost ±1 std')

    # Logistic Regression (Gray, dashed line with squares)
    ax.plot(df['Risk Prevalence (%)'], df['Logistic Regression (Linear)'],
            marker='s', markersize=8, linewidth=2,
            color='#7f7f7f', linestyle='--', markerfacecolor='white', markeredgewidth=2,
            label='Logistic Regression (Linear)', zorder=3)

    # ===========================================
    # 4. Add Annotations
    # ===========================================
    # Annotation 1: Extreme Imbalance Resilience at 1%
    auc_at_1pct = df[df['Risk Prevalence (%)'] == 1]['XGBoost (Non-Linear)'].values[0]
    ax.annotate(f'Robust under Extreme Imbalance\n(AUC = {auc_at_1pct:.4f} at 1% prevalence)',
                xy=(1, auc_at_1pct), xytext=(1.5, 0.995),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5,
                               connectionstyle="arc3,rad=-0.2"),
                fontsize=10, fontweight='bold', color='#d62728',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d62728', alpha=0.9))

    # Annotation 2: Statistical Variance at 3%
    auc_at_3pct = df[df['Risk Prevalence (%)'] == 3]['XGBoost (Non-Linear)'].values[0]
    ax.annotate('Sampling Variance\n(Small positive sample)',
                xy=(3, auc_at_3pct), xytext=(3, 0.93),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, linestyle=':'),
                fontsize=9, color='gray', ha='center')

    # Annotation 3: Environmental Invariance Platform
    xgb_min_auc = df['XGBoost (Non-Linear)'].min()
    mid_idx = len(df) // 2
    annot_x = df['Risk Prevalence (%)'].iloc[mid_idx]
    annot_y = df['XGBoost (Non-Linear)'].iloc[mid_idx]
    text_x = df['Risk Prevalence (%)'].iloc[mid_idx - 1]
    text_y = annot_y - 0.03
    ax.annotate(f'Environmental Invariance\n(Stable AUC > {xgb_min_auc:.3f})',
                xy=(annot_x, annot_y), xytext=(text_x+1.5, text_y-0.015),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                               connectionstyle="arc3,rad=0.2"),
                fontsize=10, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

    # ===========================================
    # 5. Add Performance Gap Indicator
    # ===========================================
    gap_x = 5
    xgb_y = df[df['Risk Prevalence (%)'] == gap_x]['XGBoost (Non-Linear)'].values[0]
    lr_y  = df[df['Risk Prevalence (%)'] == gap_x]['Logistic Regression (Linear)'].values[0]

    ax.annotate('', xy=(gap_x, xgb_y), xytext=(gap_x, lr_y),
                arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=2))
    ax.text(gap_x+0.2, (xgb_y + lr_y) / 2 - 0.02,
            f'Gap: {(xgb_y - lr_y)*100:.1f}pp\n(at 5% prevalence)',
            fontsize=9, fontweight='bold', color='#2ca02c', va='center')

    # ===========================================
    # 6. Axes Configuration
    # ===========================================
    ax.set_xlabel('Risk Prevalence (%)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Model Performance (AUC-ROC)', fontsize=12, fontweight='medium')

    x_vals = df['Risk Prevalence (%)'].tolist()
    x_min, x_max = x_vals[0], x_vals[-1]
    ax.set_xticks(x_vals)
    ax.set_xticklabels([make_xlabel(v, x_min, x_max) for v in x_vals], fontsize=10)

    ax.set_ylim(0.84, 1.01)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])

    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ===========================================
    # 7. Legend
    # ===========================================
    ax.legend(loc='lower right', frameon=True, fontsize=11,
              fancybox=True, shadow=False, edgecolor='gray')

    # ===========================================
    # 8. Add Reference Line at AUC = 0.97
    # ===========================================
    ax.axhline(y=0.97, color='#d62728', linestyle=':', alpha=0.5, linewidth=1.5, zorder=1)
    ax.text(x_max - 1, 0.9725, 'AUC = 0.97', fontsize=8, color='#d62728', va='center', ha='right')

    # ===========================================
    # 9. Save Figure
    # ===========================================
    plt.tight_layout()

    output_png = 'Figure5_Environmental_Invariance.png'
    output_pdf = 'Figure5_Environmental_Invariance.pdf'

    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')

    print(f"✓ Figure saved: {output_png}")
    print(f"✓ Figure saved: {output_pdf}")

    # ===========================================
    # 10. Print Summary Statistics
    # ===========================================
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"XGBoost AUC Range: {df['XGBoost (Non-Linear)'].min():.4f} - {df['XGBoost (Non-Linear)'].max():.4f}")
    print(f"XGBoost AUC Mean:  {df['XGBoost (Non-Linear)'].mean():.4f}")
    print(f"XGBoost AUC Std:   {df['XGBoost (Non-Linear)'].std():.4f}")
    print()
    print(f"LR AUC Range: {df['Logistic Regression (Linear)'].min():.4f} - {df['Logistic Regression (Linear)'].max():.4f}")
    print(f"LR AUC Mean:  {df['Logistic Regression (Linear)'].mean():.4f}")
    print(f"LR AUC Std:   {df['Logistic Regression (Linear)'].std():.4f}")
    print()
    print(f"Average Performance Gap: {(df['XGBoost (Non-Linear)'].mean() - df['Logistic Regression (Linear)'].mean())*100:.2f} percentage points")
    print("="*60)

    plt.close()


if __name__ == "__main__":
    plot_figure5()
