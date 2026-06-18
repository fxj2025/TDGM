# =============================================================
# Article : Mechanism-Guided Synthetic Tax Data Generation:
#            A Computational Framework for Tax Evasion Detection
# Authors : Xiaojing Fan
# Affiliation: Business School, University of Shanghai for
#              Science and Technology, Shanghai, China
# Corresponding author: fanxiaojing@usst.edu.cn
# =============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_figure2():
    fig, ax = plt.subplots(figsize=(14, 13.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-5.2, 12.5)
    ax.axis('off')

    # === Colors ===
    color_step = '#ebf5fb';  edge_step = '#3498db'
    color_acct = '#fdf2e9';  edge_acct = '#d4ac0d'
    color_risk = '#fae5d3';  edge_risk = '#e67e22'
    color_l1   = '#d5f5e3';  edge_l1   = '#27ae60'
    color_l2   = '#fadbd8';  edge_l2   = '#c0392b'

    def draw_box(x, y, w, h, color, edge, lw=2):
        shadow = patches.FancyBboxPatch(
            (x+0.04, y-0.04), w, h,
            boxstyle="round,pad=0.12", fc='#cccccc', ec='none', alpha=0.25)
        ax.add_patch(shadow)
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12", fc=color, ec=edge, linewidth=lw)
        ax.add_patch(box)

    def step_circle(x, y, num, color):
        c = plt.Circle((x, y), 0.24, fc=color, ec='white', lw=1.5, zorder=5)
        ax.add_patch(c)
        ax.text(x, y, str(num), ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=6)

    arrow_kw = dict(facecolor='#34495e', edgecolor='none',
                    width=2.2, headwidth=9, headlength=8)
    gap = 0.7  # vertical gap between box bottom and arrow start

    # ========================================================
    #  STEP 1: Economic Baseline (Section 4.1)
    #  — Initialization + AR(1) Temporal Evolution
    # ========================================================
    y1 = 10.3;  h1 = 1.8
    draw_box(1.0, y1, 8.0, h1, color_step, edge_step)
    step_circle(1.55, y1 + h1 - 0.32, 1, edge_step)
    ax.text(1.95, y1 + h1 - 0.32, "Economic Baseline  (Section 4.1)",
            ha='left', va='center', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.text(5.0, y1 + h1/2 - 0.1,
            "Initialization:  Revenue₀ ~ Pareto(α = 1.05);  industry cost ratio γₖ,  capital intensity φₖ\n"
            "Temporal evolution:  ln(Rₜ) = ρ·ln(Rₜ₋₁) + (1−ρ)·μᵢ + εₜ,   ρ = 0.85,  σ = 0.12\n"
            "Observation window T = 3 years,  N = 10,000 firms,  firm size: Small / Medium / Large",
            ha='center', va='center', fontsize=9, color='#34495e', linespacing=1.6)
    ax.annotate('', xy=(5, y1 - gap), xytext=(5, y1 - 0.15), arrowprops=arrow_kw)

    # ========================================================
    #  STEP 2: Accounting Identity Enforcement (Section 4.2)
    # ========================================================
    y2 = 7.8;  h2 = 1.7
    draw_box(1.0, y2, 8.0, h2, color_acct, edge_acct, lw=2.5)
    step_circle(1.55, y2 + h2 - 0.32, 2, edge_acct)
    ax.text(1.95, y2 + h2 - 0.32, "Accounting Identity Enforcement  (Section 4.2)",
            ha='left', va='center', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.text(5.0, y2 + h2/2 - 0.1,
            "Flow identity:   Profit = Revenue − Cost;    Tax = max(0, Profit × τ)\n"
            "Stock identity:  Equity = Asset − Liability    →   A − L − E ≡ 0\n"
            "Deterministic, causal propagation  →  100% accounting consistency",
            ha='center', va='center', fontsize=9, color='#34495e', linespacing=1.6)
    # Badge
    ax.text(8.75, y2 + h2/2, "Key\nDifference\nvs. GAN",
            ha='center', va='center', fontsize=7.5, fontweight='bold',
            color=edge_acct, style='italic',
            bbox=dict(facecolor='#fef9e7', edgecolor=edge_acct,
                      boxstyle='round,pad=0.3', linewidth=1.5, alpha=0.9))
    ax.annotate('', xy=(5, y2 - gap), xytext=(5, y2 - 0.15), arrowprops=arrow_kw)

    # ========================================================
    #  STEP 3: Risk Injection (Section 4.3)
    # ========================================================
    y3 = 5.3;  h3 = 1.7
    draw_box(1.0, y3, 8.0, h3, color_risk, edge_risk)
    step_circle(1.55, y3 + h3 - 0.32, 3, edge_risk)
    ax.text(1.95, y3 + h3 - 0.32,
            "Risk Injection — Markov Chain + 4 Strategies  (Section 4.3)",
            ha='left', va='center', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.text(5.0, y3 + h3/2 - 0.1,
            "Markov persistence:  p_onset = 0.03,  p_persist = 0.80  →  equilibrium ≈ 13.0%\n"
            "Strategy 1: Revenue Suppression  |  Strategy 2: Cost Inflation\n"
            "Strategy 3: Transfer Manipulation    |  Strategy 4: Shell Companies (Control Group)",
            ha='center', va='center', fontsize=9, color='#34495e', linespacing=1.6)
    ax.annotate('', xy=(5, y3 - gap), xytext=(5, y3 - 0.15), arrowprops=arrow_kw)

    # ========================================================
    #  STEP 4: Observable Signal Generation (Section 4.4)
    # ========================================================
    y4 = 2.8;  h4 = 1.7
    draw_box(1.0, y4, 8.0, h4, color_step, edge_step)
    step_circle(1.55, y4 + h4 - 0.32, 4, edge_step)
    ax.text(1.95, y4 + h4 - 0.32, "Observable Signal Generation  (Section 4.4)",
            ha='left', va='center', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.text(5.0, y4 + h4/2 - 0.1,
            "VAT sales = true revenue + 10% operational noise\n"
            "Invoice signals from overlapping distributions  →  confusion zone\n"
            "Physical verification: electricity perturbed with heavy-tailed noise",
            ha='center', va='center', fontsize=9, color='#34495e', linespacing=1.6)

    # === Branching arrows to Level 1 / Level 2 ===
    lev_y = -2.8;  lev_h = 4.0   # defined here so branch_y can reference them
    split_y = y4 - 0.75
    branch_y = lev_y + lev_h + 0.15   # stop above Level box top edge
    ax.plot([5, 5], [y4 - 0.15, split_y], color='#34495e', linewidth=2)
    ax.plot([2.6, 7.4], [split_y, split_y], color='#34495e', linewidth=2)
    ax.annotate('', xy=(2.6, branch_y), xytext=(2.6, split_y), arrowprops=arrow_kw)
    ax.annotate('', xy=(7.4, branch_y), xytext=(7.4, split_y), arrowprops=arrow_kw)

    # Section 4.5 label — placed between Step 4 box bottom and split_y
    ax.text(5.0, (y4 + split_y) / 2, "Feature Partitioning  (Section 4.5)",
            ha='center', va='center', fontsize=9, fontweight='bold', color='#7f8c8d')

    # ========================================================
    #  Level 1: Public Data (Table 3)
    # ========================================================
    draw_box(0.15, lev_y, 4.7, lev_h, color_l1, edge_l1, lw=2.5)
    ax.text(2.5, lev_y + lev_h - 0.28,
            "Level 1: Public Data (Privacy-Safe)",
            ha='center', va='top', fontsize=11.5, fontweight='bold', color='#1a5e2a')
    ax.text(2.5, lev_y + lev_h - 0.65,
            "Annual Reports & Balance Sheets",
            ha='center', va='top', fontsize=8, color='grey', style='italic')

    l1_items = [
        ("Fundamentals:", "revenue, cost, profit, tax"),
        ("Balance Sheet:", "asset, liability, equity"),
        ("Ratios:", "profit_margin, asset_turnover"),
        ("Derived:", "tax_burden"),
        ("Demographics:", "industry, age, size"),
    ]
    yy = lev_y + lev_h - 1.1
    for cat, feats in l1_items:
        ax.text(0.55, yy, f"▸ {cat}", fontsize=8.5, fontweight='bold', color='#2c3e50', va='center')
        ax.text(2.35, yy, feats, fontsize=8.5, color='#34495e', va='center')
        yy -= 0.47
    ax.text(2.5, lev_y + 0.22, "13 features  ·  Publicly accessible",
            ha='center', fontsize=8.5, color='#27ae60', fontweight='bold')

    # ========================================================
    #  Level 2: Audit Data (Table 3)
    # ========================================================
    draw_box(5.15, lev_y, 4.7, lev_h, color_l2, edge_l2, lw=2.5)
    ax.text(7.5, lev_y + lev_h - 0.28,
            "Level 2: Audit Data (Sensitive)",
            ha='center', va='top', fontsize=11.5, fontweight='bold', color='#922b21')
    ax.text(7.5, lev_y + lev_h - 0.65,
            "Tax Systems & Field Audit",
            ha='center', va='top', fontsize=8, color='grey', style='italic')

    l2_items = [
        ("Verification:", "income_gap_ratio, vat_sales"),
        ("Risk Signals:", "high_risk_invoice_ratio,\nrelated_party_revenue_ratio,\ntransfer_pricing_anomaly"),
        ("Auxiliary:", "electricity_consumption,\noperating_expenses,\nrd_expenses_ratio"),
    ]
    yy = lev_y + lev_h - 1.1
    for cat, feats in l2_items:
        lines = feats.count('\n') + 1
        ax.text(5.55, yy, f"▸ {cat}", fontsize=8.5, fontweight='bold', color='#2c3e50', va='top')
        ax.text(7.35, yy, feats, fontsize=8.5, color='#34495e', va='top', linespacing=1.4)
        yy -= 0.15 + lines * 0.38
    ax.text(7.5, lev_y + 0.22, "8 features  ·  Requires audit authority",
            ha='center', fontsize=8.5, color='#c0392b', fontweight='bold')

    # === Bottom note ===
    ax.text(5.0, -4.15,
            "Note: Level 1 / Level 2 distinction corresponds to Table 3 feature partitioning.\n"
            "Proposition 4 (Structural Shadow): Level 1 models retain 98.4% of Level 1+2 detection accuracy (AUC).",
            ha='center', fontsize=8.5, color='grey', style='italic',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='#bdc3c7',
                      boxstyle='round,pad=0.5'))

    # === Legend ===
    legend_elements = [
        patches.Patch(facecolor=color_step, edgecolor=edge_step, label='Simulation Steps (§4.1, §4.4)'),
        patches.Patch(facecolor=color_acct, edgecolor=edge_acct, label='Accounting Constraint (§4.2)'),
        patches.Patch(facecolor=color_risk, edgecolor=edge_risk, label='Risk Injection (§4.3)'),
        patches.Patch(facecolor=color_l1, edgecolor=edge_l1, label='Level 1 Output (§4.5)'),
        patches.Patch(facecolor=color_l2, edgecolor=edge_l2, label='Level 2 Output (§4.5)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              fontsize=9, framealpha=0.9, edgecolor='#bdc3c7')

    plt.tight_layout()
    plt.savefig('Figure2_TDGM_Framework.png', dpi=600, bbox_inches='tight')
    plt.savefig('Figure2_TDGM_Framework.pdf', bbox_inches='tight')
    print("Figure 2 generated.")

if __name__ == "__main__":
    plot_figure2()
