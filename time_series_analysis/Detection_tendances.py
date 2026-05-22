import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# ── 1. Chargement ─────────────────────────────────────────────
df = pd.read_csv('final_dataset_labeled.csv')

# ── 2. Agrégation mensuelle ───────────────────────────────────
monthly = (
    df.groupby(['purchase_year', 'purchase_month'])
    .agg(ca=('total_price', 'sum'))
    .reset_index()
    .sort_values(['purchase_year', 'purchase_month'])
    .reset_index(drop=True)
)

# Retirer les mois partiels (début et fin)
monthly = monthly.iloc[1:-1].reset_index(drop=True)

# Colonne label lisible (ex: "Nov 2017")
monthly['periode'] = pd.to_datetime(
    monthly[['purchase_year', 'purchase_month']].rename(
        columns={'purchase_year': 'year', 'purchase_month': 'month'}
    ).assign(day=1)
)
monthly['label'] = monthly['periode'].dt.strftime('%b %Y')

# ── 3. Variation MoM (mois sur mois) ─────────────────────────
monthly['ca_prev']  = monthly['ca'].shift(1)        # CA du mois précédent
monthly['mom_diff'] = monthly['ca'] - monthly['ca_prev']  # différence brute
monthly['mom_pct']  = (monthly['mom_diff'] / monthly['ca_prev']) * 100  # en %
monthly['tendance'] = monthly['mom_pct'].apply(
    lambda v: 'hausse' if v > 0 else ('baisse' if v < 0 else 'stable')
)

# ── 4. CA moyen par année ─────────────────────────────────────
yearly = monthly.groupby('purchase_year')['ca'].mean().reset_index()
yearly.columns = ['annee', 'ca_moyen']

# ── 5. Tendance linéaire globale ──────────────────────────────
x = np.arange(len(monthly))
z = np.polyfit(x, monthly['ca'], 1)
trend = np.poly1d(z)(x)

# ── 6. Figure avec 3 graphiques ───────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 16), facecolor='#0f1117')
fig.suptitle("Détection des Tendances — Hausse / Baisse des Ventes",
             fontsize=15, fontweight='bold', color='#e8eaf6', y=0.98)

# ── Graphique 1 : Courbe CA + zones hausse/baisse ─────────────
ax1 = axes[0]
ax1.set_facecolor('#1a1d27')

for i in range(1, len(monthly)):
    color = '#82b366' if monthly.loc[i, 'tendance'] == 'hausse' else '#e63946'
    ax1.fill_between([i-1, i],
                     [monthly.loc[i-1, 'ca'], monthly.loc[i, 'ca']],
                     alpha=0.25, color=color)
    ax1.plot([i-1, i],
             [monthly.loc[i-1, 'ca'], monthly.loc[i, 'ca']],
             color=color, lw=2.5)

ax1.plot(x, trend, color='#f4a261', lw=2, ls='--', label='Tendance globale')
ax1.scatter(range(len(monthly)), monthly['ca'], color='white', s=40, zorder=5)
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly['label'], rotation=40, ha='right', fontsize=8, color='#8890b5')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1e3:.0f}k'))
ax1.set_title("Courbe CA avec zones Hausse (vert) / Baisse (rouge)", fontsize=11, color='#e8eaf6', pad=8)
ax1.set_ylabel("CA (R$ × 1000)", color='#c8cce0')
ax1.tick_params(colors='#8890b5')
ax1.grid(True, color='#2e3147', linestyle='--', alpha=0.5)
for spine in ax1.spines.values():
    spine.set_edgecolor('#2e3147')
patch_h = mpatches.Patch(color='#82b366', label='Hausse')
patch_b = mpatches.Patch(color='#e63946', label='Baisse')
ax1.legend(handles=[patch_h, patch_b,
           plt.Line2D([0], [0], color='#f4a261', lw=2, ls='--', label='Tendance')],
           fontsize=9, framealpha=0.3, facecolor='#1a1d27', labelcolor='#c8cce0')

# ── Graphique 2 : Variation MoM en barres ─────────────────────
ax2 = axes[1]
ax2.set_facecolor('#1a1d27')

mom_data = monthly.dropna(subset=['mom_pct']).reset_index(drop=True)
colors2  = ['#82b366' if v >= 0 else '#e63946' for v in mom_data['mom_pct']]
bars = ax2.bar(range(len(mom_data)), mom_data['mom_pct'],
               color=colors2, alpha=0.85, width=0.6)

# Valeur affichée au dessus de chaque barre
for i, (bar, val) in enumerate(zip(bars, mom_data['mom_pct'])):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             val + (3 if val >= 0 else -8),
             f'{val:+.0f}%', ha='center', fontsize=7.5,
             color='#82b366' if val >= 0 else '#e63946')

ax2.axhline(0, color='#8890b5', lw=0.8)
ax2.set_xticks(range(len(mom_data)))
ax2.set_xticklabels(mom_data['label'], rotation=40, ha='right', fontsize=8, color='#8890b5')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:+.0f}%'))
ax2.set_title("Variation Mois sur Mois (MoM %)", fontsize=11, color='#e8eaf6', pad=8)
ax2.set_ylabel("Variation (%)", color='#c8cce0')
ax2.tick_params(colors='#8890b5')
ax2.grid(True, axis='y', color='#2e3147', linestyle='--', alpha=0.5)
for spine in ax2.spines.values():
    spine.set_edgecolor('#2e3147')

# ── Graphique 3 : CA moyen par année ──────────────────────────
ax3 = axes[2]
ax3.set_facecolor('#1a1d27')

year_colors = ['#6c8ebf', '#82b366', '#f4a261']
bars3 = ax3.bar(yearly['annee'].astype(str), yearly['ca_moyen'],
                color=year_colors[:len(yearly)], alpha=0.85, width=0.5)

# Valeur au dessus de chaque barre
for bar, val in zip(bars3, yearly['ca_moyen']):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             val + 5000,
             f'{val/1e3:.0f}k R$', ha='center', fontsize=10,
             color='#e8eaf6', fontweight='bold')

# Flèches de comparaison entre années
for i in range(1, len(yearly)):
    diff = yearly.loc[i, 'ca_moyen'] - yearly.loc[i-1, 'ca_moyen']
    pct  = diff / yearly.loc[i-1, 'ca_moyen'] * 100
    ax3.annotate('', xy=(i, yearly.loc[i, 'ca_moyen']),
                 xytext=(i-1, yearly.loc[i-1, 'ca_moyen']),
                 arrowprops=dict(arrowstyle='->', color='#f4a261', lw=2))
    ax3.text(i - 0.5,
             max(yearly.loc[i, 'ca_moyen'], yearly.loc[i-1, 'ca_moyen']) + 30000,
             f'{pct:+.0f}%', ha='center', fontsize=10,
             color='#f4a261', fontweight='bold')

ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1e3:.0f}k'))
ax3.set_title("CA Moyen par Année — Comparaison 2016 / 2017 / 2018",
              fontsize=11, color='#e8eaf6', pad=8)
ax3.set_ylabel("CA moyen mensuel (R$ × 1000)", color='#c8cce0')
ax3.tick_params(colors='#8890b5')
ax3.grid(True, axis='y', color='#2e3147', linestyle='--', alpha=0.5)
for spine in ax3.spines.values():
    spine.set_edgecolor('#2e3147')

# ── 7. Sauvegarde ─────────────────────────────────────────────
plt.tight_layout()
plt.savefig('detection_tendances.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅ Graphique sauvegardé : detection_tendances.png")