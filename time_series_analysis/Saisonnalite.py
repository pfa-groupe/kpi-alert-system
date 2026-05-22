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
    .agg(ca=('total_price', 'sum'), nb_orders=('order_id', 'count'))
    .reset_index()
    .sort_values(['purchase_year', 'purchase_month'])
    .reset_index(drop=True)
)

# Retirer les mois partiels
monthly = monthly.iloc[1:-1].reset_index(drop=True)

# ── 3. Saisonnalité : CA moyen par mois calendaire ────────────
# On regroupe par numéro de mois (1=Jan, 2=Fév...) toutes années confondues
# et on calcule la moyenne du CA pour ce mois
seasonality = monthly.groupby('purchase_month').agg(
    ca_moyen=('ca', 'mean'),
    nb_moyen=('nb_orders', 'mean')
).reset_index()

month_names = ['Jan','Fév','Mar','Avr','Mai','Jun',
               'Jul','Aoû','Sep','Oct','Nov','Déc']
seasonality['mois_label'] = seasonality['purchase_month'].apply(
    lambda m: month_names[m - 1]
)

# ── 4. Pivot : CA par mois ET par année ───────────────────────
# Permet de comparer les mêmes mois sur différentes années
pivot = monthly.pivot_table(values='ca', index='purchase_month',
                             columns='purchase_year', aggfunc='sum')
pivot.index = [month_names[m-1] for m in pivot.index]

# ── 5. Figure 3 graphiques ────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 16), facecolor='#0f1117')
fig.suptitle("Identification de la Saisonnalité des Ventes",
             fontsize=15, fontweight='bold', color='#e8eaf6', y=0.98)

# ── Graphique 1 : CA moyen par mois (barres) ──────────────────
ax1 = axes[0]
ax1.set_facecolor('#1a1d27')

ca_max = seasonality['ca_moyen'].max()
ca_min = seasonality['ca_moyen'].min()

# Pic en violet, creux en rouge, reste en bleu
colors1 = [
    '#c77dff' if v == ca_max else ('#e63946' if v == ca_min else '#6c8ebf')
    for v in seasonality['ca_moyen']
]
bars1 = ax1.bar(seasonality['mois_label'], seasonality['ca_moyen'],
                color=colors1, alpha=0.85, width=0.6)

# Valeur au dessus de chaque barre
for bar, val in zip(bars1, seasonality['ca_moyen']):
    ax1.text(bar.get_x() + bar.get_width()/2,
             val + 4000,
             f'{val/1e3:.0f}k', ha='center', fontsize=8.5, color='#e8eaf6')

ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/1e3:.0f}k'))
ax1.set_title("CA Moyen par Mois Calendaire (toutes années)",
              fontsize=11, color='#e8eaf6', pad=8)
ax1.set_ylabel("CA moyen (R$ × 1000)", color='#c8cce0')
ax1.tick_params(colors='#8890b5')
ax1.grid(True, axis='y', color='#2e3147', linestyle='--', alpha=0.5)
for spine in ax1.spines.values():
    spine.set_edgecolor('#2e3147')

ax1.legend(handles=[
    mpatches.Patch(color='#c77dff', label=f"Pic : {seasonality.loc[seasonality['ca_moyen'].idxmax(),'mois_label']}"),
    mpatches.Patch(color='#e63946', label=f"Creux : {seasonality.loc[seasonality['ca_moyen'].idxmin(),'mois_label']}"),
    mpatches.Patch(color='#6c8ebf', label='Autres mois'),
], fontsize=9, framealpha=0.3, facecolor='#1a1d27', labelcolor='#c8cce0')

# ── Graphique 2 : Courbe de saisonnalité ──────────────────────
ax2 = axes[1]
ax2.set_facecolor('#1a1d27')

ax2.plot(seasonality['mois_label'], seasonality['ca_moyen'],
         color='#6c8ebf', lw=2.5, marker='o', ms=7, label='CA moyen')
ax2.fill_between(range(len(seasonality)), seasonality['ca_moyen'],
                 alpha=0.2, color='#6c8ebf')

# Ligne moyenne globale
mean_global = seasonality['ca_moyen'].mean()
ax2.axhline(mean_global, color='#82b366', lw=1.2, ls=':',
            label=f'Moyenne globale ({mean_global/1e3:.0f}k R$)')

# Annoter pic et creux
idx_max = seasonality['ca_moyen'].idxmax()
idx_min = seasonality['ca_moyen'].idxmin()
ax2.annotate(f"Pic\n{seasonality.loc[idx_max,'mois_label']}",
             xy=(idx_max, seasonality.loc[idx_max,'ca_moyen']),
             xytext=(0, 14), textcoords='offset points',
             ha='center', fontsize=8.5, color='#c77dff',
             arrowprops=dict(arrowstyle='->', color='#c77dff', lw=1.2))
ax2.annotate(f"Creux\n{seasonality.loc[idx_min,'mois_label']}",
             xy=(idx_min, seasonality.loc[idx_min,'ca_moyen']),
             xytext=(0, -30), textcoords='offset points',
             ha='center', fontsize=8.5, color='#e63946',
             arrowprops=dict(arrowstyle='->', color='#e63946', lw=1.2))

ax2.set_xticks(range(len(seasonality)))
ax2.set_xticklabels(seasonality['mois_label'], fontsize=9, color='#8890b5')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/1e3:.0f}k'))
ax2.set_title("Courbe de Saisonnalité — Pic et Creux annuels",
              fontsize=11, color='#e8eaf6', pad=8)
ax2.set_ylabel("CA moyen (R$ × 1000)", color='#c8cce0')
ax2.tick_params(colors='#8890b5')
ax2.grid(True, color='#2e3147', linestyle='--', alpha=0.5)
for spine in ax2.spines.values():
    spine.set_edgecolor('#2e3147')
ax2.legend(fontsize=9, framealpha=0.3, facecolor='#1a1d27', labelcolor='#c8cce0')

# ── Graphique 3 : CA par mois pour chaque année ───────────────
ax3 = axes[2]
ax3.set_facecolor('#1a1d27')

year_colors = {2016: '#f4a261', 2017: '#6c8ebf', 2018: '#82b366'}
for year in pivot.columns:
    data = pivot[year].dropna()
    ax3.plot(data.index, data.values,
             color=year_colors.get(year, '#8890b5'),
             lw=2, marker='o', ms=5, label=str(year))

ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/1e3:.0f}k'))
ax3.set_title("CA par Mois — Comparaison Année par Année",
              fontsize=11, color='#e8eaf6', pad=8)
ax3.set_ylabel("CA (R$ × 1000)", color='#c8cce0')
ax3.set_xlabel("Mois", color='#c8cce0')
ax3.tick_params(colors='#8890b5')
ax3.grid(True, color='#2e3147', linestyle='--', alpha=0.5)
for spine in ax3.spines.values():
    spine.set_edgecolor('#2e3147')
ax3.legend(fontsize=9, framealpha=0.3, facecolor='#1a1d27', labelcolor='#c8cce0')

# ── 6. Sauvegarde ─────────────────────────────────────────────
plt.tight_layout()
plt.savefig('saisonnalite.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅ Graphique sauvegardé : saisonnalite.png")