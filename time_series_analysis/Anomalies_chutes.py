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

# Colonne label lisible
monthly['periode'] = pd.to_datetime(
    monthly[['purchase_year','purchase_month']].rename(
        columns={'purchase_year':'year','purchase_month':'month'}
    ).assign(day=1)
)
monthly['label'] = monthly['periode'].dt.strftime('%b %Y')

# ── 3. Détection des anomalies par Z-score ────────────────────
# Z-score = combien de fois le mois s'écarte de la moyenne (en nb d'écarts-types)
# Si |z| > 1.5 → le mois est anormal (trop haut ou trop bas)
mean_ca = monthly['ca'].mean()
std_ca  = monthly['ca'].std()

monthly['z_score']    = (monthly['ca'] - mean_ca) / std_ca
monthly['is_anomaly'] = monthly['z_score'].abs() > 1.5

# ── 4. Détection des chutes brutales (MoM) ────────────────────
# On calcule la variation % d'un mois à l'autre
# Si le CA chute de plus de 15% en un mois → chute brutale
monthly['ca_prev']  = monthly['ca'].shift(1)
monthly['mom_pct']  = (monthly['ca'] - monthly['ca_prev']) / monthly['ca_prev'] * 100
monthly['is_chute'] = monthly['mom_pct'] < -15

# ── 5. Bandes de contrôle ─────────────────────────────────────
upper = mean_ca + 1.5 * std_ca   # seuil haut
lower = mean_ca - 1.5 * std_ca   # seuil bas
x     = np.arange(len(monthly))

# ── 6. Figure 3 graphiques ────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 16), facecolor='#0f1117')
fig.suptitle("Détection des Chutes et Anomalies dans les Ventes",
             fontsize=15, fontweight='bold', color='#e8eaf6', y=0.98)

# ── Graphique 1 : Courbe CA + bandes de contrôle ─────────────
ax1 = axes[0]
ax1.set_facecolor('#1a1d27')

# Zone verte = zone normale
ax1.fill_between(x, lower, upper, alpha=0.12, color='#82b366', label='Zone normale (±1.5σ)')
ax1.axhline(mean_ca, color='#82b366', lw=1.2, ls='--', alpha=0.8,
            label=f'Moyenne ({mean_ca/1e3:.0f}k R$)')
ax1.axhline(upper, color='#f4a261', lw=1, ls=':', alpha=0.7,
            label=f'Seuil haut ({upper/1e3:.0f}k R$)')
ax1.axhline(lower, color='#e63946', lw=1, ls=':', alpha=0.7,
            label=f'Seuil bas ({lower/1e3:.0f}k R$)')

# Courbe CA
ax1.plot(x, monthly['ca'], color='#6c8ebf', lw=2.2, marker='o', ms=4, zorder=3)

# Points anomalies en rouge
anom = monthly[monthly['is_anomaly']]
ax1.scatter(anom.index, anom['ca'], color='#e63946', s=100, zorder=5,
            label='Anomalie détectée')
for _, row in anom.iterrows():
    ax1.annotate(f"  {row['label']}", (row.name, row['ca']),
                 fontsize=8, color='#e63946', va='center')

ax1.set_xticks(x)
ax1.set_xticklabels(monthly['label'], rotation=40, ha='right', fontsize=8, color='#8890b5')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/1e3:.0f}k'))
ax1.set_title("Courbe CA avec Bandes de Contrôle (Z-score ±1.5σ)",
              fontsize=11, color='#e8eaf6', pad=8)
ax1.set_ylabel("CA (R$ × 1000)", color='#c8cce0')
ax1.tick_params(colors='#8890b5')
ax1.grid(True, color='#2e3147', linestyle='--', alpha=0.5)
for spine in ax1.spines.values(): spine.set_edgecolor('#2e3147')
ax1.legend(fontsize=8.5, framealpha=0.3, facecolor='#1a1d27', labelcolor='#c8cce0')

# ── Graphique 2 : Z-score par mois ────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#1a1d27')

# Rouge si anomalie, bleu sinon
colors_z = ['#e63946' if abs(z) > 1.5 else '#6c8ebf' for z in monthly['z_score']]
bars2 = ax2.bar(x, monthly['z_score'], color=colors_z, alpha=0.85, width=0.6)

ax2.axhline( 1.5, color='#f4a261', lw=1.2, ls='--', label='Seuil +1.5σ')
ax2.axhline(-1.5, color='#f4a261', lw=1.2, ls='--', label='Seuil -1.5σ')
ax2.axhline(   0, color='#8890b5', lw=0.8)

# Valeur affichée sur les barres anomalies
for bar, val in zip(bars2, monthly['z_score']):
    if abs(val) > 1.5:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + (0.1 if val > 0 else -0.2),
                 f'{val:.1f}', ha='center', fontsize=8, color='#e63946')

ax2.set_xticks(x)
ax2.set_xticklabels(monthly['label'], rotation=40, ha='right', fontsize=8, color='#8890b5')
ax2.set_title("Z-score par Mois (anomalie si |z| > 1.5)",
              fontsize=11, color='#e8eaf6', pad=8)
ax2.set_ylabel("Z-score", color='#c8cce0')
ax2.tick_params(colors='#8890b5')
ax2.grid(True, axis='y', color='#2e3147', linestyle='--', alpha=0.5)
for spine in ax2.spines.values(): spine.set_edgecolor('#2e3147')
ax2.legend(fontsize=8.5, framealpha=0.3, facecolor='#1a1d27', labelcolor='#c8cce0')

# ── Graphique 3 : Chutes brutales MoM ────────────────────────
ax3 = axes[2]
ax3.set_facecolor('#1a1d27')

mom_data = monthly.dropna(subset=['mom_pct']).reset_index(drop=True)
colors3  = ['#e63946' if c else '#6c8ebf' for c in mom_data['is_chute']]
bars3 = ax3.bar(range(len(mom_data)), mom_data['mom_pct'],
                color=colors3, alpha=0.85, width=0.6)

# Valeur affichée sur les barres chutes
for bar, val, chute in zip(bars3, mom_data['mom_pct'], mom_data['is_chute']):
    if chute:
        ax3.text(bar.get_x() + bar.get_width()/2,
                 val - 4,
                 f'{val:.0f}%', ha='center', fontsize=8,
                 color='#e63946', fontweight='bold')

ax3.axhline(-15, color='#f4a261', lw=1.5, ls='--', label='Seuil chute (-15%)')
ax3.axhline(  0, color='#8890b5', lw=0.8)
ax3.set_xticks(range(len(mom_data)))
ax3.set_xticklabels(mom_data['label'], rotation=40, ha='right', fontsize=8, color='#8890b5')
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v:+.0f}%'))
ax3.set_title("Chutes Brutales — Variation MoM < -15%",
              fontsize=11, color='#e8eaf6', pad=8)
ax3.set_ylabel("Variation (%)", color='#c8cce0')
ax3.tick_params(colors='#8890b5')
ax3.grid(True, axis='y', color='#2e3147', linestyle='--', alpha=0.5)
for spine in ax3.spines.values(): spine.set_edgecolor('#2e3147')

ax3.legend(handles=[
    mpatches.Patch(color='#e63946', label='Chute détectée (< -15%)'),
    mpatches.Patch(color='#6c8ebf', label='Normal'),
    plt.Line2D([0],[0], color='#f4a261', lw=1.5, ls='--', label='Seuil -15%'),
], fontsize=8.5, framealpha=0.3, facecolor='#1a1d27', labelcolor='#c8cce0')

# ── 7. Sauvegarde ─────────────────────────────────────────────
plt.tight_layout()
plt.savefig('anomalies_chutes.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅ Graphique sauvegardé : anomalies_chutes.png")

# ── 8. Résumé console ─────────────────────────────────────────
print("\n=== ANOMALIES DÉTECTÉES (Z-score > 1.5) ===")
print(monthly[monthly['is_anomaly']][['label','ca','z_score']].to_string(index=False))
print("\n=== CHUTES BRUTALES (MoM < -15%) ===")
print(monthly[monthly['is_chute']][['label','ca','mom_pct']].to_string(index=False))