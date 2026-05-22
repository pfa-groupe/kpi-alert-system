import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# ── 3. Colonne label lisible (ex: "Nov 2017") ─────────────────
monthly['periode'] = pd.to_datetime(
    monthly[['purchase_year', 'purchase_month']].rename(
        columns={'purchase_year': 'year', 'purchase_month': 'month'}
    ).assign(day=1)
)
monthly['label'] = monthly['periode'].dt.strftime('%b %Y')

# ── 4. Calcul de la tendance linéaire ─────────────────────────
x = np.arange(len(monthly))
z = np.polyfit(x, monthly['ca'], 1)
trend = np.poly1d(z)(x)
trend_pct = (trend[-1] - trend[0]) / trend[0] * 100

# ── 5. Moyenne globale ────────────────────────────────────────
mean_ca = monthly['ca'].mean()

# ── 6. Graphique ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')

# Zone bleue sous la courbe
ax.fill_between(x, monthly['ca'], alpha=0.15, color='#6c8ebf')

# Courbe CA mensuel
ax.plot(x, monthly['ca'], color='#6c8ebf', lw=2.5,
        marker='o', ms=5, label='CA mensuel')

# Droite de tendance
ax.plot(x, trend, color='#f4a261', lw=2, ls='--',
        label=f'Tendance ({trend_pct:+.1f}%)')

# Ligne moyenne
ax.axhline(mean_ca, color='#82b366', lw=1.2, ls=':',
           label=f'Moyenne ({mean_ca:,.0f} R$)')

# ── 7. Annotations min / max ──────────────────────────────────
idx_max = monthly['ca'].idxmax()
idx_min = monthly['ca'].idxmin()

ax.annotate(
    f"Pic\n{monthly.loc[idx_max, 'label']}\n{monthly.loc[idx_max, 'ca']/1e3:.0f}k R$",
    xy=(idx_max, monthly.loc[idx_max, 'ca']),
    xytext=(idx_max - 3, monthly.loc[idx_max, 'ca'] - 120000),
    fontsize=8.5, color='#82b366',
    arrowprops=dict(arrowstyle='->', color='#82b366', lw=1.2)
)
ax.annotate(
    f"Creux\n{monthly.loc[idx_min, 'label']}\n{monthly.loc[idx_min, 'ca']/1e3:.1f}k R$",
    xy=(idx_min, monthly.loc[idx_min, 'ca']),
    xytext=(idx_min + 1, monthly.loc[idx_min, 'ca'] + 100000),
    fontsize=8.5, color='#e63946',
    arrowprops=dict(arrowstyle='->', color='#e63946', lw=1.2)
)

# ── 8. Mise en forme des axes ─────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(monthly['label'], rotation=40, ha='right',
                   fontsize=8.5, color='#8890b5')
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v/1e3:.0f}k')
)
ax.tick_params(colors='#8890b5')
for spine in ax.spines.values():
    spine.set_edgecolor('#2e3147')

# ── 9. Titres et légende ──────────────────────────────────────
ax.set_title("Évolution des Ventes — CA Mensuel (Oct 2016 → Jul 2018)",
             fontsize=14, fontweight='bold', color='#e8eaf6', pad=14)
ax.set_ylabel("Chiffre d'Affaires (R$ × 1000)", color='#c8cce0')
ax.set_xlabel("Période", color='#c8cce0')
ax.legend(fontsize=9, framealpha=0.3, facecolor='#1a1d27', labelcolor='#c8cce0')
ax.grid(True, color='#2e3147', linestyle='--', alpha=0.5)

# ── 10. Sauvegarde ────────────────────────────────────────────
plt.tight_layout()
plt.savefig('evolution_ventes.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅ Graphique sauvegardé : evolution_ventes.png")