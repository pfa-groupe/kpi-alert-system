import pandas as pd

# 1. Charger les données
df_labeled = pd.read_csv(r"C:\Users\Pc\Documents\pfa\data\processed\final_dataset_labeled.csv")
df_products = pd.read_csv(r"C:\Users\Pc\Documents\pfa\olist_products_dataset.csv")
df_trans = pd.read_csv(r"C:\Users\Pc\Documents\pfa\product_category_name_translation.csv")

# 2. Préparation : Fusionner pour avoir le nom de la catégorie
df_prod_cat = pd.merge(df_products, df_trans, on='product_category_name', how='left')
df_full = pd.merge(df_labeled, df_prod_cat[['product_id', 'product_category_name_english']], on='product_id', how='left')

# 3. Agrégation par mois et catégorie
# On définit les règles pour conserver le niveau et les alertes
df_final = df_full.groupby(['purchase_year', 'purchase_month', 'product_category_name_english']).agg(
    ca=('price', 'sum'),
    volume=('product_id', 'count'),
    taux_annulation=('is_canceled', 'mean'),
    under_performance=('under_performance', 'max'),
    # On garde le niveau max (le plus critique)
    niveau=('niveau', 'max'),
    # On concatène les alertes uniques pour garder tout le contexte
    alerts_str=('alerts_str', lambda x: ' | '.join(x.dropna().unique()))
).reset_index()

df_final.to_csv('dataset_final_agrege.csv', index=False)
print("Agrégation terminée : Niveau et alertes inclus.")