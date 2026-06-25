import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

#  Chargement des modèles entraînés ---

model_classif = joblib.load('modele_sous_performance.pkl')
le = joblib.load('label_encoder.pkl')
df_final = pd.read_csv('dataset_final_agrege.csv')

# --- 2. Fonctions de traitement temporel ---
def get_category_timeseries(df, category_name):
    cat_df = df[df['product_category_name_english'] == category_name].copy()
    cat_df['date'] = pd.to_datetime(cat_df['purchase_year'].astype(str) + '-' + 
                                    cat_df['purchase_month'].astype(str) + '-01')
    cat_df = cat_df.sort_values('date').set_index('date').asfreq('MS')
    cat_df['ca'] = cat_df['ca'].fillna(0)
    return cat_df['ca']

def get_forecasted_ca(df, category):
    ts = get_category_timeseries(df, category)
    # SARIMA a besoin d'assez d'historique (au moins 12 mois)
    if len(ts) < 12:
        return ts.mean()
    try:
        model = SARIMAX(ts, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))
        res = model.fit(disp=False)
        return res.get_forecast(steps=1).predicted_mean.iloc[0]
    except:
        return ts.mean()

# --- 3. Diagnostic prédictif ---
def diagnostic_predictif(df, category, mois_futur, volume_estime, taux_annul_estime):
    # Vérification de l'existence de la catégorie
    if category not in le.classes_:
        return {"Erreur": f"La catégorie '{category}' n'est pas reconnue par le modèle."}
    
    # A. Prédiction du CA via SARIMA
    ca_predit = get_forecasted_ca(df, category)
    
    # B. Préparation des données pour le classifieur
    cat_encoded = le.transform([category])[0]
    
    # On suppose ca_growth = 0 par défaut pour une prédiction future
    input_data = pd.DataFrame([[mois_futur, ca_predit, taux_annul_estime, volume_estime, cat_encoded, 0]], 
                              columns=['purchase_month', 'ca', 'taux_annulation', 'volume', 'cat_encoded', 'ca_growth'])
    
    # C. Prédiction du niveau
    niveau = model_classif.predict(input_data)[0]
    
    return {
        "Categorie": category,
        "CA_predit": round(ca_predit, 2),
        "Niveau_risque": int(niveau),
        "Message": f"Action niveau {int(niveau)} requise" if niveau > 0 else "RAS : Performance attendue"
    }

# --- 4. Exécution ---
if __name__ == "__main__":
    resultat = diagnostic_predictif(df_final, 'health_beauty', 11, 20, 0.05)
    print(resultat)