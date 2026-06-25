import os
import pandas as pd
import joblib
from flask import Blueprint, jsonify, request
from statsmodels.tsa.statespace.sarimax import SARIMAX

prediction_bp = Blueprint("prediction", __name__, url_prefix="/prediction")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "modele_sous_performance.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
DATA_PATH    = os.path.join(BASE_DIR, "dataset_final_agrege.csv")

# Chargement des ressources
model_classif = joblib.load(MODEL_PATH)
le            = joblib.load(ENCODER_PATH)
df_final      = pd.read_csv(DATA_PATH)

def get_category_timeseries(df, category_name):
    cat_df = df[df["product_category_name_english"] == category_name].copy()
    cat_df["date"] = pd.to_datetime(
        cat_df["purchase_year"].astype(str) + "-" + cat_df["purchase_month"].astype(str) + "-01"
    )
    cat_df = cat_df.sort_values("date").set_index("date").asfreq("MS")
    cat_df["ca"] = cat_df["ca"].fillna(0)
    return cat_df["ca"]

def get_forecasted_ca(df, category):
    ts = get_category_timeseries(df, category)
    if len(ts) < 12:
        return float(ts.mean())
    try:
        model = SARIMAX(ts, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))
        res   = model.fit(disp=False)
        return float(res.get_forecast(steps=1).predicted_mean.iloc[0])
    except Exception:
        return float(ts.mean())

def diagnostic_predictif(df, category, mois_futur):
    if category not in le.classes_:
        return {"erreur": f"La catégorie '{category}' n'est pas reconnue par le modèle."}

    cat_df = df[df["product_category_name_english"] == category]

    # 1. Calcul du CA prévu
    ts = get_category_timeseries(df, category)
    ca_dernier_reel = float(ts.iloc[-1]) if not ts.empty else 0
    ca_predit = get_forecasted_ca(df, category)

    # 2. Calcul du CA_GROWTH dynamique
    if ca_dernier_reel > 0:
        ca_growth_estime = (ca_predit - ca_dernier_reel) / ca_dernier_reel
    else:
        ca_growth_estime = 0.0

    # 3. Estimation des autres métriques
    meme_mois = cat_df[cat_df["purchase_month"] == mois_futur]
    if not meme_mois.empty:
        volume_estime     = float(meme_mois["volume"].mean())
        taux_annul_estime = float(meme_mois["taux_annulation"].mean())
    else:
        volume_estime     = float(cat_df["volume"].mean()) if not cat_df.empty else 0.0
        taux_annul_estime = float(cat_df["taux_annulation"].mean()) if not cat_df.empty else 0.0

    cat_encoded = int(le.transform([category])[0])

    # 4. Construction des features
    input_data = pd.DataFrame(
        [[mois_futur, ca_predit, taux_annul_estime, volume_estime, cat_encoded, ca_growth_estime]],
        columns=["purchase_month", "ca", "taux_annulation", "volume", "cat_encoded", "ca_growth"],
    )

    # 5. Prédiction + probabilités du modèle  ← NOUVEAU
    niveau  = model_classif.predict(input_data)[0]
    proba   = model_classif.predict_proba(input_data)[0]
    classes = model_classif.classes_
    idx     = list(classes).index(niveau)

    confiance_pourcentage = round(float(proba[idx]) * 100, 1)
    proba_par_niveau = {int(c): round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    return {
        "categorie":            category,
        "ca_predit":            round(ca_predit, 2),
        "ca_growth_estime":     round(ca_growth_estime, 4),
        "volume_estime":        round(volume_estime, 1),
        "taux_annul_estime":    round(taux_annul_estime, 4),
        "niveau_risque":        int(niveau),
        "confiance_pourcentage": confiance_pourcentage,   # ← NOUVEAU
        "proba_par_niveau":      proba_par_niveau,        # ← NOUVEAU (bonus, optionnel)
        "message": (
            f"Niveau {int(niveau)} avec {confiance_pourcentage:.0f}% de confiance — action requise"
            if niveau > 0
            else f"RAS : performance attendue ({confiance_pourcentage:.0f}% de confiance)"
        ),
    }

def _parse_and_predict(category, mois_futur):
    if not category or mois_futur is None:
        return {"erreur": "Paramètres manquants : category, mois_futur"}, 400
    resultat = diagnostic_predictif(df_final, category, mois_futur)
    status   = 400 if "erreur" in resultat else 200
    return resultat, status

@prediction_bp.route("/categories", methods=["GET"])
def list_categories():
    return jsonify({"categories": sorted(le.classes_.tolist())})

@prediction_bp.route("/forecast", methods=["GET"])
def forecast_get():
    category   = request.args.get("category")
    mois_futur = request.args.get("mois_futur", type=int)
    resultat, status = _parse_and_predict(category, mois_futur)
    return jsonify(resultat), status

@prediction_bp.route("/forecast", methods=["POST"])
def forecast_post():
    payload      = request.get_json(silent=True) or {}
    resultat, status = _parse_and_predict(
        payload.get("category"),
        payload.get("mois_futur"),
    )
    return jsonify(resultat), status