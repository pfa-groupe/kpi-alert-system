"""
monitoring.py
=============
Module de monitoring avec interface web et API pour Power BI.
Contient les routes pour le dashboard et les données de performance.
"""

from flask import Blueprint, render_template, jsonify, request
import pandas as pd
import os
from dotenv import load_dotenv
from quick_predict import LevelPredictor, load_data_from_db
import logging


load_dotenv()
logging.basicConfig(level=logging.INFO)

# Créer un Blueprint (pas une app Flask complète)
monitoring_bp = Blueprint("monitoring", __name__, url_prefix="/monitoring")

# Initialiser le prédicteur (chargé une seule fois)
predictor = LevelPredictor()

# =============================================
# ROUTES POUR LE DASHBOARD WEB
# =============================================

@monitoring_bp.route('/')
def index():
    """Page d'accueil du dashboard"""
    return render_template('dashboard.html')

@monitoring_bp.route('/api/data')
def get_data():
    """API pour récupérer les données avec prédictions"""
    try:
        df = load_data_from_db()
        
        if df.empty:
            return jsonify({'error': 'Aucune donnée trouvée'}), 404
        
        df_result = predictor.predict_with_details(df)
        data = df_result.to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'data': data,
            'statistics': {
                'total_rows': len(df_result),
                'categories': df_result['product_category_name_english'].nunique(),
                'niveau_distribution': df_result['niveau'].value_counts().sort_index().to_dict(),
                'under_performance_count': int(df_result[df_result['under_performance'] == True].shape[0])
            }
        })
        
    except Exception as e:
        logging.error(f"Erreur: {e}")
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/api/categories')
def get_categories():
    """API pour récupérer la liste des catégories"""
    try:
        df = load_data_from_db()
        categories = df['product_category_name_english'].unique().tolist()
        return jsonify({'categories': sorted(categories)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/api/filter')
def filter_data():
    """API pour filtrer les données"""
    try:
        category = request.args.get('category')
        year = request.args.get('year', type=int)
        month = request.args.get('month', type=int)
        
        df = load_data_from_db()
        
        if category:
            df = df[df['product_category_name_english'] == category]
        if year:
            df = df[df['purchase_year'] == year]
        if month:
            df = df[df['purchase_month'] == month]
        
        df_result = predictor.predict_with_details(df)
        
        return jsonify({'status': 'success', 'data': df_result.to_dict(orient='records')})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================
# ROUTES POUR POWER BI (sans le préfixe /monitoring)
# =============================================

@monitoring_bp.route('/health', methods=["GET"])
def health():
    """Vérifie que le service tourne"""
    return jsonify({
        "status": "ok",
        "model_loaded": predictor.is_loaded,
        "database": "connected"
    })

@monitoring_bp.route('/kpis', methods=["GET"])
def kpis():
    """KPIs globaux pour les cartes Power BI"""
    try:
        df = load_data_from_db()
        
        if df.empty:
            return jsonify({"error": "No data available"}), 404
        
        df_result = predictor.predict_with_details(df)
        
        return jsonify({
            "ca_total": round(float(df["ca"].sum()), 2),
            "ca_moyen": round(float(df["ca"].mean()), 2),
            "taux_annulation_moyen": round(float(df["taux_annulation"].mean()), 4),
            "volume_total": int(df["volume"].sum()),
            "nb_categories": int(df["product_category_name_english"].nunique()),
            "repartition_niveaux": {
                str(k): int(v) for k, v in df_result["niveau"].value_counts().sort_index().items()
            },
            "under_performance_count": int(df_result[df_result["under_performance"] == True].shape[0])
        })
        
    except Exception as e:
        logging.error(f"Erreur: {e}")
        return jsonify({"error": str(e)}), 500

@monitoring_bp.route('/categories', methods=["GET"])
def categories():
    """Table détaillée par catégorie/année/mois pour Power BI"""
    try:
        df = load_data_from_db()
        
        if df.empty:
            return jsonify({"error": "No data available"}), 404
        
        year = request.args.get("year", type=int)
        month = request.args.get("month", type=int)
        
        if year:
            df = df[df["purchase_year"] == year]
        if month:
            df = df[df["purchase_month"] == month]
        
        df_result = predictor.predict_with_details(df)
        
        cols = [
            "product_category_name_english",
            "purchase_year",
            "purchase_month",
            "ca",
            "taux_annulation",
            "volume",
            "niveau",
            "under_performance",
            "alerts_str"
        ]
        
        return jsonify(df_result[cols].to_dict(orient="records"))
        
    except Exception as e:
        logging.error(f"Erreur: {e}")
        return jsonify({"error": str(e)}), 500

@monitoring_bp.route('/alerts', methods=["GET"])
def alerts():
    """Liste des lignes en sous-performance (niveau > 0)"""
    try:
        df = load_data_from_db()
        
        if df.empty:
            return jsonify([])
        
        df_result = predictor.predict_with_details(df)
        alertes = df_result[df_result["niveau"] > 0]
        
        cols = [
            "product_category_name_english",
            "purchase_year",
            "purchase_month",
            "ca",
            "taux_annulation",
            "volume",
            "niveau",
            "alerts_str"
        ]
        alertes = alertes[cols].sort_values("niveau", ascending=False)
        
        return jsonify(alertes.to_dict(orient="records"))
        
    except Exception as e:
        logging.error(f"Erreur: {e}")
        return jsonify({"error": str(e)}), 500