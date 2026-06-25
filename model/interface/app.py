"""
Point d'entrée de l'API.
Assemble les deux modules : monitoring (dashboard / Power BI) et prediction (prévision du CA).

Lancement local :
    python app.py
L'API est alors disponible sur http://localhost:5000

Dans Power BI Desktop :
    Obtenir les données > Web > coller par exemple :
    http://localhost:5000/monitoring/categories
"""

from flask import Flask, jsonify
from flask_cors import CORS

from monitoring import monitoring_bp
from prediction import prediction_bp

app = Flask(__name__)
CORS(app) 

app.register_blueprint(monitoring_bp)
app.register_blueprint(prediction_bp)


@app.route("/")
def index():
    return jsonify(
        {
            "service": "API Performance des catégories produit",
            "modules": {
                "monitoring": {
                    "description": "Données du dashboard, consommées par Power BI",
                    "endpoints": [
                        "GET /monitoring/health",
                        "GET /monitoring/kpis",
                        "GET /monitoring/categories?year=&month=",
                        "GET /monitoring/alerts",
                    ],
                },
                "prediction": {
                    "description": "Prévision du CA du mois suivant et du niveau de risque",
                    "endpoints": [
                        "GET /prediction/categories",
                        "GET /prediction/forecast?category=&mois_futur=&volume_estime=&taux_annul_estime=",
                        "POST /prediction/forecast (body JSON)",
                    ],
                },
            },
        }
    )


if __name__ == "__main__":
    # use_reloader=False permet d'éviter les conflits avec le debugger de VS Code
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)