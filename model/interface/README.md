# Interface API — Monitoring & Prédiction

API Flask qui expose en REST les deux briques de votre projet :

- **monitoring** : données du dashboard, à brancher dans Power BI.
- **prediction** : prévision du CA du mois suivant et du niveau de risque.

## 1. Arborescence

```
interface/
├── app.py                       # point d'entrée Flask
├── monitoring.py                # endpoints dashboard
├── prediction.py                # endpoints prédiction
├── requirements.txt
├── dataset_final_agrege.csv     # ⚠️ à copier ici (votre dataset)
├── modele_sous_performance.pkl  # ⚠️ à copier ici (modèle entraîné)
└── label_encoder.pkl            # ⚠️ à copier ici (encodeur entraîné)
```

Les 3 fichiers marqués ⚠️ sont ceux que produit/utilise déjà votre code d'entraînement
(`dump`/`read_csv`). Placez-les simplement dans le même dossier que `app.py`.

## 2. Installation

```bash
pip install -r requirements.txt
```

## 3. Lancement

```bash
python app.py
```

L'API démarre sur `http://localhost:5000`.

## 4. Endpoints — Monitoring (dashboard)

| Méthode | URL | Description |
|---|---|---|
| GET | `/monitoring/health` | Vérifie que le service répond |
| GET | `/monitoring/kpis` | KPIs globaux (CA total, taux d'annulation moyen, volume, répartition des niveaux) |
| GET | `/monitoring/categories?year=2018&month=11` | Détail ligne par ligne (catégorie/année/mois/CA/taux/volume/niveau) — **endpoint principal pour Power BI** |
| GET | `/monitoring/alerts` | Lignes en sous-performance (niveau > 0), triées par criticité |

### Connexion depuis Power BI

1. Power BI Desktop → **Obtenir les données** → **Web**.
2. Coller l'URL, par exemple : `http://localhost:5000/monitoring/categories`
3. Power BI affiche le JSON sous forme de table : cliquez **Vers Table**, puis dépliez la colonne.
4. Pour rafraîchir automatiquement, configurez une passerelle de données (Gateway) si l'API
   tourne sur un serveur on-premise, ou publiez l'API sur un serveur accessible (Azure, etc.)
   si vous voulez un rafraîchissement planifié dans le Service Power BI.

## 5. Endpoints — Prediction

| Méthode | URL | Description |
|---|---|---|
| GET | `/prediction/categories` | Liste des catégories reconnues par le modèle |
| GET | `/prediction/forecast?category=health_beauty&mois_futur=11&volume_estime=20&taux_annul_estime=0.05` | Prédiction (paramètres en query string) |
| POST | `/prediction/forecast` | Prédiction (paramètres en JSON dans le corps) |

Exemple `POST` :

```bash
curl -X POST http://localhost:5000/prediction/forecast \
  -H "Content-Type: application/json" \
  -d '{"category": "health_beauty", "mois_futur": 11, "volume_estime": 20, "taux_annul_estime": 0.05}'
```

Réponse :

```json
{
  "categorie": "health_beauty",
  "ca_predit": 18234.56,
  "niveau_risque": 1,
  "message": "Action niveau 1 requise"
}
```

## 6. Notes

- Le modèle et l'encodeur sont chargés **une seule fois** au démarrage du serveur (pas à
  chaque appel), pour éviter de recharger un `.pkl` à chaque requête.
- Le CSV du monitoring est rechargé à chaque appel (`load_data()`), pour toujours servir
  les données les plus récentes si le fichier est régénéré périodiquement.
- En production, désactivez `debug=True` dans `app.py` et utilisez un serveur WSGI
  (gunicorn, waitress...) plutôt que le serveur de développement Flask.
