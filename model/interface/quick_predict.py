"""
quick_predict.py
Lit les données directement depuis PostgreSQL et applique le modèle de prédiction.
Pas besoin de CSV !
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

#configuration

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'postgres'),  # Utilise postgres car la table y est
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modele_sous_performance.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

#classes de prediction
class LevelPredictor:
    """Prédicteur de niveau utilisant le modèle entraîné"""
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.features = ['purchase_month', 'ca', 'taux_annulation', 'volume', 'cat_encoded', 'ca_growth']
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Charge le modèle et l'encodeur"""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                logging.info(f" Modèle chargé: {MODEL_PATH}")
            else:
                logging.warning(f" Modèle non trouvé: {MODEL_PATH}")
                self.model = None
            
            if os.path.exists(ENCODER_PATH):
                self.encoder = joblib.load(ENCODER_PATH)
                logging.info(f"Encodeur chargé: {ENCODER_PATH}")
            else:
                logging.warning(f" Encodeur non trouvé: {ENCODER_PATH}")
                self.encoder = None
            
            self.is_loaded = (self.model is not None and self.encoder is not None)
            
        except Exception as e:
            logging.error(f" Erreur lors du chargement: {e}")
            self.model = None
            self.encoder = None
            self.is_loaded = False
    
    def prepare_features(self, df):
        """Prépare les features pour la prédiction"""
        df_pred = df.copy()
        
        # Encoder les catégories
        if self.encoder is not None:
            try:
                df_pred['cat_encoded'] = self.encoder.transform(df_pred['product_category_name_english'])
            except Exception as e:
                logging.warning(f"Erreur d'encodage: {e}, utilisation du fallback")
                df_pred['cat_encoded'] = df_pred['product_category_name_english'].astype('category').cat.codes
        else:
            df_pred['cat_encoded'] = df_pred['product_category_name_english'].astype('category').cat.codes
        
        # Calculer la croissance du CA
        df_pred = df_pred.sort_values(['product_category_name_english', 'purchase_year', 'purchase_month'])
        df_pred['ca_growth'] = df_pred.groupby('product_category_name_english')['ca'].pct_change().fillna(0)
        df_pred['ca_growth'] = df_pred['ca_growth'].replace([np.inf, -np.inf], 0).fillna(0)
        
        return df_pred[self.features]
    
    def predict(self, df):
        """Prédit les niveaux pour un DataFrame"""
        if not self.is_loaded:
            logging.warning("Modèle non chargé, utilisation des règles métier")
            return self.predict_by_rules(df)
        
        try:
            X = self.prepare_features(df)
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction: {e}")
            return self.predict_by_rules(df)
    
    def predict_by_rules(self, df):
        """Prédiction basée sur des règles métier (fallback)"""
        def calculate_niveau(row):
            score = 0
            
            # Taux d'annulation
            if row['taux_annulation'] > 0.5:
                score += 3
            elif row['taux_annulation'] > 0.3:
                score += 2
            elif row['taux_annulation'] > 0.15:
                score += 1
            
            # Volume
            if row['volume'] < 10:
                score += 2
            elif row['volume'] < 50:
                score += 1
            
            # CA
            if row['ca'] < 100:
                score += 1
            
            # Combinaison
            if row['taux_annulation'] > 0.3 and row['volume'] < 30:
                score += 1
            
            return min(score, 4)
        
        return df.apply(calculate_niveau, axis=1)
    
    def predict_with_details(self, df):
        """
        Prédit le niveau, under_performance et génère les alertes
        """
        # Prédire le niveau
        df['niveau'] = self.predict(df)
        
        # Déterminer under_performance
        df['under_performance'] = df['niveau'] > 0
        
        # Générer les alertes
        def generate_alerts(row):
            alerts = []
            if row['ca'] < 100:
                alerts.append("CA faible")
            if row['taux_annulation'] > 0.3:
                alerts.append("Taux d'annulation élevé")
            elif row['taux_annulation'] > 0.15:
                alerts.append("Taux d'annulation modéré")
            if row['volume'] < 10:
                alerts.append("Faible volume de commandes")
            if row['niveau'] >= 3:
                alerts.append("Risque critique")
            elif row['niveau'] == 2:
                alerts.append("Risque modéré")
            
            return " | ".join(alerts) if alerts else "Performance OK"
        
        df['alerts_str'] = df.apply(generate_alerts, axis=1)
        
        return df

#connexion a la base de donnees

def get_engine():
    """Crée la connexion à PostgreSQL"""
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(connection_string)

def load_data_from_db():
    """Charge les données depuis PostgreSQL"""
    try:
        engine = get_engine()
        
        # requete pour récupérer toutes les données
        query = """
            SELECT 
                product_category_name_english,
                purchase_year,
                purchase_month,
                ca,
                volume,
                taux_annulation
            FROM monitoring_data
            WHERE is_active = true
            ORDER BY purchase_year, purchase_month
        """
        
        logging.info(" Lecture des données depuis PostgreSQL...")
        df = pd.read_sql_query(query, engine)
        engine.dispose()
        
        logging.info(f" {len(df)} lignes chargées depuis la base")
        return df
        
    except Exception as e:
        logging.error(f"Erreur lors du chargement: {e}")
        return pd.DataFrame()

def test_connection():
    """Teste la connexion à PostgreSQL"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            conn.commit()
        engine.dispose()
        return True
    except Exception as e:
        logging.error(f" Erreur de connexion: {e}")
        return False

def get_all_categories():
    """Récupère la liste de toutes les catégories"""
    try:
        engine = get_engine()
        query = """
            SELECT DISTINCT product_category_name_english
            FROM monitoring_data
            WHERE is_active = true
            ORDER BY product_category_name_english
        """
        df = pd.read_sql_query(query, engine)
        engine.dispose()
        return df['product_category_name_english'].tolist()
    except Exception as e:
        logging.error(f" Erreur: {e}")
        return []

def predict_for_category(category_name):
    """Prédit pour une catégorie spécifique"""
    try:
        engine = get_engine()
        query = """
            SELECT 
                product_category_name_english,
                purchase_year,
                purchase_month,
                ca,
                volume,
                taux_annulation
            FROM monitoring_data
            WHERE is_active = true
            AND product_category_name_english = :category
            ORDER BY purchase_year, purchase_month
        """
        df = pd.read_sql_query(query, engine, params={'category': category_name})
        engine.dispose()
        
        if df.empty:
            return None
        
        predictor = LevelPredictor()
        return predictor.predict_with_details(df)
    except Exception as e:
        logging.error(f" Erreur: {e}")
        return None

def predict_for_period(year, month):
    """Prédit pour une période spécifique"""
    try:
        engine = get_engine()
        query = """
            SELECT 
                product_category_name_english,
                purchase_year,
                purchase_month,
                ca,
                volume,
                taux_annulation
            FROM monitoring_data
            WHERE is_active = true
            AND purchase_year = :year
            AND purchase_month = :month
            ORDER BY product_category_name_english
        """
        df = pd.read_sql_query(query, engine, params={'year': year, 'month': month})
        engine.dispose()
        
        if df.empty:
            return None
        
        predictor = LevelPredictor()
        return predictor.predict_with_details(df)
    except Exception as e:
        logging.error(f" Erreur: {e}")
        return None

#main

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST DE QUICK_PREDICT")
    print("="*60)
    
    if test_connection():
        print(" Connexion OK")
        df = load_data_from_db()
        if not df.empty:
            predictor = LevelPredictor()
            df_result = predictor.predict_with_details(df)
            print(f"\n {len(df_result)} lignes prédites")
            print("\nDistribution des niveaux:")
            print(df_result['niveau'].value_counts().sort_index())
    else:
        print(" Connexion échouée")