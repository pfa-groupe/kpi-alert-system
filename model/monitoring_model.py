import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Chargement et Préparation
df = pd.read_csv('dataset_final_agrege.csv')
le = LabelEncoder()
df['cat_encoded'] = le.fit_transform(df['product_category_name_english'])

# 2. FEATURE ENGINEERING : Ajout de la tendance (crucial pour la précision)
df = df.sort_values(['product_category_name_english', 'purchase_year', 'purchase_month'])
df['ca_growth'] = df.groupby('product_category_name_english')['ca'].pct_change().fillna(0)

# 3. Mélange et Split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
features = ['purchase_month', 'ca', 'taux_annulation', 'volume', 'cat_encoded', 'ca_growth']
X = df[features]
y = df['niveau']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraînement avec équilibrage (class_weight='balanced')
model = RandomForestClassifier(
    n_estimators=300, 
    class_weight='balanced', 
    max_depth=15, 
    random_state=42
)
model.fit(X_train, y_train)

# 5. Résultat
score = model.score(X_test, y_test)
print(f"Nouvelle précision : {score:.2%}")

# import matplotlib.pyplot as plt
# pd.Series(model.feature_importances_, index=features).sort_values().plot(kind='barh')
# plt.show()

import joblib

# Sauvegarder le modèle et l'encodeur
joblib.dump(model, 'modele_sous_performance.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Modèle sauvegardé avec succès.")