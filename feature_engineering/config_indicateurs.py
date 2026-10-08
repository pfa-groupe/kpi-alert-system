import pandas as pd
import numpy as np

df=pd.read_csv(r"C:\Users\Pc\Documents\pfa\data\processed\final_dataset.csv")

#on definit les indicateurs de sous performance

#taux d'annulation eleve

def cancel_KPI(df, year, month):

    df_filtered = df[(df["purchase_year"] == year) & (df["purchase_month"] == month)]

    if df_filtered.empty :
        return None  # mois inexistant dans les données  #ignorer si moins de 30 commandes ce mois
    
    month_rate = df_filtered["is_canceled"].mean() #taux d annulation par le mois choisi

    SEUIL_ANNULATION = 0.01  # 1%  #car notre dataset ne contient que 6 lignes ou is_canceled est True

    return month_rate >  SEUIL_ANNULATION

 #*******************************************************

#chiffre d'affaire par mois fabile 

def ca_KPI(df,year,month):

    df_filtered = df[(df["purchase_year"] == year) & (df["purchase_month"] == month)]

    if df_filtered.empty:
        return None  # mois inexistant dans les données
    
    month_rate = df_filtered["total_price"].sum() #CA du mois choisi

    # exclure le mois evalué avant de calculer la moyenne
    df_other       = df[~((df["purchase_year"] == year) & (df["purchase_month"] == month))]
    monthly_ca     = df_other.groupby(["purchase_year", "purchase_month"])["total_price"].sum() 
    avg_monthly_ca = monthly_ca.mean() # Moyenne du CA par mois sur tous les  autres mois disponibles

    return month_rate < avg_monthly_ca

 #*******************************************************

#retard de livraison

def dely_KPI(df,year,month):

    df_filtered = df[(df["purchase_year"] == year) & (df["purchase_month"] == month)]

    if df_filtered.empty:
        return None  # mois inexistant dans les données
    
    month_dely= df_filtered["dely"].mean() #moyenne de retad de livraison par ce mois

    # exclure le mois evalué avant de calculer la moyenne
    df_other     = df[~((df["purchase_year"] == year) & (df["purchase_month"] == month))]
    global_delay = df_other["dely"].mean() #moyenne sur tous les autres mois

    return month_dely > global_delay 

 #*******************************************************
#  Volume de commandes faible
def volume_KPI(df, year, month):
    df_filtered = df[(df["purchase_year"] == year) & (df["purchase_month"] == month)]
    if df_filtered.empty:
        return None  # mois inexistant dans les données
    
    month_volume  = len(df_filtered)
    avg_volume    = df.groupby(["purchase_year","purchase_month"]).size().mean()
    
    return month_volume < avg_volume

 #*******************************************************
def rapport_KPI(df, year, month):

    ca_indicator     = ca_KPI(df, year, month) 
    cancel_indicator = cancel_KPI(df, year, month) 
    dely_indicator   = dely_KPI(df, year, month)
    orders_indicator =volume_KPI(df, year, month)

    # Si le mois n'existe pas dans les données, on arrete
    if any(v is None for v in [ca_indicator, cancel_indicator, dely_indicator, orders_indicator]):
        return False, 0, []

    # Convertir en bool explicitement
    ca_indicator     = bool(ca_indicator)
    cancel_indicator = bool(cancel_indicator)
    dely_indicator   = bool(dely_indicator)
    orders_indicator = bool(orders_indicator)

     # Compter le nombre d'indicateurs en alerte
    score = sum([ca_indicator, cancel_indicator, dely_indicator, orders_indicator])

    #niveaux_de_sous_performance
    niveau=0      # 0 => y a pas la sous performance:niveau normal
                  # 3 => niveau critique
                  # 2 =>niveau severe
                  # 1 =>niveau modere

    alerts=[]
    if ca_indicator:
        alerts.append("CA faible")

    if cancel_indicator:
        alerts.append("Taux d'annulation élevé")

    if dely_indicator:
        alerts.append("Retards de livraison")

    if orders_indicator:
        alerts.append("Faible volume de commandes")


    if score >= 3:
        niveau = 3 

    elif score == 2:
        niveau = 2 

    elif score == 1:
        niveau = 1

    else:
        niveau=0

    under_performance = niveau > 0
    return under_performance,niveau,alerts


# test
result, niveau, alerts = rapport_KPI(df, 2017, 8)
print(result)
print(niveau)
print(alerts)


# sauvgarder le label de sous performance dans notre dataset


# Calculer le niveau pour chaque ligne à partir de son (year, month)
month_labels = (
    df[["purchase_year", "purchase_month"]].drop_duplicates().apply(
        lambda row: pd.Series(
            rapport_KPI(df, row["purchase_year"], row["purchase_month"]),
            index=["under_performance", "niveau", "alerts"]
        ),
        axis=1
    )
)

month_labels = pd.concat(
    [df[["purchase_year", "purchase_month"]].drop_duplicates().reset_index(drop=True),
     month_labels.reset_index(drop=True)],
    axis=1
)


# Joindre les labels sur le dataset original
df = df.merge(month_labels, on=["purchase_year", "purchase_month"], how="left")

# convertir 'alerts' en chaîne pour le CSV
df["alerts_str"] = df["alerts"].apply(lambda x: " | ".join(x) if isinstance(x, list) else "")

df.to_csv(r"C:\Users\Pc\Documents\pfa\data\processed\final_dataset_labeled.csv", index=False)

print(df[["purchase_year", "purchase_month", "under_performance", "niveau", "alerts_str"]].drop_duplicates())
