import pandas as pd

#les deux tables orders et order_items
orders=pd.read_csv(r"C:\Users\Pc\Documents\pfa\data\raw\orders.csv")
order_items=pd.read_csv(r"C:\Users\Pc\Documents\pfa\data\raw\order_items.csv")

#dataframe qui combine les deux tables
df=pd.merge(orders,order_items ,on='order_id')

print(df.head())
print("colonnes de notre dataset:\n", df.columns)

# #transformer en csv
# df.to_csv(r"C:\Users\Pc\Documents\pfa\data\processed\final_dataset.csv",index=False)

