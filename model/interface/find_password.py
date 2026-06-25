# find_password.py
import psycopg2

passwords = ['', 'postgres', 'admin', 'root', 'password', '1234', 'postgresql','hasnaa']

print("🔍 Recherche du mot de passe PostgreSQL...")
print("="*50)

for pwd in passwords:
    try:
        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='postgres',
            user='postgres',
            password=pwd,
            connect_timeout=3
        )
        print(f"✅ SUCCÈS ! Le mot de passe est : '{pwd}'")
        conn.close()
        break
    except Exception as e:
        print(f"❌ '{pwd}' : Échec")