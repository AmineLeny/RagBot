import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database_if_not_exists():
    db_name = "ragbot_db"
    db_user = "postgres"
    db_password = "amine123"
    db_host = "localhost"
    db_port = "5432"

    try:
        # Connexion à la base 'postgres'
        conn = psycopg2.connect(
            dbname="postgres",
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Vérifier si la base existe
        cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (db_name,))
        exists = cursor.fetchone()

        # Si elle n'existe pas, la créer
        if not exists:
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"La base '{db_name}' a été créée.")
        else:
            print(f"La base '{db_name}' existe déjà.")

        cursor.close()
        conn.close()

    except Exception as e:
        print("❌ Erreur lors de la création de la base :", e)