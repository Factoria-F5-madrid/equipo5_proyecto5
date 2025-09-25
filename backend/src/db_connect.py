import os
from sqlalchemy import create_engine
import psycopg2
import psycopg2.extras

# Configuración de base de datos
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'healthdb'),
    'user': os.getenv('DB_USER', 'admin'),
    'password': os.getenv('DB_PASSWORD', 'admin')
}

# URL para SQLAlchemy
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Engine para pandas
engine = create_engine(DATABASE_URL)

def get_connection():
    """Obtener conexión directa a PostgreSQL"""
    return psycopg2.connect(**DB_CONFIG)

def get_cursor():
    """Obtener cursor con diccionarios"""
    conn = get_connection()
    return conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
