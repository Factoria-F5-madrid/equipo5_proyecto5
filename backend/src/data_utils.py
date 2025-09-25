import pandas as pd
from db_connect import engine, get_connection
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_world_health():
    """Leer datos de salud mundial desde la base de datos"""
    try:
        query = "SELECT * FROM world_health"
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"Error reading world_health data: {e}")
        return pd.DataFrame()

def read_feedback():
    """Leer feedback de usuarios desde la base de datos"""
    try:
        query = "SELECT * FROM feedback"
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"Error reading feedback data: {e}")
        return pd.DataFrame()

def read_training_data():
    """Leer datos de entrenamiento desde la base de datos"""
    try:
        query = "SELECT * FROM training_data"
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"Error reading training data: {e}")
        return pd.DataFrame()

def read_reference_data():
    """Leer datos de referencia para drift monitoring"""
    try:
        query = "SELECT * FROM reference_data"
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"Error reading reference data: {e}")
        return pd.DataFrame()

def save_training_data(df):
    """Guardar datos de entrenamiento en la base de datos"""
    try:
        df.to_sql('training_data', engine, if_exists='replace', index=False)
        logger.info(f"Training data saved: {len(df)} rows")
        return True
    except Exception as e:
        logger.error(f"Error saving training data: {e}")
        return False

def save_reference_data(df):
    """Guardar datos de referencia en la base de datos"""
    try:
        df.to_sql('reference_data', engine, if_exists='replace', index=False)
        logger.info(f"Reference data saved: {len(df)} rows")
        return True
    except Exception as e:
        logger.error(f"Error saving reference data: {e}")
        return False
