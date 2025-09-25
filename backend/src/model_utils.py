from db_connect import get_connection, get_cursor
import json
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def insert_model(version, algorithm, hyperparams, metrics, status="candidate", file_path=None):
    """Insertar nuevo modelo en la base de datos"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            INSERT INTO models (version, algorithm, hyperparams, metrics, status, file_path, trained_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING model_id;
        """, (version, algorithm, json.dumps(hyperparams), json.dumps(metrics), status, file_path, datetime.now()))
        model_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Model inserted with ID: {model_id}")
        return model_id
    except Exception as e:
        logger.error(f"Error inserting model: {e}")
        return None

def get_active_model():
    """Obtener el modelo activo"""
    try:
        conn, cur = get_cursor()
        cur.execute("SELECT * FROM models WHERE status='active' ORDER BY trained_at DESC LIMIT 1;")
        model = cur.fetchone()
        cur.close()
        conn.close()
        return model
    except Exception as e:
        logger.error(f"Error getting active model: {e}")
        return None

def get_model_by_id(model_id):
    """Obtener modelo por ID"""
    try:
        conn, cur = get_cursor()
        cur.execute("SELECT * FROM models WHERE model_id = %s;", (model_id,))
        model = cur.fetchone()
        cur.close()
        conn.close()
        return model
    except Exception as e:
        logger.error(f"Error getting model by ID: {e}")
        return None

def update_model_status(model_id, status):
    """Actualizar estado del modelo"""
    try:
        conn, cur = get_cursor()
        cur.execute("UPDATE models SET status = %s WHERE model_id = %s;", (status, model_id))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Model {model_id} status updated to {status}")
        return True
    except Exception as e:
        logger.error(f"Error updating model status: {e}")
        return False

def get_all_models():
    """Obtener todos los modelos"""
    try:
        conn, cur = get_cursor()
        cur.execute("SELECT * FROM models ORDER BY trained_at DESC;")
        models = cur.fetchall()
        cur.close()
        conn.close()
        return models
    except Exception as e:
        logger.error(f"Error getting all models: {e}")
        return []

def save_feature_importance(model_id, feature_importance_df):
    """Guardar importancia de características"""
    try:
        conn, cur = get_cursor()
        for idx, row in feature_importance_df.iterrows():
            cur.execute("""
                INSERT INTO feature_importance (model_id, feature_name, importance_score, rank)
                VALUES (%s, %s, %s, %s)
            """, (model_id, row['feature'], row['importance'], idx + 1))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Feature importance saved for model {model_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving feature importance: {e}")
        return False
