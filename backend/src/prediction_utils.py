from db_connect import get_connection, get_cursor
import json
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_prediction(country, year, status, predicted_life_expectancy, model_version, 
                   input_data, confidence_score=None, processing_time_ms=None, 
                   user_session_id=None):
    """Guardar predicción en la base de datos"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            INSERT INTO predictions 
            (country, year, status, predicted_life_expectancy, confidence_score, 
             model_version, input_data, processing_time_ms, user_session_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING prediction_id;
        """, (country, year, status, predicted_life_expectancy, confidence_score,
              model_version, json.dumps(input_data), processing_time_ms, 
              user_session_id, datetime.now()))
        prediction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Prediction saved with ID: {prediction_id}")
        return prediction_id
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        return None

def get_predictions(limit=100, country=None, start_date=None, end_date=None):
    """Obtener predicciones con filtros"""
    try:
        conn, cur = get_cursor()
        query = """
            SELECT p.*, m.algorithm, m.status as model_status
            FROM predictions p
            LEFT JOIN models m ON p.model_version = m.version
            WHERE 1=1
        """
        params = []
        
        if country:
            query += " AND p.country = %s"
            params.append(country)
        
        if start_date:
            query += " AND p.created_at >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND p.created_at <= %s"
            params.append(end_date)
        
        query += " ORDER BY p.created_at DESC LIMIT %s"
        params.append(limit)
        
        cur.execute(query, params)
        predictions = cur.fetchall()
        cur.close()
        conn.close()
        return predictions
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return []

def get_prediction_stats(start_date=None, end_date=None):
    """Obtener estadísticas de predicciones"""
    try:
        conn, cur = get_cursor()
        query = """
            SELECT 
                COUNT(*) as total_predictions,
                AVG(predicted_life_expectancy) as avg_life_expectancy,
                MIN(predicted_life_expectancy) as min_life_expectancy,
                MAX(predicted_life_expectancy) as max_life_expectancy,
                COUNT(DISTINCT country) as countries_count
            FROM predictions
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)
        
        cur.execute(query, params)
        stats = cur.fetchone()
        cur.close()
        conn.close()
        return dict(stats) if stats else {}
    except Exception as e:
        logger.error(f"Error getting prediction stats: {e}")
        return {}

def get_predictions_by_country():
    """Obtener predicciones agrupadas por país"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            SELECT 
                country,
                COUNT(*) as prediction_count,
                AVG(predicted_life_expectancy) as avg_prediction,
                MAX(created_at) as last_prediction
            FROM predictions
            GROUP BY country
            ORDER BY prediction_count DESC
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error getting predictions by country: {e}")
        return []
