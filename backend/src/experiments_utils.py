from db_connect import get_connection, get_cursor
import json
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_experiment(name, description, traffic_split, model_a_id, model_b_id, created_by=None):
    """Crear nuevo experimento A/B"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            INSERT INTO experiments
            (name, description, traffic_split, model_a_id, model_b_id, created_by, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING experiment_id;
        """, (name, description, json.dumps(traffic_split), model_a_id, model_b_id, created_by, datetime.now()))
        experiment_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Experiment created with ID: {experiment_id}")
        return experiment_id
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        return None

def log_experiment_result(experiment_id, model_id, feedback_id, success, latency_ms):
    """Registrar resultado de experimento"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            INSERT INTO experiment_results
            (experiment_id, model_id, feedback_id, success, latency_ms, created_at)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (experiment_id, model_id, feedback_id, success, latency_ms, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Experiment result logged for experiment {experiment_id}")
        return True
    except Exception as e:
        logger.error(f"Error logging experiment result: {e}")
        return False

def get_experiment(experiment_id):
    """Obtener experimento por ID"""
    try:
        conn, cur = get_cursor()
        cur.execute("SELECT * FROM experiments WHERE experiment_id = %s;", (experiment_id,))
        experiment = cur.fetchone()
        cur.close()
        conn.close()
        return experiment
    except Exception as e:
        logger.error(f"Error getting experiment: {e}")
        return None

def get_active_experiments():
    """Obtener experimentos activos"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            SELECT * FROM experiments
            WHERE status = 'running'
            ORDER BY created_at DESC
        """)
        experiments = cur.fetchall()
        cur.close()
        conn.close()
        return experiments
    except Exception as e:
        logger.error(f"Error getting active experiments: {e}")
        return []

def update_experiment_status(experiment_id, status):
    """Actualizar estado del experimento"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            UPDATE experiments
            SET status = %s, updated_at = %s
            WHERE experiment_id = %s
        """, (status, datetime.now(), experiment_id))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Experiment {experiment_id} status updated to {status}")
        return True
    except Exception as e:
        logger.error(f"Error updating experiment status: {e}")
        return False

def get_experiment_results(experiment_id):
    """Obtener resultados de experimento"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            SELECT * FROM experiment_results
            WHERE experiment_id = %s
            ORDER BY created_at DESC
        """, (experiment_id,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error getting experiment results: {e}")
        return []

def get_experiment_summary(experiment_id):
    """Obtener resumen de experimento"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            SELECT
                e.name,
                e.traffic_split,
                e.status,
                COUNT(er.experiment_id) as total_samples,
                AVG(er.latency_ms) as avg_latency,
                SUM(CASE WHEN er.success THEN 1 ELSE 0 END) as successful_predictions,
                SUM(CASE WHEN er.success THEN 0 ELSE 1 END) as failed_predictions
            FROM experiments e
            LEFT JOIN experiment_results er ON e.experiment_id = er.experiment_id
            WHERE e.experiment_id = %s
            GROUP BY e.experiment_id, e.name, e.traffic_split, e.status
        """, (experiment_id,))
        summary = cur.fetchone()
        cur.close()
        conn.close()
        return dict(summary) if summary else {}
    except Exception as e:
        logger.error(f"Error getting experiment summary: {e}")
        return {}
