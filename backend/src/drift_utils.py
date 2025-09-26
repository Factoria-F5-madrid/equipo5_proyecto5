from db_connect import get_connection, get_cursor
import json
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_drift(column_name, reference_stats, current_stats, drift_metric,
              drift_detected=False, alert_level='green'):
    """Registrar drift de datos"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            INSERT INTO data_drift
            (column_name, reference_stats, current_stats, drift_metric,
             drift_detected, alert_level, checked_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (column_name, json.dumps(reference_stats), json.dumps(current_stats),
              drift_metric, drift_detected, alert_level, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Drift logged for column: {column_name}")
        return True
    except Exception as e:
        logger.error(f"Error logging drift: {e}")
        return False

def get_recent_drifts(limit=10):
    """Obtener drifts recientes"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            SELECT * FROM data_drift
            ORDER BY checked_at DESC
            LIMIT %s;
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error getting recent drifts: {e}")
        return []

def get_drift_summary(days=7):
    """Obtener resumen de drift por feature"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            SELECT
                column_name,
                COUNT(*) as total_checks,
                SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) as drift_incidents,
                AVG(drift_metric) as avg_drift_score,
                MAX(checked_at) as last_check
            FROM data_drift
            WHERE checked_at >= NOW() - INTERVAL '%s days'
            GROUP BY column_name
            ORDER BY drift_incidents DESC
        """, (days,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error getting drift summary: {e}")
        return []

def create_drift_alert(alert_type, severity, message, affected_features):
    """Crear alerta de drift"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            INSERT INTO drift_alerts
            (alert_type, severity, message, affected_features, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (alert_type, severity, message, json.dumps(affected_features), datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        logger.warning(f"Drift alert created: {alert_type} - {message}")
        return True
    except Exception as e:
        logger.error(f"Error creating drift alert: {e}")
        return False

def get_active_alerts():
    """Obtener alertas activas"""
    try:
        conn, cur = get_cursor()
        cur.execute("""
            SELECT * FROM drift_alerts
            WHERE status = 'open'
            ORDER BY created_at DESC
        """)
        alerts = cur.fetchall()
        cur.close()
        conn.close()
        return alerts
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        return []
