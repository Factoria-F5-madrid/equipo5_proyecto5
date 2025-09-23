from db_connection import engine
import json

def log_drift(column_name, reference_stats, current_stats, drift_metric):
  conn = engine()
  cur = conn.cursor()
  cur.execute("""
    INSERT INTO data_drift (column_name, reference_stats, current_stats, drift_metric)
    VALUES (%s, %s, %s, %s);
  """, (column_name, json.dumps(reference_stats), json.dumps(current_stats), drift_metric))
  conn.commit()
  cur.close()
  conn.close()

def get_recent_drifts(limit=10):
  conn = engine()
  cur = conn.cursor()
  cur.execute("""
    SELECT * FROM data_drift
    ORDER BY checked_at DESC
    LIMIT %s;
  """, (limit,))
  rows = cur.fetchall()
  cur.close()
  conn.close()
  return rows
