from db_utils import get_connection
import json

def insert_model(version, algorithm, hyperparams, metrics, status="candidate"):
  conn = get_connection()
  cur = conn.cursor()
  cur.execute("""
    INSERT INTO models (version, algorithm, hyperparams, metrics, status)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING model_id;
  """, (version, algorithm, json.dumps(hyperparams), json.dumps(metrics), status))
  model_id = cur.fetchone()[0]
  conn.commit()
  cur.close()
  conn.close()
  return model_id

def get_active_model():
  conn = get_connection()
  cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
  cur.execute("SELECT * FROM models WHERE status='active' ORDER BY trained_at DESC LIMIT 1;")
  model = cur.fetchone()
  cur.close()
  conn.close()
  return model
