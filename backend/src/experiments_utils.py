from db_connection import engine
import json

def create_experiment(name, description, traffic_split):
  conn = engine()
  cur = conn.cursor()
  cur.execute("""
    INSERT INTO experiments (name, description, traffic_split)
    VALUES (%s, %s, %s)
    RETURNING experiment_id;
  """, (name, description, json.dumps(traffic_split)))
  experiment_id = cur.fetchone()[0]
  conn.commit()
  cur.close()
  conn.close()
  return experiment_id

def log_experiment_result(experiment_id, model_id, feedback_id, success, latency_ms):
  conn = engine()
  cur = conn.cursor()
  cur.execute("""
    INSERT INTO experiment_results (experiment_id, model_id, feedback_id, success, latency_ms)
    VALUES (%s, %s, %s, %s, %s);
  """, (experiment_id, model_id, feedback_id, success, latency_ms))
  conn.commit()
  cur.close()
  conn.close()
