import pandas as pd
from db_connection import engine

def read_world_health():
    query = "SELECT * FROM world_health"
    return pd.read_sql(query, engine)

def read_feedback():
    query = "SELECT * FROM feedback"
    return pd.read_sql(query, engine)
