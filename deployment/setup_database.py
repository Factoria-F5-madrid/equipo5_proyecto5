import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

def setup_database():
    """Configurar base de datos con datos iniciales"""
    
  
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'healthdb'),
        'user': os.getenv('DB_USER', 'admin'),
        'password': os.getenv('DB_PASSWORD', 'admin')
    }
    
  
    
    try:
   
        DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(DATABASE_URL)
        
      
        create_tables_sql = """
        -- Tabla principal de datos
        CREATE TABLE IF NOT EXISTS world_health (
          id SERIAL PRIMARY KEY,
          country TEXT,
          year INT,
          status TEXT,
          life_expectancy FLOAT,
          adult_mortality FLOAT,
          infant_deaths INT,
          alcohol FLOAT,
          percentage_expenditure FLOAT,
          hepatitis_b FLOAT,
          measles INT,
          bmi FLOAT,
          under_five_deaths INT,
          polio FLOAT,
          total_expenditure FLOAT,
          diphtheria FLOAT,
          hiv_aids FLOAT,
          gdp FLOAT,
          population FLOAT,
          thinness_1_19_years FLOAT,
          thinness_5_9_years FLOAT,
          income_composition_of_resources FLOAT,
          schooling FLOAT
        );

        -- Tablas para MLOps
        CREATE TABLE IF NOT EXISTS models (
          model_id SERIAL PRIMARY KEY,
          version VARCHAR(50) UNIQUE NOT NULL,
          trained_at TIMESTAMP DEFAULT NOW(),
          algorithm VARCHAR(100),
          hyperparams JSONB,
          metrics JSONB,
          status VARCHAR(20) DEFAULT 'candidate'
        );

        CREATE TABLE IF NOT EXISTS experiments (
          experiment_id SERIAL PRIMARY KEY,
          name VARCHAR(100),
          description TEXT,
          created_at TIMESTAMP DEFAULT NOW(),
          traffic_split JSONB
        );

        CREATE TABLE IF NOT EXISTS experiment_results (
          result_id SERIAL PRIMARY KEY,
          experiment_id INT REFERENCES experiments(experiment_id),
          model_id INT REFERENCES models(model_id),
          feedback_id INT REFERENCES feedback(feedback_id),
          success BOOLEAN,
          latency_ms INT,
          created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS data_drift (
          drift_id SERIAL PRIMARY KEY,
          checked_at TIMESTAMP DEFAULT NOW(),
          column_name VARCHAR(100),
          reference_stats JSONB,
          current_stats JSONB,
          drift_metric FLOAT
        );

        CREATE TABLE IF NOT EXISTS feedback (
          feedback_id SERIAL PRIMARY KEY,
          input JSONB,
          prediction FLOAT,
          actual FLOAT,
          feedback_text TEXT,
          created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        with engine.connect() as conn:
            conn.execute(create_tables_sql)
            conn.commit()
        
     
        df = pd.read_csv('data/clean_data.csv')
        
     
        df.to_sql('world_health', engine, if_exists='replace', index=False)
    
        model_data = {
            'version': 'v1.0',
            'algorithm': 'RandomForestRegressor',
            'hyperparams': '{"n_estimators": 200, "max_depth": null, "random_state": 42}',
            'metrics': '{"r2": 0.969, "rmse": 1.649, "mae": 1.074}',
            'status': 'active'
        }
        
        with engine.connect() as conn:
            conn.execute("""
                INSERT INTO models (version, algorithm, hyperparams, metrics, status)
                VALUES (%(version)s, %(algorithm)s, %(hyperparams)s::jsonb, %(metrics)s::jsonb, %(status)s)
                ON CONFLICT (version) DO NOTHING
            """, model_data)
            conn.commit()
        
       
    except Exception as e:
        
        return False
    
    return True

if __name__ == "__main__":
    success = setup_database()
   
    sys.exit(1)
