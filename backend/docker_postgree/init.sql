-- init.sql
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

-- tablas para reentrenamiento:
CREATE TABLE models (
  model_id SERIAL PRIMARY KEY,
  version VARCHAR(50) UNIQUE NOT NULL,
  trained_at TIMESTAMP DEFAULT NOW(),
  algorithm VARCHAR(100),
  hyperparams JSONB,
  metrics JSONB,
  status VARCHAR(20) DEFAULT 'candidate' -- candidate / active / archived
);

CREATE TABLE experiments (
  experiment_id SERIAL PRIMARY KEY,
  name VARCHAR(100),
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  traffic_split JSONB -- {"model_v1": 0.5, "model_v2": 0.5}
);

CREATE TABLE experiment_results (
  result_id SERIAL PRIMARY KEY,
  experiment_id INT REFERENCES experiments(experiment_id),
  model_id INT REFERENCES models(model_id),
  feedback_id INT REFERENCES feedback(feedback_id),
  success BOOLEAN,
  latency_ms INT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE data_drift (
  drift_id SERIAL PRIMARY KEY,
  checked_at TIMESTAMP DEFAULT NOW(),
  column_name VARCHAR(100),
  reference_stats JSONB, -- baseline
  current_stats JSONB,   -- datos recientes
  drift_metric FLOAT     -- ej. KS-test, Jensen-Shannon, etc.
);

CREATE TABLE feedback (
  feedback_id SERIAL PRIMARY KEY,
  user_id VARCHAR(50),
  input JSONB,
  prediction FLOAT,
  actual FLOAT,
  created_at TIMESTAMP DEFAULT NOW()
);


COPY world_health(country, year, status, life_expectancy, adult_mortality, infant_deaths, alcohol,
                  percentage_expenditure, hepatitis_b, measles, bmi, under_five_deaths, polio,
                  total_expenditure, diphtheria, hiv_aids, gdp, population, thinness_1_19_years,
                  thinness_5_9_years, income_composition_of_resources, schooling)
FROM '/docker-entrypoint-initdb.d/Life_Expectancy_Data.csv'
DELIMITER ','
CSV HEADER;