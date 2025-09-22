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

CREATE TABLE IF NOT EXISTS feedback (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMP DEFAULT NOW(),
  input_data JSONB,
  prediction FLOAT,
  feedback TEXT
);

COPY world_health(country, year, status, life_expectancy, adult_mortality, infant_deaths, alcohol,
                  percentage_expenditure, hepatitis_b, measles, bmi, under_five_deaths, polio,
                  total_expenditure, diphtheria, hiv_aids, gdp, population, thinness_1_19_years,
                  thinness_5_9_years, income_composition_of_resources, schooling)
FROM '/docker-entrypoint-initdb.d/Life_Expectancy_Data.csv'
DELIMITER ','
CSV HEADER;