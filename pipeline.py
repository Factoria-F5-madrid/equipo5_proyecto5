import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

class LifeExpectancyPipeline:
    def __init__(self):
        print("Initializing Life Expectancy Pipeline...")

        self.model_path = 'models/best_life_expectancy_model.pkl'
        self.preprocessor_path = 'models/preprocessor.pkl'

        # Cargar modelo
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print("Model not found, will create a new one.")
            self.model = None

        # Cargar preprocessor
        if os.path.exists(self.preprocessor_path):
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)
                print("Preprocessor loaded successfully")
            except Exception as e:
                print(f"Error loading preprocessor: {e}")
                self.preprocessor = None
        else:
            print("Preprocessor not found, will create a new one.")
            self.preprocessor = None

    def create_preprocessor_and_model(self, data_path='data/clean_data.csv'):
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found. Cannot train model.")
            return

        df = pd.read_csv(data_path)
        target = 'life_expectancy'
        exclude_cols = ['country', 'year', 'status', target]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        num_features = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
        cat_features = [c for c in feature_cols if df[c].dtype == 'object']

        # Crear preprocessor
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])

        X = df[feature_cols]
        y = df[target]

        X_processed = preprocessor.fit_transform(X)

        # Entrenar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_processed, y)

        # Guardar preprocessor y modelo
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, self.preprocessor_path)
        joblib.dump(model, self.model_path)

        self.preprocessor = preprocessor
        self.model = model

        print("Preprocessor and model created and saved successfully.")

    def transform_data(self, user_data):
        if self.preprocessor is None:
            print("Preprocessor not found. Creating from default data...")
            self.create_preprocessor_and_model()
        df = pd.DataFrame([user_data])
        return self.preprocessor.transform(df)

    def predict(self, user_data):
        transformed = self.transform_data(user_data)
        if self.model is None:
            print("Model not found. Creating new one...")
            self.create_preprocessor_and_model()
        return round(self.model.predict(transformed)[0], 2)

# Test rápido
if __name__ == "__main__":
    pipeline = LifeExpectancyPipeline()
    sample_data = {
        'country': 'Spain',
        'year': 2020,
        'status': 'Developed',
        'adult_mortality': 50.0,
        'infant_deaths': 100,
        'alcohol': 10.5,
        'percentage_expenditure': 200.0,
        'hepatitis_b': 80.0,
        'measles': 1000,
        'bmi': 25.0,
        'under_five_deaths': 150,
        'polio': 90.0,
        'total_expenditure': 10.0,
        'diphtheria': 85.0,
        'hiv/aids': 0.5,
        'gdp': 30000.0,
        'population': 50000000,
        'thinness__1_19_years': 5.0,
        'thinness_5_9_years': 4.0,
        'income_composition_of_resources': 0.8,
        'schooling': 12.0
    }
    pred = pipeline.predict(sample_data)
    print(f"Predicted life expectancy: {pred} years")
