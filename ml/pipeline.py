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
        
        self.model_path = 'models/best_life_expectancy_model.pkl'
        self.preprocessor_path = 'models/preprocessor.pkl'

      
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
             
            except Exception as e:
                
                self.model = None
        else:
            
            self.model = None

       
        if os.path.exists(self.preprocessor_path):
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)
            except Exception as e:
            
                self.preprocessor = None
        else:
          
            self.preprocessor = None

    def create_preprocessor_and_model(self, data_path='data/clean_data.csv'):
        if not os.path.exists(data_path):
        
            return

        df = pd.read_csv(data_path)
        target = 'life_expectancy'
        exclude_cols = ['country', 'year', 'status', target]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        num_features = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
        cat_features = [c for c in feature_cols if df[c].dtype == 'object']

       
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

      
        df_filtered = df.copy()
        
       
        from .country_data import get_developed_countries, get_country_corrections
        
      
        developed_countries = get_developed_countries()
        
        for country in developed_countries:
            mask = df_filtered['country'] == country
            df_filtered.loc[mask, 'status'] = 'Developed'
        
   
        country_corrections = get_country_corrections()
        
        corrections_applied = 0
        for country, correct_life in country_corrections.items():
            mask = df_filtered['country'] == country
            if mask.any():
                current_life = df_filtered.loc[mask, 'life_expectancy']
                if len(current_life) > 0 and abs(current_life.iloc[0] - correct_life) > 5:
                    df_filtered.loc[mask, 'life_expectancy'] = correct_life
                    corrections_applied += 1
        

        developed_mask = df_filtered['status'] == 'Developed'
        low_life_mask = df_filtered['life_expectancy'] < 70
        error_mask = developed_mask & low_life_mask
        
        if error_mask.sum() > 0:
            df_filtered = df_filtered[~error_mask]
        
        X = df_filtered[feature_cols]
        y = df_filtered[target]

        X_processed = preprocessor.fit_transform(X)

       
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.8
        )
        
       
        cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='neg_mean_absolute_error')
        
        model.fit(X_processed, y)

     
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, self.preprocessor_path)
        joblib.dump(model, self.model_path)

        self.preprocessor = preprocessor
        self.model = model


    def transform_data(self, user_data):
        try:
            if self.preprocessor is None:
                self.create_preprocessor_and_model()
            df = pd.DataFrame([user_data])
            return self.preprocessor.transform(df)
        except AttributeError as e:
            if "monotonic_cst" in str(e):
                self.create_preprocessor_and_model()
                df = pd.DataFrame([user_data])
                return self.preprocessor.transform(df)
            else:
                raise e

    def predict(self, user_data):
        try:
            transformed = self.transform_data(user_data)
            if self.model is None:
                self.create_preprocessor_and_model()
            
           
            ml_prediction = self.model.predict(transformed)[0]
            
          
            final_prediction = self._hybrid_prediction(user_data, ml_prediction)
            
            return round(final_prediction, 2)
        except AttributeError as e:
            if "monotonic_cst" in str(e):
                self.create_preprocessor_and_model()
                transformed = self.transform_data(user_data)
                ml_prediction = self.model.predict(transformed)[0]
                final_prediction = self._hybrid_prediction(user_data, ml_prediction)
                return round(final_prediction, 2)
            else:
                raise e
    
    def _hybrid_prediction(self, user_data, ml_prediction):
        """Estrategia híbrida: combinar predicción ML con datos reales"""
        
        from .country_data import get_real_life_expectancy, is_developed_country
        
        country = user_data.get('country', '')
        status = user_data.get('status', '')
        gdp = user_data.get('gdp', 0)
        
    
        real_life = get_real_life_expectancy(country)
        
        if real_life is not None:
            
            hybrid_prediction = 0.7 * real_life + 0.3 * ml_prediction
            
           
            hybrid_prediction = self._apply_feature_adjustments(user_data, hybrid_prediction, real_life)
            
            return hybrid_prediction
        
      
        else:
            return self._apply_domain_corrections(user_data, ml_prediction)
    
    def _apply_feature_adjustments(self, user_data, prediction, base_real_life):
        """Aplicar ajustes basados en características específicas del país"""
        
       
        gdp = user_data.get('gdp', 0)
        schooling = user_data.get('schooling', 0)
        adult_mortality = user_data.get('adult_mortality', 0)
        percentage_expenditure = user_data.get('percentage_expenditure', 0)
        
      
        if gdp > 50000:  
            prediction += 1.0
        elif gdp > 30000: 
            prediction += 0.5
        elif gdp < 5000: 
            prediction -= 1.0
        
      
        if schooling > 15:  
            prediction += 0.5
        elif schooling < 8:  
            prediction -= 1.0
        
      
        if adult_mortality > 200:  
            prediction -= 2.0
        elif adult_mortality < 50:  
            prediction += 1.0
        
      
        if percentage_expenditure > 10: 
            prediction += 0.5
        elif percentage_expenditure < 3: 
            prediction -= 1.0
        
      
        max_adjustment = 3.0
        if abs(prediction - base_real_life) > max_adjustment:
            if prediction > base_real_life:
                prediction = base_real_life + max_adjustment
            else:
                prediction = base_real_life - max_adjustment
        
        return prediction
    
    def _apply_domain_corrections(self, user_data, prediction):
        """Aplicar correcciones basadas en conocimiento del dominio"""
        
        from .country_data import get_real_life_expectancy, is_developed_country
        
        country = user_data.get('country', '')
        status = user_data.get('status', '')
        gdp = user_data.get('gdp', 0)
        
       
        real_life = get_real_life_expectancy(country)
        
        if real_life is not None:
           
            if abs(prediction - real_life) > 3:
                prediction = real_life
        
       
        elif status == 'Developed' and gdp > 20000:
            if prediction < 75:
                prediction = max(prediction, 78)
        
      
        elif status == 'Developing' and gdp < 5000:
            if prediction > 85:
                prediction = min(prediction, 80)
        
        return prediction


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
