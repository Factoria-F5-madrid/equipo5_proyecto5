import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

class LifeExpectancyPipeline:
    """
    Complete pipeline for life expectancy prediction
    Handles validation, transformation, prediction, and storage
    """
    
    def __init__(self):
        """Initialize the pipeline with model and preprocessor"""
        print(" Initializing Life Expectancy Pipeline...")
        
        # Load trained model
        try:
            self.model = joblib.load('models/best_life_expectancy_model.pkl')
            print(" Model loaded successfully")
        except FileNotFoundError:
            print(" Model file not found. Please train the model first.")
            self.model = None
        
        # Load preprocessor (if available)
        try:
            self.preprocessor = joblib.load('models/preprocessor.pkl')
            print(" Preprocessor loaded successfully")
        except FileNotFoundError:
            print(" Preprocessor not found. Will create basic preprocessing.")
            self.preprocessor = None
        
        # Define valid ranges for each variable
        self.valid_ranges = self._get_valid_ranges()
        
        # Valid countries and status
        self.valid_countries = self._get_valid_countries()
        self.valid_status = ['Developing', 'Developed']
        
        print("Pipeline initialized successfully")
    
    def _get_valid_ranges(self):
        """
        Define valid ranges for each numerical variable
        Based on analysis of original dataset
        """
        return {
            'adult_mortality': (0, 1000),      # Deaths per 1000 population
            'infant_deaths': (0, 10000),       # Infant deaths
            'alcohol': (0, 20),                # Alcohol consumption per capita
            'percentage_expenditure': (0, 1000), # Health expenditure percentage
            'hepatitis_b': (0, 100),           # Hepatitis B vaccination coverage
            'measles': (0, 100000),            # Measles cases
            'bmi': (10, 50),                   # Body mass index
            'under_five_deaths': (0, 10000),   # Under 5 deaths
            'polio': (0, 100),                 # Polio vaccination coverage
            'total_expenditure': (0, 20),      # Total health expenditure
            'diphtheria': (0, 100),            # Diphtheria vaccination coverage
            'hiv/aids': (0, 50),               # HIV/AIDS prevalence
            'gdp': (0, 100000),                # GDP per capita
            'population': (1000, 2000000000),  # Country population (up to 2 billion)
            'thinness__1_19_years': (0, 50),   # Thinness 1-19 years
            'thinness_5_9_years': (0, 50),     # Thinness 5-9 years
            'income_composition_of_resources': (0, 1), # Income composition
            'schooling': (0, 20)               # Years of schooling
        }
    
    def _get_valid_countries(self):
        """
        Get list of valid countries from the original dataset
        """
        try:
            # Load original data to get country list
            df = pd.read_csv('data/clean_data.csv')
            return sorted(df['country'].unique().tolist())
        except FileNotFoundError:
            # Fallback list if data file not found
            return ['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina', 
                   'Australia', 'Austria', 'Bangladesh', 'Belgium', 'Brazil',
                   'Canada', 'Chile', 'China', 'Colombia', 'Cuba', 'Denmark',
                   'Egypt', 'Finland', 'France', 'Germany', 'Ghana', 'Greece',
                   'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel',
                   'Italy', 'Japan', 'Kenya', 'Kuwait', 'Malaysia', 'Mexico',
                   'Netherlands', 'New Zealand', 'Nigeria', 'Norway', 'Pakistan',
                   'Peru', 'Philippines', 'Poland', 'Portugal', 'Russia',
                   'Saudi Arabia', 'Singapore', 'South Africa', 'South Korea',
                   'Spain', 'Sri Lanka', 'Sweden', 'Switzerland', 'Thailand',
                   'Turkey', 'Ukraine', 'United Kingdom', 'United States',
                   'Venezuela', 'Vietnam']
    
    def validate_data(self, user_data):
        """
        STEP 1: DATA VALIDATION
        Validates user input data against defined ranges and formats
        
        Args:
            user_data (dict): User input data
        
        Returns:
            tuple: (is_valid, message)
        """
        print(" Validating user data...")
        
        errors = []
        
        # Validate numerical variables
        for variable, (min_val, max_val) in self.valid_ranges.items():
            if variable in user_data:
                value = user_data[variable]
                
                # Check if it's a number (exclude booleans)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"{variable}: Must be a number, received {type(value).__name__}")
                    continue
                
                # Check range
                if not (min_val <= value <= max_val):
                    errors.append(f"{variable}: {value} is outside valid range ({min_val}-{max_val})")
        
        # Validate country
        if 'country' in user_data:
            if user_data['country'] not in self.valid_countries:
                errors.append(f"Country: '{user_data['country']}' is not valid")
        
        # Validate status
        if 'status' in user_data:
            if user_data['status'] not in self.valid_status:
                errors.append(f"Status: '{user_data['status']}' is not valid. Must be 'Developing' or 'Developed'")
        
        # Validate year
        if 'year' in user_data:
            year = user_data['year']
            if not isinstance(year, int) or not (2000 <= year <= 2025):
                errors.append(f"Year: {year} must be an integer between 2000 and 2025")
        
        # Return result
        if errors:
            return False, errors
        else:
            return True, "Data is valid"
    
    def create_preprocessor(self):
        """
        STEP 2A: CREATE PREPROCESSOR
        Creates the same preprocessor used in model training
        """
        print(" Creating preprocessor...")
        
        # Load original data to fit preprocessor
        try:
            df = pd.read_csv('data/clean_data.csv')
            print(" Original data loaded for preprocessor fitting")
        except FileNotFoundError:
            print(" Error: Original data not found. Cannot create preprocessor.")
            return None
        
        # Define feature columns (same as in training)
        target = 'life_expectancy'
        exclude_cols = ['country', 'year', 'status', target]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Separate numerical and categorical features
        numerical_features = []
        categorical_features = []
        
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        print(f" Numerical features: {len(numerical_features)}")
        print(f" Categorical features: {len(categorical_features)}")
        
        # Create preprocessing pipelines
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        # Numerical transformer
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit preprocessor on training data
        X = df[feature_cols]
        preprocessor.fit(X)
        
        # Save preprocessor
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        print(" Preprocessor created and saved successfully")
        
        return preprocessor
    
    def transform_data(self, user_data):
        """
        STEP 2B: DATA TRANSFORMATION
        Transforms user data to the format expected by the model
        
        Args:
            user_data (dict): Validated user input data
        
        Returns:
            numpy.ndarray: Transformed data ready for prediction
        """
        print(" Transforming user data...")
        
        # Create preprocessor if not available
        if self.preprocessor is None:
            print(" Preprocessor not found. Creating new one...")
            self.preprocessor = self.create_preprocessor()
            if self.preprocessor is None:
                return None
        
        try:
            # Convert user data to DataFrame
            df = pd.DataFrame([user_data])
            
            # Apply same preprocessing as training
            processed_data = self.preprocessor.transform(df)
            
            print(f" Data transformed successfully. Shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            print(f" Error transforming data: {e}")
            return None
    
    def predict(self, transformed_data):
        """
        STEP 3: PREDICTION
        Uses the trained model to predict life expectancy
        
        Args:
            transformed_data (numpy.ndarray): Preprocessed data ready for prediction
        
        Returns:
            float: Predicted life expectancy in years
        """
        print(" Making prediction...")
        
        if self.model is None:
            print(" Error: Model not loaded. Cannot make prediction.")
            return None
        
        try:
            # Make prediction using the trained model
            prediction = self.model.predict(transformed_data)[0]
            
            # Round to 2 decimal places
            prediction = round(prediction, 2)
            
            print(f" Prediction successful: {prediction} years")
            return prediction
            
        except Exception as e:
            print(f" Error making prediction: {e}")
            return None
    
    def test_prediction(self):
        """
        Test the complete prediction pipeline with sample data
        """
        print(" Testing complete prediction pipeline...")
        
        # Sample valid data
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
        
        print("\n" + "="*30)
        print("COMPLETE PREDICTION PIPELINE TEST")
        print("="*30)
        
        # Step 1: Validate data
        print("\n1. VALIDATING DATA:")
        is_valid, message = self.validate_data(sample_data)
        if not is_valid:
            print(f" Validation failed: {message}")
            return
        print(f" Validation passed: {message}")
        
        # Step 2: Transform data
        print("\n2. TRANSFORMING DATA:")
        transformed_data = self.transform_data(sample_data)
        if transformed_data is None:
            print(" Transformation failed!")
            return
        print(f" Transformation successful! Shape: {transformed_data.shape}")
        
        # Step 3: Make prediction
        print("\n3. MAKING PREDICTION:")
        prediction = self.predict(transformed_data)
        if prediction is None:
            print(" Prediction failed!")
            return
        
        # Display results
        print("\n" + "="*30)
        print("PREDICTION RESULTS")
        print("="*30)
        print(f" Input data: Spain, 2020, Developed country")
        print(f" Predicted life expectancy: {prediction} years")
        print(f" Model used: Random Forest (Optimized)")
        print("="*30)
        
        return prediction
    
    def test_transformation(self):
        """
        Test the transformation function with sample data
        """
        print(" Testing transformation function...")
        
        # Sample valid data
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
        
        # Test transformation
        print("\nTesting data transformation:")
        transformed_data = self.transform_data(sample_data)
        
        if transformed_data is not None:
            print(f" Transformation successful!")
            print(f" Transformed data shape: {transformed_data.shape}")
            print(f" Data type: {transformed_data.dtype}")
            print(f" Sample values: {transformed_data[0][:5]}...")  # First 5 values
        else:
            print(" Transformation failed!")
    
    def test_validation(self):
        """
        Test the validation function with sample data
        """
        print(" Testing validation function...")
        
        # Valid data
        valid_data = {
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
        
        # Invalid data
        invalid_data = {
            'country': 'Spain',
            'year': 2020,
            'status': 'Invalid',
            'adult_mortality': -50.0,  # Invalid negative value
            'infant_deaths': 100,
            'alcohol': 50.0,  # Too high value
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
        
        # Test valid data
        print("\nTesting valid data:")
        is_valid, message = self.validate_data(valid_data)
        print(f"Result: {is_valid}")
        print(f"Message: {message}")
        
        # Test invalid data
        print("\nTesting invalid data:")
        is_valid, message = self.validate_data(invalid_data)
        print(f"Result: {is_valid}")
        print(f"Message: {message}")


def main():
    """
    Main function to test the pipeline
    """
    print("=" * 50)
    print("LIFE EXPECTANCY PREDICTION PIPELINE")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = LifeExpectancyPipeline()
    
    # Test validation
    pipeline.test_validation()
    
    print("\n" + "=" * 50)
    print("PIPELINE STEP 1 COMPLETED: DATA VALIDATION")
    print("=" * 50)
    
    # Test transformation
    pipeline.test_transformation()
    
    print("\n" + "=" * 50)
    print("PIPELINE STEP 2 COMPLETED: DATA TRANSFORMATION")
    print("=" * 50)
    
    # Test complete prediction pipeline
    pipeline.test_prediction()
    
    print("\n" + "=" * 50)
    print("PIPELINE STEP 3 COMPLETED: PREDICTION")
    print("=" * 50)


if __name__ == "__main__":
    main()
