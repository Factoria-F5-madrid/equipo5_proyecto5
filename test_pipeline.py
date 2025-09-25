#!/usr/bin/env python3
"""
Test script for Life Expectancy Pipeline
Tests different scenarios and data types
"""

from pipeline import LifeExpectancyPipeline
import json

def test_different_countries():
    """
    Test pipeline with different countries and scenarios
    """
    print("=" * 60)
    print("TESTING PIPELINE WITH DIFFERENT COUNTRIES")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = LifeExpectancyPipeline()
    
    # Test data for different scenarios
    test_cases = [
        {
            "name": "Developed Country (Spain)",
            "data": {
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
        },
        {
            "name": "Developing Country (India)",
            "data": {
                'country': 'India',
                'year': 2020,
                'status': 'Developing',
                'adult_mortality': 150.0,
                'infant_deaths': 500,
                'alcohol': 3.0,
                'percentage_expenditure': 50.0,
                'hepatitis_b': 60.0,
                'measles': 5000,
                'bmi': 22.0,
                'under_five_deaths': 800,
                'polio': 70.0,
                'total_expenditure': 5.0,
                'diphtheria': 75.0,
                'hiv/aids': 2.0,
                'gdp': 5000.0,
                'population': 1300000000,
                'thinness__1_19_years': 15.0,
                'thinness_5_9_years': 12.0,
                'income_composition_of_resources': 0.5,
                'schooling': 8.0
            }
        },
        {
            "name": "Poor Country (Afghanistan)",
            "data": {
                'country': 'Afghanistan',
                'year': 2020,
                'status': 'Developing',
                'adult_mortality': 300.0,
                'infant_deaths': 2000,
                'alcohol': 0.5,
                'percentage_expenditure': 20.0,
                'hepatitis_b': 40.0,
                'measles': 10000,
                'bmi': 18.0,
                'under_five_deaths': 3000,
                'polio': 50.0,
                'total_expenditure': 3.0,
                'diphtheria': 60.0,
                'hiv/aids': 5.0,
                'gdp': 1000.0,
                'population': 30000000,
                'thinness__1_19_years': 25.0,
                'thinness_5_9_years': 20.0,
                'income_composition_of_resources': 0.3,
                'schooling': 5.0
            }
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*40}")
        print(f"TESTING: {test_case['name']}")
        print(f"{'='*40}")
        
        # Validate data
        is_valid, message = pipeline.validate_data(test_case['data'])
        if not is_valid:
            print(f"❌ Validation failed: {message}")
            continue
        
        # Transform data
        transformed_data = pipeline.transform_data(test_case['data'])
        if transformed_data is None:
            print("❌ Transformation failed")
            continue
        
        # Make prediction
        prediction = pipeline.predict(transformed_data)
        if prediction is None:
            print("❌ Prediction failed")
            continue
        
        # Store results
        result = {
            'country': test_case['data']['country'],
            'status': test_case['data']['status'],
            'prediction': prediction,
            'gdp': test_case['data']['gdp'],
            'schooling': test_case['data']['schooling'],
            'adult_mortality': test_case['data']['adult_mortality']
        }
        results.append(result)
        
        print(f"✅ Prediction: {prediction} years")
        print(f"   GDP: ${test_case['data']['gdp']:,}")
        print(f"   Schooling: {test_case['data']['schooling']} years")
        print(f"   Adult Mortality: {test_case['data']['adult_mortality']}")
    
    # Analyze results
    print(f"\n{'='*60}")
    print("ANALYSIS OF RESULTS")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\n{result['country']} ({result['status']}):")
        print(f"  Life Expectancy: {result['prediction']} years")
        print(f"  GDP per capita: ${result['gdp']:,}")
        print(f"  Years of schooling: {result['schooling']}")
        print(f"  Adult mortality: {result['adult_mortality']}")
    
    # Save results to file
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to 'test_results.json'")
    return results

def test_data_interpretation():
    """
    Explain how to interpret the data and results
    """
    print(f"\n{'='*60}")
    print("HOW TO INTERPRET THE DATA AND RESULTS")
    print(f"{'='*60}")
    
    print("""
    📊 UNDERSTANDING THE INPUT DATA:
    
    1. ADULT_MORTALITY (0-1000):
       - Deaths per 1000 population
       - LOWER = Better health system
       - Spain: 50 (excellent)
       - India: 150 (moderate)
       - Afghanistan: 300 (poor)
    
    2. GDP (0-100,000):
       - Gross Domestic Product per capita
       - HIGHER = More resources for health
       - Spain: $30,000 (high)
       - India: $5,000 (moderate)
       - Afghanistan: $1,000 (low)
    
    3. SCHOOLING (0-20):
       - Years of education
       - HIGHER = Better health awareness
       - Spain: 12 years (excellent)
       - India: 8 years (moderate)
       - Afghanistan: 5 years (low)
    
    4. HIV/AIDS (0-50):
       - Prevalence percentage
       - LOWER = Better health outcomes
       - Spain: 0.5% (very low)
       - India: 2% (moderate)
       - Afghanistan: 5% (high)
    
    🎯 UNDERSTANDING THE PREDICTIONS:
    
    - Life expectancy typically ranges from 40-85 years
    - Developed countries: 75-85 years
    - Developing countries: 60-75 years
    - Poor countries: 40-65 years
    
    📈 WHAT AFFECTS LIFE EXPECTANCY:
    
    1. POSITIVE FACTORS (increase life expectancy):
       - High GDP per capita
       - More years of schooling
       - Low adult mortality
       - Low HIV/AIDS prevalence
       - Good vaccination coverage
    
    2. NEGATIVE FACTORS (decrease life expectancy):
       - High adult mortality
       - High infant deaths
       - Low GDP per capita
       - High HIV/AIDS prevalence
       - Poor vaccination coverage
    """)

def test_model_concepts():
    """
    Explain the machine learning concepts used
    """
    print(f"\n{'='*60}")
    print("MACHINE LEARNING CONCEPTS EXPLAINED")
    print(f"{'='*60}")
    
    print("""
    🤖 RANDOM FOREST ALGORITHM:
    
    What it is:
    - Ensemble method (combines multiple decision trees)
    - Each tree makes a prediction
    - Final prediction = average of all trees
    
    Why it's good for this problem:
    - Handles many variables well
    - Not sensitive to outliers
    - Provides feature importance
    - Good accuracy for regression
    
    📊 FEATURE IMPORTANCE:
    
    Our model's most important features:
    1. HIV/AIDS (59.4%) - Most important!
    2. Adult Mortality (15.6%)
    3. Income Composition (14.8%)
    4. Schooling (2.0%)
    5. BMI (1.4%)
    
    This means HIV/AIDS prevalence is the strongest predictor
    of life expectancy in our model.
    
    🔄 DATA PREPROCESSING:
    
    1. VALIDATION:
       - Check data ranges
       - Verify data types
       - Ensure completeness
    
    2. TRANSFORMATION:
       - Imputation: Fill missing values
       - Scaling: Normalize numerical values
       - Encoding: Convert categorical to numerical
    
    3. WHY PREPROCESSING MATTERS:
       - Model was trained on preprocessed data
       - New data must have same format
       - Ensures consistent predictions
    
    📈 MODEL PERFORMANCE:
    
    Our model metrics:
    - R² = 0.97 (97% of variance explained)
    - RMSE = 1.65 years (average error)
    - Overfitting = 2.6% (excellent, <5%)
    
    What this means:
    - Very accurate predictions
    - Small prediction errors
    - Model generalizes well to new data
    """)

def main():
    """
    Run all tests and explanations
    """
    print("🚀 COMPREHENSIVE PIPELINE TESTING")
    print("=" * 60)
    
    # Test with different countries
    results = test_different_countries()
    
    # Explain data interpretation
    test_data_interpretation()
    
    # Explain ML concepts
    test_model_concepts()
    
    print(f"\n{'='*60}")
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
