import unittest
import pandas as pd
import numpy as np
import json
import os
import sys
from unittest.mock import patch, MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class TestModelLoading(unittest.TestCase):
    """Test model loading and basic functionality"""

    def test_model_file_exists(self):
        """Test that model file exists"""
        model_path = 'models/best_life_expectancy_model.pkl'
        self.assertTrue(os.path.exists(model_path), f"Model file not found at {model_path}")

    def test_model_loading(self):
        """Test that model can be loaded successfully"""
        try:
            model = joblib.load('models/best_life_expectancy_model.pkl')
            self.assertIsInstance(model, RandomForestRegressor)
        except Exception as e:
            self.fail(f"Failed to load model: {e}")

    def test_model_has_required_attributes(self):
        """Test that loaded model has required attributes"""
        model = joblib.load('models/best_life_expectancy_model.pkl')
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'feature_importances_'))
        self.assertTrue(hasattr(model, 'n_estimators'))
        self.assertTrue(hasattr(model, 'max_depth'))

    def test_model_prediction_shape(self):
        """Test that model predictions have correct shape"""
        model = joblib.load('models/best_life_expectancy_model.pkl')
        dummy_data = np.random.rand(5, 25) 

        try:
            predictions = model.predict(dummy_data)
            self.assertEqual(predictions.shape, (5,))
            self.assertTrue(np.all(np.isfinite(predictions)))
        except Exception as e:
            class TestModelPerformance(unittest.TestCase):
    """Test model performance metrics"""

    def test_model_results_file_exists(self):
        """Test that model results file exists"""
        results_path = 'models/model_results.json'
        self.assertTrue(os.path.exists(results_path), f"Model results file not found at {results_path}")

    def test_model_results_structure(self):
        """Test that model results have expected structure"""
        with open('models/model_results.json', 'r') as f:
            results = json.load(f)
        required_keys = ['model', 'test_rmse', 'test_mae', 'test_r2', 'train_r2', 'overfitting', 'best_params']
        for key in required_keys:
            self.assertIn(key, results, f"Missing key: {key}")

    def test_model_performance_metrics(self):
        """Test that model performance metrics are within acceptable ranges"""
        with open('models/model_results.json', 'r') as f:
            results = json.load(f)
        r2_score = results['test_r2']
        self.assertGreater(r2_score, 0.8, f"R² score too low: {r2_score}")
        self.assertLessEqual(r2_score, 1.0, f"R² score invalid: {r2_score}")
        rmse = results['test_rmse']
        self.assertLess(rmse, 10.0, f"RMSE too high: {rmse}")
        self.assertGreater(rmse, 0.0, f"RMSE invalid: {rmse}")
        mae = results['test_mae']
        self.assertLess(mae, 10.0, f"MAE too high: {mae}")
        self.assertGreater(mae, 0.0, f"MAE invalid: {mae}")
        overfitting = results['overfitting']
        self.assertLess(overfitting, 0.1, f"Overfitting too high: {overfitting}")
        self.assertGreaterEqual(overfitting, 0.0, f"Overfitting invalid: {overfitting}")

    def test_hyperparameters_structure(self):
        """Test that hyperparameters have expected structure"""
        with open('models/model_results.json', 'r') as f:
            results = json.load(f)

        best_params = results['best_params']
        self.assertIsInstance(best_params, dict)
        expected_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        for param in expected_params:
            self.assertIn(param, best_params, f"Missing hyperparameter: {param}")

class TestFeatureImportance(unittest.TestCase):
    """Test feature importance analysis"""

    def test_feature_importance_file_exists(self):
        """Test that feature importance file exists"""
        importance_path = 'models/feature_importance.csv'
        self.assertTrue(os.path.exists(importance_path), f"Feature importance file not found at {importance_path}")

    def test_feature_importance_structure(self):
        """Test that feature importance file has expected structure"""
        df = pd.read_csv('models/feature_importance.csv')
        self.assertIn('feature', df.columns)
        self.assertIn('importance', df.columns)
        self.assertTrue(df['importance'].dtype in ['float64', 'int64'])
        self.assertTrue((df['importance'] >= 0).all(), "Importance values should be non-negative")
        self.assertTrue((df['importance'] <= 1).all(), "Importance values should be <= 1")

    def test_feature_importance_sum(self):
        """Test that feature importance values sum to approximately 1"""
        df = pd.read_csv('models/feature_importance.csv')
        importance_sum = df['importance'].sum()
        self.assertAlmostEqual(importance_sum, 1.0, places=2,
                             msg=f"Feature importance sum should be ~1.0, got {importance_sum}")

    def test_top_features_exist(self):
        """Test that we have meaningful top features"""
        df = pd.read_csv('models/feature_importance.csv')
        top_features = df.nlargest(5, 'importance')
        self.assertGreaterEqual(len(top_features), 5)
        top_importance = top_features.iloc[0]['importance']
        self.assertGreater(top_importance, 0.01, "Top feature should have importance > 0.01")

class TestDataConsistency(unittest.TestCase):
    """Test data consistency and preprocessing"""

    def test_clean_data_exists(self):
        """Test that clean data file exists"""
        data_path = 'data/clean_data.csv'
        self.assertTrue(os.path.exists(data_path), f"Clean data file not found at {data_path}")

    def test_clean_data_structure(self):
        """Test that clean data has expected structure"""
        df = pd.read_csv('data/clean_data.csv')
        self.assertGreater(len(df), 0, "Clean data should not be empty")
        required_columns = ['life_expectancy', 'country', 'year', 'status']
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    def test_target_variable_range(self):
        """Test that target variable (life_expectancy) has reasonable range"""
        df = pd.read_csv('data/clean_data.csv')

        life_expectancy = df['life_expectancy']
        self.assertGreater(life_expectancy.min(), 30, f"Life expectancy too low: {life_expectancy.min()}")
        self.assertLess(life_expectancy.max(), 100, f"Life expectancy too high: {life_expectancy.max()}")
        missing_count = life_expectancy.isnull().sum()
        self.assertEqual(missing_count, 0, f"Life expectancy has {missing_count} missing values")

if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestModelLoading))
    test_suite.addTest(unittest.makeSuite(TestModelPerformance))
    test_suite.addTest(unittest.makeSuite(TestFeatureImportance))
    test_suite.addTest(unittest.makeSuite(TestDataConsistency))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    }")
    }")
    - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        for test, traceback in result.failures:
            if result.errors:
        for test, traceback in result.errors:
