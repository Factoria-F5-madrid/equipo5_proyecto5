import unittest
import pandas as pd
import numpy as np
import json
import os
import sys
from unittest.mock import patch, MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import LifeExpectancyPipeline

class TestLifeExpectancyPipeline(unittest.TestCase):
    """Test cases for LifeExpectancyPipeline class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        with patch('joblib.load') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([75.5])
            mock_load.return_value = mock_model

            self.pipeline = LifeExpectancyPipeline()

    def test_valid_ranges_initialization(self):
        """Test that valid ranges are properly initialized"""
        valid_ranges = self.pipeline._get_valid_ranges()
        self.assertEqual(valid_ranges['adult_mortality'], (0, 1000))
        self.assertEqual(valid_ranges['gdp'], (0, 100000))
        self.assertEqual(valid_ranges['population'], (1000, 2000000000))
        self.assertEqual(valid_ranges['schooling'], (0, 20))
        expected_vars = [
            'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure',
            'hepatitis_b', 'measles', 'bmi', 'under_five_deaths', 'polio',
            'total_expenditure', 'diphtheria', 'hiv/aids', 'gdp', 'population',
            'thinness__1_19_years', 'thinness_5_9_years', 'income_composition_of_resources',
            'schooling'
        ]

        for var in expected_vars:
            self.assertIn(var, valid_ranges)

    def test_validate_data_valid_input(self):
        """Test validation with valid input data"""
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

        is_valid, message = self.pipeline.validate_data(valid_data)

        self.assertTrue(is_valid)
        self.assertEqual(message, "Data is valid")

    def test_validate_data_invalid_ranges(self):
        """Test validation with invalid data ranges"""
        invalid_data = {
            'country': 'Spain',
            'year': 2020,
            'status': 'Developed',
            'adult_mortality': -50.0, 
            'infant_deaths': 100,
            'alcohol': 50.0, 
            'gdp': 30000.0,
            'population': 50000000
        }

        is_valid, message = self.pipeline.validate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIsInstance(message, list)
        self.assertGreater(len(message), 0)

    def test_validate_data_invalid_country(self):
        """Test validation with invalid country"""
        invalid_data = {
            'country': 'InvalidCountry',
            'year': 2020,
            'status': 'Developed',
            'adult_mortality': 50.0,
            'gdp': 30000.0
        }

        is_valid, message = self.pipeline.validate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("Country", str(message))

    def test_validate_data_invalid_status(self):
        """Test validation with invalid status"""
        invalid_data = {
            'country': 'Spain',
            'year': 2020,
            'status': 'InvalidStatus',
            'adult_mortality': 50.0,
            'gdp': 30000.0
        }

        is_valid, message = self.pipeline.validate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("Status", str(message))

    def test_validate_data_invalid_year(self):
        """Test validation with invalid year"""
        invalid_data = {
            'country': 'Spain',
            'year': 1990,  
            'status': 'Developed',
            'adult_mortality': 50.0,
            'gdp': 30000.0
        }

        is_valid, message = self.pipeline.validate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("Year", str(message))

    @patch('pandas.read_csv')
    @patch('joblib.dump')
    def test_create_preprocessor(self, mock_dump, mock_read_csv):
        """Test preprocessor creation"""
        mock_df = pd.DataFrame({
            'country': ['Spain', 'France'],
            'year': [2020, 2020],
            'status': ['Developed', 'Developed'],
            'life_expectancy': [80.0, 82.0],
            'adult_mortality': [50.0, 45.0],
            'gdp': [30000.0, 35000.0],
            'schooling': [12.0, 13.0]
        })
        mock_read_csv.return_value = mock_df

        preprocessor = self.pipeline.create_preprocessor()

        self.assertIsNotNone(preprocessor)
        mock_dump.assert_called_once()

    @patch('pandas.DataFrame')
    def test_transform_data(self, mock_dataframe):
        """Test data transformation"""
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[1.0, 2.0, 3.0]])
        self.pipeline.preprocessor = mock_preprocessor

        user_data = {
            'country': 'Spain',
            'adult_mortality': 50.0,
            'gdp': 30000.0
        }

        result = self.pipeline.transform_data(user_data)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 3))
        mock_preprocessor.transform.assert_called_once()

    def test_predict(self):
        """Test prediction functionality"""
        transformed_data = np.array([[1.0, 2.0, 3.0]])

        prediction = self.pipeline.predict(transformed_data)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction, 75.5)  
        self.assertIsInstance(prediction, float)

    def test_predict_no_model(self):
        """Test prediction when model is not loaded"""
        self.pipeline.model = None

        prediction = self.pipeline.predict(np.array([[1.0, 2.0, 3.0]]))

        self.assertIsNone(prediction)

class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        with patch('joblib.load') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([75.5])
            mock_load.return_value = mock_model

            self.pipeline = LifeExpectancyPipeline()

    @patch('pandas.read_csv')
    @patch('joblib.dump')
    def test_complete_pipeline_flow(self, mock_dump, mock_read_csv):
        """Test the complete pipeline flow from validation to prediction"""
        mock_df = pd.DataFrame({
            'country': ['Spain', 'France'],
            'year': [2020, 2020],
            'status': ['Developed', 'Developed'],
            'life_expectancy': [80.0, 82.0],
            'adult_mortality': [50.0, 45.0],
            'gdp': [30000.0, 35000.0],
            'schooling': [12.0, 13.0]
        })
        mock_read_csv.return_value = mock_df
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[1.0, 2.0, 3.0]])

        with patch.object(self.pipeline, 'create_preprocessor', return_value=mock_preprocessor):
            test_data = {
                'country': 'Spain',
                'year': 2020,
                'status': 'Developed',
                'adult_mortality': 50.0,
                'gdp': 30000.0,
                'schooling': 12.0
            }
            is_valid, message = self.pipeline.validate_data(test_data)
            self.assertTrue(is_valid)
            transformed_data = self.pipeline.transform_data(test_data)
            self.assertIsNotNone(transformed_data)
            prediction = self.pipeline.predict(transformed_data)
            self.assertIsNotNone(prediction)
            self.assertEqual(prediction, 75.5)

if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestLifeExpectancyPipeline))
    test_suite.addTest(unittest.makeSuite(TestPipelineIntegration))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    }")
    }")
    - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        for test, traceback in result.failures:
            if result.errors:
        for test, traceback in result.errors:
