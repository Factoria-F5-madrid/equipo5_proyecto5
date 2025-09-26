import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import LifeExpectancyPipeline

class TestDataValidation(unittest.TestCase):
    """Test data validation edge cases and boundary conditions"""

    def setUp(self):
        """Set up test fixtures"""
        with unittest.mock.patch('joblib.load'):
            self.pipeline = LifeExpectancyPipeline()

    def test_boundary_values(self):
        """Test validation at boundary values"""
        min_valid_data = {
            'country': 'Spain',
            'year': 2000,
            'status': 'Developed',
            'adult_mortality': 0.0,
            'infant_deaths': 0,
            'alcohol': 0.0,
            'percentage_expenditure': 0.0,
            'hepatitis_b': 0.0,
            'measles': 0,
            'bmi': 10.0,
            'under_five_deaths': 0,
            'polio': 0.0,
            'total_expenditure': 0.0,
            'diphtheria': 0.0,
            'hiv/aids': 0.0,
            'gdp': 0.0,
            'population': 1000,
            'thinness__1_19_years': 0.0,
            'thinness_5_9_years': 0.0,
            'income_composition_of_resources': 0.0,
            'schooling': 0.0
        }

        is_valid, message = self.pipeline.validate_data(min_valid_data)
        self.assertTrue(is_valid, f"Min valid data should pass: {message}")
        max_valid_data = {
            'country': 'Spain',
            'year': 2025,
            'status': 'Developed',
            'adult_mortality': 1000.0,
            'infant_deaths': 10000,
            'alcohol': 20.0,
            'percentage_expenditure': 1000.0,
            'hepatitis_b': 100.0,
            'measles': 100000,
            'bmi': 50.0,
            'under_five_deaths': 10000,
            'polio': 100.0,
            'total_expenditure': 20.0,
            'diphtheria': 100.0,
            'hiv/aids': 50.0,
            'gdp': 100000.0,
            'population': 2000000000,
            'thinness__1_19_years': 50.0,
            'thinness_5_9_years': 50.0,
            'income_composition_of_resources': 1.0,
            'schooling': 20.0
        }

        is_valid, message = self.pipeline.validate_data(max_valid_data)
        self.assertTrue(is_valid, f"Max valid data should pass: {message}")

    def test_boundary_violations(self):
        """Test validation with boundary violations"""
        boundary_violations = [
            ('adult_mortality', -0.1, "Negative adult mortality"),
            ('adult_mortality', 1000.1, "Adult mortality too high"),
            ('gdp', -1.0, "Negative GDP"),
            ('gdp', 100001.0, "GDP too high"),
            ('population', 999, "Population too low"),
            ('population', 2000000001, "Population too high"),
            ('schooling', -0.1, "Negative schooling"),
            ('schooling', 20.1, "Schooling too high"),
            ('bmi', 9.9, "BMI too low"),
            ('bmi', 50.1, "BMI too high"),
            ('income_composition_of_resources', -0.1, "Negative income composition"),
            ('income_composition_of_resources', 1.1, "Income composition too high")
        ]

        for field, value, description in boundary_violations:
            with self.subTest(field=field, value=value):
                test_data = {
                    'country': 'Spain',
                    'year': 2020,
                    'status': 'Developed',
                    field: value
                }

                is_valid, message = self.pipeline.validate_data(test_data)
                self.assertFalse(is_valid, f"{description} should fail validation")
                self.assertIn(field, str(message), f"Error message should mention {field}")

    def test_data_type_validation(self):
        """Test validation with wrong data types"""
        type_violations = [
            ('adult_mortality', "not_a_number", "String instead of number"),
            ('gdp', [1, 2, 3], "List instead of number"),
            ('population', None, "None instead of number"),
            ('schooling', True, "Boolean instead of number")
        ]

        for field, value, description in type_violations:
            with self.subTest(field=field, value=value):
                test_data = {
                    'country': 'Spain',
                    'year': 2020,
                    'status': 'Developed',
                    field: value
                }

                is_valid, message = self.pipeline.validate_data(test_data)
                self.assertFalse(is_valid, f"{description} should fail validation")
                self.assertIn("Must be a number", str(message), f"Error message should mention data type")

    def test_country_validation(self):
        """Test country validation"""
        valid_countries = ['Spain', 'France', 'Germany', 'United States of America']
        for country in valid_countries:
            test_data = {
                'country': country,
                'year': 2020,
                'status': 'Developed',
                'adult_mortality': 50.0
            }

            is_valid, message = self.pipeline.validate_data(test_data)
            self.assertTrue(is_valid, f"Valid country '{country}' should pass")
        invalid_countries = ['InvalidCountry', '', '123', None]
        for country in invalid_countries:
            test_data = {
                'country': country,
                'year': 2020,
                'status': 'Developed',
                'adult_mortality': 50.0
            }

            is_valid, message = self.pipeline.validate_data(test_data)
            self.assertFalse(is_valid, f"Invalid country '{country}' should fail")

    def test_status_validation(self):
        """Test status validation"""
        valid_statuses = ['Developed', 'Developing']
        for status in valid_statuses:
            test_data = {
                'country': 'Spain',
                'year': 2020,
                'status': status,
                'adult_mortality': 50.0
            }

            is_valid, message = self.pipeline.validate_data(test_data)
            self.assertTrue(is_valid, f"Valid status '{status}' should pass")
        invalid_statuses = ['InvalidStatus', 'developed', 'DEVELOPED', '', None]
        for status in invalid_statuses:
            test_data = {
                'country': 'Spain',
                'year': 2020,
                'status': status,
                'adult_mortality': 50.0
            }

            is_valid, message = self.pipeline.validate_data(test_data)
            self.assertFalse(is_valid, f"Invalid status '{status}' should fail")

    def test_year_validation(self):
        """Test year validation"""
        valid_years = [2000, 2010, 2020, 2025]
        for year in valid_years:
            test_data = {
                'country': 'Spain',
                'year': year,
                'status': 'Developed',
                'adult_mortality': 50.0
            }

            is_valid, message = self.pipeline.validate_data(test_data)
            self.assertTrue(is_valid, f"Valid year {year} should pass")
        invalid_years = [1999, 2026, 0, -1, "2020", 2020.5]
        for year in invalid_years:
            test_data = {
                'country': 'Spain',
                'year': year,
                'status': 'Developed',
                'adult_mortality': 50.0
            }

            is_valid, message = self.pipeline.validate_data(test_data)
            self.assertFalse(is_valid, f"Invalid year {year} should fail")

    def test_multiple_validation_errors(self):
        """Test validation with multiple errors"""
        invalid_data = {
            'country': 'InvalidCountry',
            'year': 1990,
            'status': 'InvalidStatus',
            'adult_mortality': -50.0,
            'gdp': -1000.0,
            'population': 500  
        }

        is_valid, message = self.pipeline.validate_data(invalid_data)
        self.assertFalse(is_valid)
        self.assertIsInstance(message, list)
        self.assertGreater(len(message), 1, "Should have multiple validation errors")
        error_text = ' '.join(message)
        self.assertIn('Country', error_text)
        self.assertIn('Year', error_text)
        self.assertIn('Status', error_text)
        self.assertIn('adult_mortality', error_text)
        self.assertIn('gdp', error_text)
        self.assertIn('population', error_text)

    def test_partial_data_validation(self):
        """Test validation with partial data (only some fields)"""
        minimal_data = {
            'country': 'Spain',
            'year': 2020,
            'status': 'Developed'
        }

        is_valid, message = self.pipeline.validate_data(minimal_data)
        self.assertTrue(is_valid, f"Minimal valid data should pass: {message}")
        empty_data = {}

        is_valid, message = self.pipeline.validate_data(empty_data)
        self.assertTrue(is_valid, f"Empty data should pass (no validation errors)")

class TestDataQuality(unittest.TestCase):
    """Test data quality and consistency checks"""

    def setUp(self):
        """Set up test fixtures"""
        with unittest.mock.patch('joblib.load'):
            self.pipeline = LifeExpectancyPipeline()

    def test_valid_ranges_completeness(self):
        """Test that all expected variables have valid ranges defined"""
        valid_ranges = self.pipeline._get_valid_ranges()
        expected_variables = [
            'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure',
            'hepatitis_b', 'measles', 'bmi', 'under_five_deaths', 'polio',
            'total_expenditure', 'diphtheria', 'hiv/aids', 'gdp', 'population',
            'thinness__1_19_years', 'thinness_5_9_years', 'income_composition_of_resources',
            'schooling'
        ]

        for var in expected_variables:
            self.assertIn(var, valid_ranges, f"Missing valid range for {var}")
            range_val = valid_ranges[var]
            self.assertIsInstance(range_val, tuple, f"Range for {var} should be tuple")
            self.assertEqual(len(range_val), 2, f"Range for {var} should have 2 elements")
            min_val, max_val = range_val
            self.assertLessEqual(min_val, max_val, f"Min should be <= max for {var}")

    def test_valid_ranges_reasonable(self):
        """Test that valid ranges are reasonable"""
        valid_ranges = self.pipeline._get_valid_ranges()
        self.assertEqual(valid_ranges['adult_mortality'], (0, 1000))
        self.assertEqual(valid_ranges['gdp'], (0, 100000))
        self.assertEqual(valid_ranges['population'], (1000, 2000000000))
        self.assertEqual(valid_ranges['schooling'], (0, 20))
        self.assertEqual(valid_ranges['bmi'], (10, 50))
        self.assertEqual(valid_ranges['income_composition_of_resources'], (0, 1))

    def test_country_list_completeness(self):
        """Test that country list is not empty and has reasonable countries"""
        countries = self.pipeline.valid_countries

        self.assertGreater(len(countries), 0, "Country list should not be empty")
        self.assertIsInstance(countries, list, "Countries should be a list")
        expected_countries = ['Spain', 'France', 'Germany', 'United States of America', 'China', 'India']
        for country in expected_countries:
            self.assertIn(country, countries, f"Expected country {country} not in list")

    def test_status_list_correctness(self):
        """Test that status list is correct"""
        statuses = self.pipeline.valid_status

        self.assertEqual(set(statuses), {'Developed', 'Developing'},
                        f"Status list should be exactly ['Developed', 'Developing'], got {statuses}")

if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestDataValidation))
    test_suite.addTest(unittest.makeSuite(TestDataQuality))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    }")
    }")
    - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        for test, traceback in result.failures:
            if result.errors:
        for test, traceback in result.errors:
