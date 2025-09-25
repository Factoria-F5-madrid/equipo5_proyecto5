"""
Tests for A/B Testing System
Tests model comparison, traffic splitting, and statistical analysis
"""

import unittest
import pandas as pd
import numpy as np
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ab_testing import ABTestingSystem


class TestABTestingSystem(unittest.TestCase):
    """Test cases for A/B Testing System"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the data loading to avoid file dependencies
        with patch('pandas.read_csv') as mock_read_csv:
            # Create mock data
            mock_data = pd.DataFrame({
                'country': ['Spain', 'France', 'Germany'] * 100,
                'year': [2020, 2021, 2022] * 100,
                'status': ['Developed', 'Developing'] * 150,
                'life_expectancy': np.random.normal(75, 10, 300),
                'adult_mortality': np.random.normal(100, 50, 300),
                'gdp': np.random.normal(20000, 10000, 300),
                'schooling': np.random.normal(10, 3, 300),
                'bmi': np.random.normal(25, 5, 300),
                'hiv/aids': np.random.normal(2, 1, 300),
                'alcohol': np.random.normal(5, 2, 300),
                'hepatitis_b': np.random.normal(80, 20, 300),
                'measles': np.random.normal(1000, 500, 300),
                'polio': np.random.normal(85, 15, 300),
                'total_expenditure': np.random.normal(8, 3, 300),
                'diphtheria': np.random.normal(80, 20, 300),
                'infant_deaths': np.random.normal(1000, 500, 300),
                'percentage_expenditure': np.random.normal(200, 100, 300),
                'under_five_deaths': np.random.normal(1500, 800, 300),
                'population': np.random.normal(50000000, 20000000, 300),
                'thinness__1_19_years': np.random.normal(10, 5, 300),
                'thinness_5_9_years': np.random.normal(8, 4, 300),
                'income_composition_of_resources': np.random.normal(0.7, 0.2, 300)
            })
            mock_read_csv.return_value = mock_data
            
            self.ab_system = ABTestingSystem()
    
    def test_initialization(self):
        """Test A/B testing system initialization"""
        self.assertIsNotNone(self.ab_system.models)
        self.assertIsNotNone(self.ab_system.results)
        self.assertEqual(len(self.ab_system.models), 3)
        self.assertIn('RandomForest', self.ab_system.models)
        self.assertIn('GradientBoosting', self.ab_system.models)
        self.assertIn('LinearRegression', self.ab_system.models)
    
    def test_data_preparation(self):
        """Test data preparation for A/B testing"""
        self.assertIsNotNone(self.ab_system.X)
        self.assertIsNotNone(self.ab_system.y)
        self.assertGreater(len(self.ab_system.X), 0)
        self.assertGreater(len(self.ab_system.y), 0)
        self.assertEqual(len(self.ab_system.X), len(self.ab_system.y))
    
    def test_model_training(self):
        """Test model training functionality"""
        # Train models
        self.ab_system.train_models(test_size=0.2)
        
        # Check that all models were trained
        self.assertEqual(len(self.ab_system.results), 3)
        
        # Check that each model has required metrics
        for model_name, results in self.ab_system.results.items():
            self.assertIn('rmse', results)
            self.assertIn('mae', results)
            self.assertIn('r2', results)
            self.assertIn('training_time', results)
            self.assertIn('predictions', results)
            self.assertIn('test_actual', results)
            
            # Check metric values are reasonable
            self.assertGreaterEqual(results['r2'], -1)  # R² can be negative for bad models
            self.assertLessEqual(results['r2'], 1)
            self.assertGreater(results['rmse'], 0)
            self.assertGreater(results['mae'], 0)
            self.assertGreater(results['training_time'], 0)
    
    def test_ab_test_execution(self):
        """Test A/B test execution"""
        # Train models first
        self.ab_system.train_models()
        
        # Run A/B test
        ab_results = self.ab_system.run_ab_test(traffic_split=0.5, duration_days=7)
        
        # Check A/B test results structure
        self.assertIn('model_a', ab_results)
        self.assertIn('model_b', ab_results)
        self.assertIn('combined', ab_results)
        self.assertIn('traffic_split', ab_results)
        self.assertIn('duration_days', ab_results)
        self.assertIn('total_samples', ab_results)
        
        # Check model A and B have required fields
        for model_key in ['model_a', 'model_b']:
            model_data = ab_results[model_key]
            self.assertIn('name', model_data)
            self.assertIn('samples', model_data)
            self.assertIn('rmse', model_data)
            self.assertIn('mae', model_data)
            self.assertIn('r2', model_data)
            
            # Check values are reasonable
            self.assertGreater(model_data['samples'], 0)
            self.assertGreaterEqual(model_data['r2'], -1)  # R² can be negative for bad models
            self.assertLessEqual(model_data['r2'], 1)
            self.assertGreater(model_data['rmse'], 0)
            self.assertGreater(model_data['mae'], 0)
        
        # Check traffic split
        self.assertEqual(ab_results['traffic_split'], 0.5)
        self.assertEqual(ab_results['duration_days'], 7)
    
    def test_results_analysis(self):
        """Test A/B test results analysis"""
        # Train models and run A/B test
        self.ab_system.train_models()
        self.ab_system.run_ab_test()
        
        # Analyze results
        analysis = self.ab_system.analyze_results()
        
        # Check analysis results structure
        self.assertIn('winner', analysis)
        self.assertIn('improvement', analysis)
        self.assertIn('significance', analysis)
        self.assertIn('recommendation', analysis)
        
        # Check winner is valid
        self.assertIn(analysis['winner'], ['A', 'B'])
        
        # Check improvement is a number
        self.assertIsInstance(analysis['improvement'], (int, float))
        self.assertGreaterEqual(analysis['improvement'], 0)
        
        # Check significance is boolean
        self.assertIsInstance(analysis['significance'], bool)
        
        # Check recommendation is a string
        self.assertIsInstance(analysis['recommendation'], str)
        self.assertGreater(len(analysis['recommendation']), 0)
    
    def test_traffic_split_variations(self):
        """Test different traffic split configurations"""
        # Train models first
        self.ab_system.train_models()
        
        # Test different traffic splits
        traffic_splits = [0.3, 0.5, 0.7]
        
        for split in traffic_splits:
            with self.subTest(split=split):
                ab_results = self.ab_system.run_ab_test(traffic_split=split)
                
                # Check that traffic split is correctly applied
                self.assertEqual(ab_results['traffic_split'], split)
                
                # Check that samples are distributed correctly
                model_a_samples = ab_results['model_a']['samples']
                model_b_samples = ab_results['model_b']['samples']
                total_samples = ab_results['total_samples']
                
                expected_a_samples = int(total_samples * split)
                expected_b_samples = total_samples - expected_a_samples
                
                self.assertEqual(model_a_samples, expected_a_samples)
                self.assertEqual(model_b_samples, expected_b_samples)
    
    def test_model_performance_comparison(self):
        """Test that different models have different performance"""
        # Train models
        self.ab_system.train_models()
        
        # Check that models have different performance
        model_names = list(self.ab_system.results.keys())
        performances = [self.ab_system.results[name]['rmse'] for name in model_names]
        
        # At least one model should be different (not all identical)
        self.assertNotEqual(len(set(performances)), 1, "All models have identical performance")
        
        # Check that models have different performance (not all identical)
        lr_rmse = self.ab_system.results['LinearRegression']['rmse']
        rf_rmse = self.ab_system.results['RandomForest']['rmse']
        gb_rmse = self.ab_system.results['GradientBoosting']['rmse']
        
        # At least one model should be different (not all identical)
        self.assertNotEqual(lr_rmse, rf_rmse, "Linear regression and Random Forest should have different performance")
        self.assertNotEqual(lr_rmse, gb_rmse, "Linear regression and Gradient Boosting should have different performance")
        self.assertNotEqual(rf_rmse, gb_rmse, "Random Forest and Gradient Boosting should have different performance")
    
    def test_results_saving(self):
        """Test saving A/B test results"""
        # Train models and run A/B test
        self.ab_system.train_models()
        self.ab_system.run_ab_test()
        self.ab_system.analyze_results()
        
        # Save results
        filename = 'test_ab_results.json'
        self.ab_system.save_results(filename)
        
        # Check file was created
        self.assertTrue(os.path.exists(filename))
        
        # Check file content
        with open(filename, 'r') as f:
            saved_data = json.load(f)
        
        self.assertIn('timestamp', saved_data)
        self.assertIn('ab_results', saved_data)
        self.assertIn('analysis_results', saved_data)
        
        # Clean up
        os.remove(filename)
    
    def test_visualization_creation(self):
        """Test visualization creation"""
        # Train models and run A/B test
        self.ab_system.train_models()
        self.ab_system.run_ab_test()
        
        # Create visualizations
        self.ab_system.create_visualizations()
        
        # Check that plot files were created
        expected_plots = [
            'plots/ab_test_model_comparison.png',
            'plots/ab_test_predictions_comparison.png',
            'plots/ab_test_error_distribution.png'
        ]
        
        for plot_file in expected_plots:
            self.assertTrue(os.path.exists(plot_file), f"Plot file {plot_file} not created")
    
    def test_statistical_significance_calculation(self):
        """Test statistical significance calculation"""
        # Train models and run A/B test
        self.ab_system.train_models()
        self.ab_system.run_ab_test()
        
        # Analyze results
        analysis = self.ab_system.analyze_results()
        
        # Check that significance is calculated
        self.assertIsInstance(analysis['significance'], bool)
        
        # Check that recommendation is based on significance
        if analysis['significance']:
            self.assertIn('significant', analysis['recommendation'].lower())
        else:
            self.assertIn('not statistically significant', analysis['recommendation'].lower())


class TestABTestingIntegration(unittest.TestCase):
    """Integration tests for A/B Testing System"""
    
    def test_complete_ab_testing_workflow(self):
        """Test complete A/B testing workflow from start to finish"""
        with patch('pandas.read_csv') as mock_read_csv:
            # Create mock data
            mock_data = pd.DataFrame({
                'country': ['Spain', 'France'] * 50,
                'year': [2020, 2021] * 50,
                'status': ['Developed', 'Developing'] * 50,
                'life_expectancy': np.random.normal(75, 10, 100),
                'adult_mortality': np.random.normal(100, 50, 100),
                'gdp': np.random.normal(20000, 10000, 100),
                'schooling': np.random.normal(10, 3, 100),
                'bmi': np.random.normal(25, 5, 100),
                'hiv/aids': np.random.normal(2, 1, 100),
                'alcohol': np.random.normal(5, 2, 100),
                'hepatitis_b': np.random.normal(80, 20, 100),
                'measles': np.random.normal(1000, 500, 100),
                'polio': np.random.normal(85, 15, 100),
                'total_expenditure': np.random.normal(8, 3, 100),
                'diphtheria': np.random.normal(80, 20, 100),
                'infant_deaths': np.random.normal(1000, 500, 100),
                'percentage_expenditure': np.random.normal(200, 100, 100),
                'under_five_deaths': np.random.normal(1500, 800, 100),
                'population': np.random.normal(50000000, 20000000, 100),
                'thinness__1_19_years': np.random.normal(10, 5, 100),
                'thinness_5_9_years': np.random.normal(8, 4, 100),
                'income_composition_of_resources': np.random.normal(0.7, 0.2, 100)
            })
            mock_read_csv.return_value = mock_data
            
            # Initialize system
            ab_system = ABTestingSystem()
            
            # Train models
            ab_system.train_models()
            
            # Run A/B test
            ab_results = ab_system.run_ab_test()
            
            # Analyze results
            analysis = ab_system.analyze_results()
            
            # Create visualizations
            ab_system.create_visualizations()
            
            # Save results
            ab_system.save_results('test_complete_workflow.json')
            
            # Verify everything worked
            self.assertIsNotNone(ab_results)
            self.assertIsNotNone(analysis)
            self.assertTrue(os.path.exists('test_complete_workflow.json'))
            
            # Clean up
            os.remove('test_complete_workflow.json')


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestABTestingSystem))
    test_suite.addTest(unittest.makeSuite(TestABTestingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"A/B TESTING TESTS SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
