"""
Test runner for all tests in the project
Runs all test suites and provides comprehensive reporting
"""

import unittest
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_pipeline import TestLifeExpectancyPipeline, TestPipelineIntegration
from test_model import (TestModelLoading, TestModelPerformance,
                       TestFeatureImportance, TestDataConsistency)
from test_data_validation import TestDataValidation, TestDataQuality
from test_ab_testing import TestABTestingSystem, TestABTestingIntegration

def run_all_tests():
    """Run all tests and provide comprehensive reporting"""

    .strftime('%Y-%m-%d %H:%M:%S')}")
    test_suite = unittest.TestSuite()
    test_classes = [
        TestLifeExpectancyPipeline,
        TestPipelineIntegration,
        TestModelLoading,
        TestModelPerformance,
        TestFeatureImportance,
        TestDataConsistency,
        TestDataValidation,
        TestDataQuality,
        TestABTestingSystem,
        TestABTestingIntegration
    ]
    for test_class in test_classes:
        tests = unittest.makeSuite(test_class)
        test_suite.addTest(tests)
    runner = unittest.TextTestRunner(
        verbosity=2,
        descriptions=True,
        failfast=False
    )

    } individual tests...")
    result = runner.run(test_suite)
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped

    :.1f}%")
    } classes")
    } classes")
    } classes")
    } classes")
    if failures > 0:
        :")
        for i, (test, traceback) in enumerate(result.failures, 1):
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            if errors > 0:
        :")
        for i, (test, traceback) in enumerate(result.errors, 1):
            error_msg = traceback.split('\n')[-2]
    if failures == 0 and errors == 0:
        else:
    .strftime('%H:%M:%S')}")

    return result

def run_specific_test_category(category):
    """Run tests for a specific category"""

    categories = {
        'pipeline': [TestLifeExpectancyPipeline, TestPipelineIntegration],
        'model': [TestModelLoading, TestModelPerformance, TestFeatureImportance, TestDataConsistency],
        'validation': [TestDataValidation, TestDataQuality],
        'all': [TestLifeExpectancyPipeline, TestPipelineIntegration, TestModelLoading,
                TestModelPerformance, TestFeatureImportance, TestDataConsistency,
                TestDataValidation, TestDataQuality]
    }

    if category not in categories:
        )}")
        return None

    } tests...")
    test_suite = unittest.TestSuite()
    for test_class in categories[category]:
        tests = unittest.makeSuite(test_class)
        test_suite.addTest(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run tests for Life Expectancy ML Pipeline')
    parser.add_argument('--category', '-c',
                       choices=['pipeline', 'model', 'validation', 'all'],
                       default='all',
                       help='Test category to run (default: all)')

    args = parser.parse_args()

    if args.category == 'all':
        run_all_tests()
    else:
        run_specific_test_category(args.category)
