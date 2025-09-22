"""
Test runner for all tests in the project
Runs all test suites and provides comprehensive reporting
"""

import unittest
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_pipeline import TestLifeExpectancyPipeline, TestPipelineIntegration
from test_model import (TestModelLoading, TestModelPerformance, 
                       TestFeatureImportance, TestDataConsistency)
from test_data_validation import TestDataValidation, TestDataQuality


def run_all_tests():
    """Run all tests and provide comprehensive reporting"""
    
    print("🧪 COMPREHENSIVE TEST SUITE FOR LIFE EXPECTANCY ML PIPELINE")
    print("=" * 70)
    print(f"Test execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        # Pipeline tests
        TestLifeExpectancyPipeline,
        TestPipelineIntegration,
        
        # Model tests
        TestModelLoading,
        TestModelPerformance,
        TestFeatureImportance,
        TestDataConsistency,
        
        # Data validation tests
        TestDataValidation,
        TestDataQuality
    ]
    
    # Add all test classes to suite
    for test_class in test_classes:
        tests = unittest.makeSuite(test_class)
        test_suite.addTest(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        descriptions=True,
        failfast=False
    )
    
    print(f"\nRunning {test_suite.countTestCases()} individual tests...")
    print("-" * 70)
    
    result = runner.run(test_suite)
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 70)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"📈 Total Tests Run: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"🚫 Errors: {errors}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"📊 Success Rate: {(passed / total_tests * 100):.1f}%")
    
    # Test coverage by category
    print(f"\n📋 TEST COVERAGE BY CATEGORY:")
    print(f"   🔧 Pipeline Tests: {len([t for t in test_classes[:2]])} classes")
    print(f"   🤖 Model Tests: {len([t for t in test_classes[2:6]])} classes")
    print(f"   ✅ Validation Tests: {len([t for t in test_classes[6:8]])} classes")
    
    # Detailed failure/error reporting
    if failures > 0:
        print(f"\n❌ FAILURES ({failures}):")
        print("-" * 50)
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"{i}. {test}")
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"   Error: {error_msg}")
            print()
    
    if errors > 0:
        print(f"\n🚫 ERRORS ({errors}):")
        print("-" * 50)
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"{i}. {test}")
            error_msg = traceback.split('\n')[-2]
            print(f"   Error: {error_msg}")
            print()
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if failures == 0 and errors == 0:
        print("   🎉 All tests passed! Your pipeline is working correctly.")
        print("   🚀 Ready for production deployment.")
    else:
        print("   🔧 Fix failing tests before deploying to production.")
        print("   📝 Review error messages for debugging guidance.")
        print("   🧪 Consider adding more edge case tests.")
    
    # Performance metrics
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   🕐 Test execution time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"   📊 Tests per second: {total_tests / 1:.1f}")  # Approximate
    
    print("\n" + "=" * 70)
    print("🏁 TEST EXECUTION COMPLETED")
    print("=" * 70)
    
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
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return None
    
    print(f"🧪 Running {category.upper()} tests...")
    print("-" * 50)
    
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
