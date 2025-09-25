#!/usr/bin/env python3
"""
Test script for PostgreSQL database integration
Tests all database operations for the MLOps system
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_manager import DatabaseConnection

def test_database_connection():
    """Test basic database connection"""
    print("🔌 Testing database connection...")
    
    try:
        with DatabaseConnection() as db:
            if db.health_check():
                print("✅ Database connection successful")
                return True
            else:
                print("❌ Database connection failed")
                return False
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return False

def test_prediction_operations():
    """Test prediction save and retrieve operations"""
    print("\n📊 Testing prediction operations...")
    
    try:
        with DatabaseConnection() as db:
            # Sample prediction data
            sample_input = {
                'adult_mortality': 50.0,
                'infant_deaths': 100,
                'alcohol': 10.5,
                'gdp': 30000.0,
                'schooling': 12.0,
                'country': 'Spain',
                'status': 'Developed',
                'year': 2020
            }
            
            # Save prediction
            prediction_id = db.save_prediction(
                country='Spain',
                year=2020,
                status='Developed',
                predicted_life_expectancy=80.5,
                model_version='v1.0.0',
                input_data=sample_input,
                confidence_score=0.95,
                processing_time_ms=150,
                user_session_id='test_session_123'
            )
            
            print(f"✅ Prediction saved with ID: {prediction_id}")
            
            # Retrieve predictions
            predictions = db.get_predictions(limit=5)
            print(f"✅ Retrieved {len(predictions)} predictions")
            
            # Get prediction stats
            stats = db.get_prediction_stats()
            print(f"✅ Prediction stats: {stats}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error in prediction operations: {e}")
        return False

def test_model_version_operations():
    """Test model version save and retrieve operations"""
    print("\n🤖 Testing model version operations...")
    
    try:
        with DatabaseConnection() as db:
            # Sample model data
            hyperparameters = {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
            
            performance_metrics = {
                'rmse': 1.649,
                'mae': 1.074,
                'r2': 0.969,
                'overfitting': 0.026
            }
            
            # Save model version
            model_id = db.save_model_version(
                model_name='RandomForest',
                version='v1.0.0',
                algorithm='RandomForestRegressor',
                hyperparameters=hyperparameters,
                performance_metrics=performance_metrics,
                training_data_hash='abc123def456',
                file_path='models/best_life_expectancy_model.pkl',
                training_duration_seconds=300,
                training_samples=2000,
                validation_samples=500
            )
            
            print(f"✅ Model version saved with ID: {model_id}")
            
            # Save feature importance
            feature_importance_df = pd.DataFrame({
                'feature': ['hiv/aids', 'adult_mortality', 'income_composition', 'schooling', 'gdp'],
                'importance': [0.594, 0.156, 0.148, 0.020, 0.012]
            })
            
            db.save_feature_importance(model_id, feature_importance_df)
            print("✅ Feature importance saved")
            
            # Get active model
            active_model = db.get_active_model()
            if active_model:
                print(f"✅ Active model: {active_model['model_name']} v{active_model['version']}")
            else:
                print("⚠️ No active model found")
            
            return True
            
    except Exception as e:
        print(f"❌ Error in model version operations: {e}")
        return False

def test_drift_monitoring_operations():
    """Test data drift monitoring operations"""
    print("\n🔍 Testing drift monitoring operations...")
    
    try:
        with DatabaseConnection() as db:
            # Sample drift results
            drift_results = {
                'adult_mortality': {
                    'drift_detected': False,
                    'drift_score': 0.05,
                    'ks_statistic': 0.1,
                    'ks_p_value': 0.8,
                    'reference_mean': 164.8,
                    'current_mean': 165.2,
                    'reference_std': 124.1,
                    'current_std': 125.0,
                    'reference_median': 144.0,
                    'current_median': 145.0,
                    'alert_level': 'green'
                },
                'gdp': {
                    'drift_detected': True,
                    'drift_score': 0.15,
                    'ks_statistic': 0.2,
                    'ks_p_value': 0.01,
                    'reference_mean': 7483.2,
                    'current_mean': 8200.0,
                    'reference_std': 13136.8,
                    'current_std': 14000.0,
                    'reference_median': 3116.6,
                    'current_median': 3500.0,
                    'alert_level': 'yellow'
                },
                'overall_drift_detected': True,
                'drift_threshold': 0.1
            }
            
            # Save drift check
            monitoring_id = db.save_drift_check(drift_results)
            print(f"✅ Drift monitoring results saved with ID: {monitoring_id}")
            
            # Get drift summary
            drift_summary = db.get_drift_summary(days=7)
            print(f"✅ Drift summary retrieved: {len(drift_summary)} features")
            
            return True
            
    except Exception as e:
        print(f"❌ Error in drift monitoring operations: {e}")
        return False

def test_ab_testing_operations():
    """Test A/B testing operations"""
    print("\n🧪 Testing A/B testing operations...")
    
    try:
        with DatabaseConnection() as db:
            # Create A/B experiment
            experiment_id = db.create_ab_experiment(
                experiment_name='RandomForest_vs_GradientBoosting',
                model_a_id=1,  # Assuming model ID 1 exists
                model_b_id=2,  # Assuming model ID 2 exists
                traffic_split=0.5,
                start_date=datetime.now(),
                description='Comparing Random Forest vs Gradient Boosting',
                created_by='test_user'
            )
            
            print(f"✅ A/B experiment created with ID: {experiment_id}")
            
            # Save A/B test results
            model_a_metrics = {
                'rmse': 1.649,
                'mae': 1.074,
                'r2': 0.969
            }
            
            model_b_metrics = {
                'rmse': 1.610,
                'mae': 1.066,
                'r2': 0.970
            }
            
            results_id = db.save_ab_test_results(
                experiment_id=experiment_id,
                model_a_metrics=model_a_metrics,
                model_b_metrics=model_b_metrics,
                winner='B',
                improvement_percentage=2.38,
                statistical_significance=False,
                p_value=0.15,
                sample_size=588
            )
            
            print(f"✅ A/B test results saved with ID: {results_id}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error in A/B testing operations: {e}")
        return False

def test_system_logging():
    """Test system logging operations"""
    print("\n📝 Testing system logging operations...")
    
    try:
        with DatabaseConnection() as db:
            # Log system events
            db.log_system_event(
                log_level='INFO',
                component='pipeline',
                message='Model prediction completed successfully',
                details={'prediction_id': 123, 'processing_time': 150},
                user_id='test_user',
                session_id='test_session_123'
            )
            
            db.log_system_event(
                log_level='WARNING',
                component='drift_monitor',
                message='Data drift detected in GDP feature',
                details={'feature': 'gdp', 'drift_score': 0.15},
                user_id='system'
            )
            
            print("✅ System events logged")
            
            # Save system metrics
            db.save_system_metric(
                metric_name='prediction_latency_ms',
                metric_value=150.5,
                unit='milliseconds',
                tags={'environment': 'test', 'model_version': 'v1.0.0'}
            )
            
            db.save_system_metric(
                metric_name='model_accuracy',
                metric_value=0.969,
                unit='r2_score',
                tags={'model': 'RandomForest', 'dataset': 'test'}
            )
            
            print("✅ System metrics saved")
            
            return True
            
    except Exception as e:
        print(f"❌ Error in system logging operations: {e}")
        return False

def main():
    """Run all database tests"""
    print("=" * 60)
    print("🧪 MLOps Database Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Prediction Operations", test_prediction_operations),
        ("Model Version Operations", test_model_version_operations),
        ("Drift Monitoring Operations", test_drift_monitoring_operations),
        ("A/B Testing Operations", test_ab_testing_operations),
        ("System Logging Operations", test_system_logging)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Database integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the database configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
