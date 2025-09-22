"""
A/B Testing System for Life Expectancy ML Models
Compares different models and automatically selects the best performing one
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ABTestingSystem:
    """
    A/B Testing system for comparing ML models
    """
    
    def __init__(self, data_path: str = 'data/clean_data.csv'):
        """
        Initialize A/B Testing system
        
        Args:
            data_path: Path to the clean dataset
        """
        print("🚀 Initializing A/B Testing System...")
        
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.traffic_split = 0.5  # 50/50 split by default
        self.min_samples = 100  # Minimum samples for statistical significance
        
        # Load data
        self.df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        
        # Prepare features and target
        self._prepare_data()
        
        # Initialize models
        self._initialize_models()
        
        print("✅ A/B Testing System initialized successfully")
    
    def _prepare_data(self):
        """Prepare features and target for model training"""
        print("📊 Preparing data for A/B testing...")
        
        # Define target and features
        target = 'life_expectancy'
        exclude_cols = ['country', 'year', 'status', target]
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Separate features and target
        self.X = self.df[feature_cols]
        self.y = self.df[target]
        
        # Handle missing values
        self.X = self.X.fillna(self.X.median())
        
        print(f"✅ Features prepared: {self.X.shape[1]} features")
        print(f"✅ Target prepared: {self.y.shape[0]} samples")
    
    def _initialize_models(self):
        """Initialize different models for A/B testing"""
        print("🤖 Initializing models for A/B testing...")
        
        # Model A: Random Forest (current best)
        self.models['RandomForest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # Model B: Gradient Boosting
        self.models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Model C: Linear Regression (baseline)
        self.models['LinearRegression'] = LinearRegression()
        
        print(f"✅ {len(self.models)} models initialized for A/B testing")
    
    def train_models(self, test_size: float = 0.2):
        """
        Train all models for A/B testing
        
        Args:
            test_size: Proportion of data for testing
        """
        print("🎯 Training models for A/B testing...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        # Train each model
        for name, model in self.models.items():
            print(f"   Training {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'training_time': training_time,
                'predictions': y_pred,
                'test_actual': y_test
            }
            
            print(f"   ✅ {name}: RMSE={rmse:.3f}, R²={r2:.3f}, Time={training_time:.2f}s")
        
        print("✅ All models trained successfully")
    
    def run_ab_test(self, traffic_split: float = 0.5, duration_days: int = 7):
        """
        Run A/B test simulation
        
        Args:
            traffic_split: Proportion of traffic for each model (0.5 = 50/50)
            duration_days: Duration of A/B test in days
        """
        print(f"🧪 Running A/B test simulation...")
        print(f"   Traffic split: {traffic_split*100:.1f}% / {(1-traffic_split)*100:.1f}%")
        print(f"   Duration: {duration_days} days")
        
        self.traffic_split = traffic_split
        
        # Simulate traffic distribution
        n_samples = len(self.X_test)
        n_model_a = int(n_samples * traffic_split)
        n_model_b = n_samples - n_model_a
        
        print(f"   Model A samples: {n_model_a}")
        print(f"   Model B samples: {n_model_b}")
        
        # Get model names
        model_names = list(self.models.keys())
        model_a_name = model_names[0]  # Random Forest
        model_b_name = model_names[1]  # Gradient Boosting
        
        # Simulate predictions for each model
        model_a_pred = self.results[model_a_name]['predictions'][:n_model_a]
        model_b_pred = self.results[model_b_name]['predictions'][:n_model_b]
        
        # Combine predictions and actual values
        combined_predictions = np.concatenate([model_a_pred, model_b_pred])
        combined_actual = np.concatenate([
            self.y_test[:n_model_a], 
            self.y_test[:n_model_b]
        ])
        
        # Calculate combined metrics
        combined_rmse = np.sqrt(mean_squared_error(combined_actual, combined_predictions))
        combined_mae = mean_absolute_error(combined_actual, combined_predictions)
        combined_r2 = r2_score(combined_actual, combined_predictions)
        
        # Store A/B test results
        self.ab_results = {
            'model_a': {
                'name': model_a_name,
                'samples': n_model_a,
                'rmse': self.results[model_a_name]['rmse'],
                'mae': self.results[model_a_name]['mae'],
                'r2': self.results[model_a_name]['r2']
            },
            'model_b': {
                'name': model_b_name,
                'samples': n_model_b,
                'rmse': self.results[model_b_name]['rmse'],
                'mae': self.results[model_b_name]['mae'],
                'r2': self.results[model_b_name]['r2']
            },
            'combined': {
                'rmse': combined_rmse,
                'mae': combined_mae,
                'r2': combined_r2
            },
            'traffic_split': traffic_split,
            'duration_days': duration_days,
            'total_samples': n_samples
        }
        
        print("✅ A/B test simulation completed")
        return self.ab_results
    
    def analyze_results(self):
        """Analyze A/B test results and determine winner"""
        print("📊 Analyzing A/B test results...")
        
        if not hasattr(self, 'ab_results'):
            print("❌ No A/B test results found. Run ab_test() first.")
            return None
        
        model_a = self.ab_results['model_a']
        model_b = self.ab_results['model_b']
        
        print(f"\n{'='*60}")
        print("A/B TEST RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\n📈 MODEL A ({model_a['name']}):")
        print(f"   Samples: {model_a['samples']}")
        print(f"   RMSE: {model_a['rmse']:.4f}")
        print(f"   MAE: {model_a['mae']:.4f}")
        print(f"   R²: {model_a['r2']:.4f}")
        
        print(f"\n📈 MODEL B ({model_b['name']}):")
        print(f"   Samples: {model_b['samples']}")
        print(f"   RMSE: {model_b['rmse']:.4f}")
        print(f"   MAE: {model_b['mae']:.4f}")
        print(f"   R²: {model_b['r2']:.4f}")
        
        # Determine winner based on RMSE (lower is better)
        if model_a['rmse'] < model_b['rmse']:
            winner = 'A'
            improvement = ((model_b['rmse'] - model_a['rmse']) / model_b['rmse']) * 100
        else:
            winner = 'B'
            improvement = ((model_a['rmse'] - model_b['rmse']) / model_a['rmse']) * 100
        
        print(f"\n🏆 WINNER: MODEL {winner}")
        print(f"   Improvement: {improvement:.2f}% better RMSE")
        
        # Statistical significance check
        significance = self._check_statistical_significance()
        print(f"   Statistical significance: {significance}")
        
        # Store analysis results
        self.analysis_results = {
            'winner': winner,
            'improvement': improvement,
            'significance': significance,
            'recommendation': self._get_recommendation(winner, improvement, significance)
        }
        
        print(f"\n💡 RECOMMENDATION: {self.analysis_results['recommendation']}")
        print(f"{'='*60}")
        
        return self.analysis_results
    
    def _check_statistical_significance(self, alpha: float = 0.05):
        """
        Check if the difference between models is statistically significant
        
        Args:
            alpha: Significance level (default 0.05)
        
        Returns:
            bool: True if statistically significant
        """
        # Simple significance check based on sample size and improvement
        # In a real scenario, you'd use proper statistical tests
        
        model_a = self.ab_results['model_a']
        model_b = self.ab_results['model_b']
        
        # Check if we have enough samples
        min_samples = min(model_a['samples'], model_b['samples'])
        if min_samples < self.min_samples:
            return False
        
        # Check if improvement is substantial
        rmse_diff = abs(model_a['rmse'] - model_b['rmse'])
        avg_rmse = (model_a['rmse'] + model_b['rmse']) / 2
        relative_improvement = rmse_diff / avg_rmse
        
        # Consider significant if relative improvement > 5%
        return relative_improvement > 0.05
    
    def _get_recommendation(self, winner: str, improvement: float, significance: bool):
        """Get recommendation based on analysis results"""
        if not significance:
            return "Continue testing - results not statistically significant"
        
        if improvement < 2:
            return "Marginal improvement - consider keeping current model"
        elif improvement < 5:
            return "Moderate improvement - consider switching to winner"
        else:
            return "Significant improvement - strongly recommend switching to winner"
    
    def save_results(self, filename: str = 'ab_test_results.json'):
        """Save A/B test results to file"""
        if not hasattr(self, 'ab_results'):
            print("❌ No A/B test results to save")
            return
        
        # Prepare results for saving
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'ab_results': self.ab_results,
            'analysis_results': getattr(self, 'analysis_results', None)
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"✅ Results saved to {filename}")
    
    def create_visualizations(self):
        """Create visualizations for A/B test results"""
        print("📊 Creating A/B test visualizations...")
        
        if not hasattr(self, 'ab_results'):
            print("❌ No A/B test results found. Run ab_test() first.")
            return
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        # 1. Model comparison bar chart
        self._plot_model_comparison()
        
        # 2. Prediction vs actual scatter plot
        self._plot_predictions_comparison()
        
        # 3. Error distribution
        self._plot_error_distribution()
        
        print("✅ Visualizations created in plots/ directory")
    
    def _plot_model_comparison(self):
        """Create bar chart comparing model metrics"""
        model_a = self.ab_results['model_a']
        model_b = self.ab_results['model_b']
        
        metrics = ['RMSE', 'MAE', 'R²']
        model_a_values = [model_a['rmse'], model_a['mae'], model_a['r2']]
        model_b_values = [model_b['rmse'], model_b['mae'], model_b['r2']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, model_a_values, width, label=model_a['name'], alpha=0.8)
        bars2 = ax.bar(x + width/2, model_b_values, width, label=model_b['name'], alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('A/B Test: Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/ab_test_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions_comparison(self):
        """Create scatter plot comparing predictions vs actual"""
        model_a = self.ab_results['model_a']
        model_b = self.ab_results['model_b']
        
        # Get predictions and actual values
        model_a_pred = self.results[model_a['name']]['predictions']
        model_b_pred = self.results[model_b['name']]['predictions']
        actual = self.results[model_a['name']]['test_actual']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Model A
        ax1.scatter(actual, model_a_pred, alpha=0.6, s=20)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Life Expectancy')
        ax1.set_ylabel('Predicted Life Expectancy')
        ax1.set_title(f'{model_a["name"]} - Predictions vs Actual')
        ax1.grid(True, alpha=0.3)
        
        # Model B
        ax2.scatter(actual, model_b_pred, alpha=0.6, s=20)
        ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Life Expectancy')
        ax2.set_ylabel('Predicted Life Expectancy')
        ax2.set_title(f'{model_b["name"]} - Predictions vs Actual')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/ab_test_predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self):
        """Create histogram of prediction errors"""
        model_a = self.ab_results['model_a']
        model_b = self.ab_results['model_b']
        
        # Calculate errors
        model_a_errors = self.results[model_a['name']]['test_actual'] - self.results[model_a['name']]['predictions']
        model_b_errors = self.results[model_b['name']]['test_actual'] - self.results[model_b['name']]['predictions']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(model_a_errors, bins=30, alpha=0.7, label=model_a['name'], density=True)
        ax.hist(model_b_errors, bins=30, alpha=0.7, label=model_b['name'], density=True)
        
        ax.set_xlabel('Prediction Error (Actual - Predicted)')
        ax.set_ylabel('Density')
        ax.set_title('A/B Test: Error Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/ab_test_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run A/B testing"""
    print("🧪 A/B TESTING SYSTEM FOR LIFE EXPECTANCY ML MODELS")
    print("=" * 60)
    
    # Initialize A/B testing system
    ab_system = ABTestingSystem()
    
    # Train models
    ab_system.train_models()
    
    # Run A/B test
    ab_system.run_ab_test(traffic_split=0.5, duration_days=7)
    
    # Analyze results
    ab_system.analyze_results()
    
    # Create visualizations
    ab_system.create_visualizations()
    
    # Save results
    ab_system.save_results()
    
    print("\n🎉 A/B Testing completed successfully!")
    print("📊 Check the plots/ directory for visualizations")
    print("📄 Check ab_test_results.json for detailed results")


if __name__ == "__main__":
    main()
