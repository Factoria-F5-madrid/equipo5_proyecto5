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
        
        
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.traffic_split = 0.5  
        self.min_samples = 100  
        
     
        self.df = pd.read_csv(data_path)
     
   
        self._prepare_data()
        
      
        self._initialize_models()
        
       
    
    def _prepare_data(self):
        """Prepare features and target for model training"""
      
        
        
        target = 'life_expectancy'
        exclude_cols = ['country', 'year', 'status', target]
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
     
        self.X = self.df[feature_cols]
        self.y = self.df[target]
        
      
        self.X = self.X.fillna(self.X.median())
        
   
    def _initialize_models(self):
        """Initialize different models for A/B testing"""
      
        self.models['RandomForest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
      
        self.models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        
        self.models['LinearRegression'] = LinearRegression()
        
     
    
    def train_models(self, test_size: float = 0.2):
        """
        Train all models for A/B testing
        
        Args:
            test_size: Proportion of data for testing
        """
       
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
    
        for name, model in self.models.items():
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            
            y_pred = model.predict(X_test)
            
         
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            
            self.results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'training_time': training_time,
                'predictions': y_pred,
                'test_actual': y_test
            }
            
           
    
    def run_ab_test(self, traffic_split: float = 0.5, duration_days: int = 7):
        """
        Run A/B test simulation
        
        Args:
            traffic_split: Proportion of traffic for each model (0.5 = 50/50)
            duration_days: Duration of A/B test in days
        """
       
        
        self.traffic_split = traffic_split
        
        
        n_samples = len(self.X_test)
        n_model_a = int(n_samples * traffic_split)
        n_model_b = n_samples - n_model_a
        
       
        model_names = list(self.models.keys())
        model_a_name = model_names[0] 
        model_b_name = model_names[1]  
        
     
        model_a_pred = self.results[model_a_name]['predictions'][:n_model_a]
        model_b_pred = self.results[model_b_name]['predictions'][:n_model_b]
        
    
        combined_predictions = np.concatenate([model_a_pred, model_b_pred])
        combined_actual = np.concatenate([
            self.y_test[:n_model_a], 
            self.y_test[:n_model_b]
        ])
        
     
        combined_rmse = np.sqrt(mean_squared_error(combined_actual, combined_predictions))
        combined_mae = mean_absolute_error(combined_actual, combined_predictions)
        combined_r2 = r2_score(combined_actual, combined_predictions)
        
      
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
        
      
        return self.ab_results
    
    def analyze_results(self):
        """Analyze A/B test results and determine winner"""
       
        
        if not hasattr(self, 'ab_results'):
          
            return None
        
        model_a = self.ab_results['model_a']
        model_b = self.ab_results['model_b']
        
       
        if model_a['rmse'] < model_b['rmse']:
            winner = 'A'
            improvement = ((model_b['rmse'] - model_a['rmse']) / model_b['rmse']) * 100
        else:
            winner = 'B'
            improvement = ((model_a['rmse'] - model_b['rmse']) / model_a['rmse']) * 100
        
     
        significance = self._check_statistical_significance()
        
        self.analysis_results = {
            'winner': winner,
            'improvement': improvement,
            'significance': significance,
            'recommendation': self._get_recommendation(winner, improvement, significance)
        }
        
      
        
        return self.analysis_results
    
    def _check_statistical_significance(self, alpha: float = 0.05):
        """
        Check if the difference between models is statistically significant
        
        Args:
            alpha: Significance level (default 0.05)
        
        Returns:
            bool: True if statistically significant
        """
     
        
        model_a = self.ab_results['model_a']
        model_b = self.ab_results['model_b']
        
     
        min_samples = min(model_a['samples'], model_b['samples'])
        if min_samples < self.min_samples:
            return False
        
     
        rmse_diff = abs(model_a['rmse'] - model_b['rmse'])
        avg_rmse = (model_a['rmse'] + model_b['rmse']) / 2
        relative_improvement = rmse_diff / avg_rmse
        
     
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
          
            return
        
   
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'ab_results': self.ab_results,
            'analysis_results': getattr(self, 'analysis_results', None)
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
       
    
    def create_visualizations(self):
        """Create visualizations for A/B test results"""
       
        
        if not hasattr(self, 'ab_results'):
           
            return
   
        os.makedirs('plots', exist_ok=True)
        
  
        self._plot_model_comparison()
        
       
        self._plot_predictions_comparison()
        
     
        self._plot_error_distribution()
        
        
    
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
        
      
        model_a_pred = self.results[model_a['name']]['predictions']
        model_b_pred = self.results[model_b['name']]['predictions']
        actual = self.results[model_a['name']]['test_actual']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
       
        ax1.scatter(actual, model_a_pred, alpha=0.6, s=20)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Life Expectancy')
        ax1.set_ylabel('Predicted Life Expectancy')
        ax1.set_title(f'{model_a["name"]} - Predictions vs Actual')
        ax1.grid(True, alpha=0.3)
        
    
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
  
    ab_system = ABTestingSystem()
    

    ab_system.train_models()
    

    ab_system.run_ab_test(traffic_split=0.5, duration_days=7)

    ab_system.analyze_results()
    
   
    ab_system.create_visualizations()
    

    ab_system.save_results()

if __name__ == "__main__":
    main()
