"""
Auto Model Replacement System for Life Expectancy ML Pipeline
Automatically replaces models in production when better ones are found
"""

import pandas as pd
import numpy as np
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st


class ModelAutoReplacement:
    """
    Automatically replace models in production when better ones are found
    """
    
    def __init__(self, 
                 current_model_path: str = 'models/best_life_expectancy_model.pkl',
                 backup_dir: str = 'models/backups',
                 performance_threshold: float = 0.05):  # 5% improvement threshold
        """
        Initialize Auto Model Replacement system
        
        Args:
            current_model_path: Path to current production model
            backup_dir: Directory to store model backups
            performance_threshold: Minimum improvement required to replace model
        """
        self.current_model_path = current_model_path
        self.backup_dir = backup_dir
        self.performance_threshold = performance_threshold
        self.replacement_history = []
        self.model_registry = {}
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Load current model info
        self._load_current_model_info()
    
    def _load_current_model_info(self):
        """Load information about current production model"""
        try:
            if os.path.exists(self.current_model_path):
                self.current_model = joblib.load(self.current_model_path)
                self.current_model_info = {
                    'path': self.current_model_path,
                    'type': type(self.current_model).__name__,
                    'loaded_at': datetime.now().isoformat(),
                    'performance': self._evaluate_model_performance(self.current_model)
                }
            else:
                self.current_model = None
                self.current_model_info = None
        except Exception as e:
            print(f"Error loading current model: {e}")
            self.current_model = None
            self.current_model_info = None
    
    def _evaluate_model_performance(self, model, test_data: pd.DataFrame = None) -> Dict:
        """Evaluate model performance"""
        if test_data is None:
            # Use a small sample for evaluation
            test_data = pd.read_csv('data/clean_data.csv').head(100)
        
        # Prepare test data
        target = 'life_expectancy'
        exclude_cols = ['country', 'year', 'status', target]
        feature_cols = [col for col in test_data.columns if col not in exclude_cols]
        
        X_test = test_data[feature_cols].fillna(test_data[feature_cols].median())
        y_test = test_data[target]
        
        try:
            y_pred = model.predict(X_test)
            
            return {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'evaluated_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': -float('inf'),
                'error': str(e)
            }
    
    def register_new_model(self, model, model_name: str, model_type: str, 
                          training_data: pd.DataFrame = None) -> Dict:
        """
        Register a new model for potential replacement
        
        Args:
            model: Trained model object
            model_name: Name for the model
            model_type: Type of model (e.g., 'RandomForest', 'GradientBoosting')
            training_data: Data used for training (for evaluation)
        
        Returns:
            Dict with registration results
        """
        print(f"📝 Registering new model: {model_name} ({model_type})")
        
        # Evaluate model performance
        performance = self._evaluate_model_performance(model, training_data)
        
        # Register model
        model_id = f"{model_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.model_registry[model_id] = {
            'model': model,
            'name': model_name,
            'type': model_type,
            'performance': performance,
            'registered_at': datetime.now().isoformat(),
            'status': 'candidate'
        }
        
        print(f"✅ Model registered with ID: {model_id}")
        print(f"   Performance: RMSE={performance['rmse']:.4f}, R²={performance['r2']:.4f}")
        
        return {
            'model_id': model_id,
            'performance': performance,
            'status': 'registered'
        }
    
    def check_for_replacement(self, candidate_model_id: str) -> Dict:
        """
        Check if a candidate model should replace the current model
        
        Args:
            candidate_model_id: ID of the candidate model
        
        Returns:
            Dict with replacement decision
        """
        if candidate_model_id not in self.model_registry:
            return {
                'should_replace': False,
                'reason': 'Model not found in registry',
                'improvement': 0.0
            }
        
        candidate_model = self.model_registry[candidate_model_id]
        candidate_performance = candidate_model['performance']
        
        # Check if current model exists
        if self.current_model_info is None:
            return {
                'should_replace': True,
                'reason': 'No current model in production',
                'improvement': float('inf')
            }
        
        current_performance = self.current_model_info['performance']
        
        # Calculate improvement
        rmse_improvement = (current_performance['rmse'] - candidate_performance['rmse']) / current_performance['rmse']
        r2_improvement = candidate_performance['r2'] - current_performance['r2']
        
        # Check if improvement meets threshold
        should_replace = rmse_improvement >= self.performance_threshold
        
        decision = {
            'should_replace': should_replace,
            'reason': f"RMSE improvement: {rmse_improvement:.2%}, R² improvement: {r2_improvement:.4f}",
            'improvement': rmse_improvement,
            'current_performance': current_performance,
            'candidate_performance': candidate_performance,
            'threshold': self.performance_threshold
        }
        
        if should_replace:
            print(f"🔄 Model replacement recommended!")
            print(f"   Improvement: {rmse_improvement:.2%}")
            print(f"   Current RMSE: {current_performance['rmse']:.4f}")
            print(f"   Candidate RMSE: {candidate_performance['rmse']:.4f}")
        else:
            print(f"❌ Model replacement not recommended")
            print(f"   Improvement: {rmse_improvement:.2%} (threshold: {self.performance_threshold:.2%})")
        
        return decision
    
    def replace_model(self, candidate_model_id: str, backup_current: bool = True) -> Dict:
        """
        Replace current model with candidate model
        
        Args:
            candidate_model_id: ID of the candidate model
            backup_current: Whether to backup current model
        
        Returns:
            Dict with replacement results
        """
        if candidate_model_id not in self.model_registry:
            return {
                'success': False,
                'error': 'Model not found in registry'
            }
        
        candidate_model = self.model_registry[candidate_model_id]
        
        try:
            # Backup current model if it exists
            if backup_current and self.current_model_info is not None:
                backup_path = os.path.join(
                    self.backup_dir, 
                    f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                )
                shutil.copy2(self.current_model_path, backup_path)
                print(f"📦 Current model backed up to: {backup_path}")
            
            # Save new model
            joblib.dump(candidate_model['model'], self.current_model_path)
            
            # Update current model info
            self.current_model = candidate_model['model']
            self.current_model_info = {
                'path': self.current_model_path,
                'type': candidate_model['type'],
                'loaded_at': datetime.now().isoformat(),
                'performance': candidate_model['performance'],
                'replaced_from': candidate_model_id
            }
            
            # Update model registry
            self.model_registry[candidate_model_id]['status'] = 'production'
            
            # Record replacement
            replacement_record = {
                'timestamp': datetime.now().isoformat(),
                'old_model': self.current_model_info.get('type', 'Unknown'),
                'new_model': candidate_model['type'],
                'new_model_id': candidate_model_id,
                'improvement': self.check_for_replacement(candidate_model_id)['improvement'],
                'backup_created': backup_current
            }
            
            self.replacement_history.append(replacement_record)
            
            print(f"✅ Model successfully replaced!")
            print(f"   New model: {candidate_model['type']}")
            print(f"   Performance: RMSE={candidate_model['performance']['rmse']:.4f}")
            
            return {
                'success': True,
                'new_model_type': candidate_model['type'],
                'new_model_performance': candidate_model['performance'],
                'replacement_record': replacement_record
            }
            
        except Exception as e:
            print(f"❌ Error replacing model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def auto_replace_if_better(self, candidate_model_id: str) -> Dict:
        """
        Automatically replace model if it's better than current
        
        Args:
            candidate_model_id: ID of the candidate model
        
        Returns:
            Dict with auto-replacement results
        """
        print(f"🤖 Auto-replacement check for model: {candidate_model_id}")
        
        # Check if replacement is recommended
        decision = self.check_for_replacement(candidate_model_id)
        
        if decision['should_replace']:
            print(f"🔄 Auto-replacing model...")
            replacement_result = self.replace_model(candidate_model_id)
            
            return {
                'auto_replaced': True,
                'decision': decision,
                'replacement_result': replacement_result
            }
        else:
            print(f"❌ Auto-replacement skipped - improvement insufficient")
            return {
                'auto_replaced': False,
                'decision': decision,
                'reason': 'Improvement below threshold'
            }
    
    def get_streamlit_dashboard_data(self) -> Dict:
        """Get data formatted for Streamlit dashboard"""
        return {
            'current_model': self.current_model_info,
            'candidate_models': len([m for m in self.model_registry.values() if m['status'] == 'candidate']),
            'production_models': len([m for m in self.model_registry.values() if m['status'] == 'production']),
            'replacement_history': self.replacement_history[-10:],  # Last 10 replacements
            'performance_threshold': self.performance_threshold,
            'total_models_registered': len(self.model_registry)
        }
    
    def create_streamlit_visualizations(self):
        """Create visualizations for Streamlit dashboard"""
        if not self.replacement_history:
            return None
        
        # Create replacement history chart
        timestamps = [entry['timestamp'] for entry in self.replacement_history]
        improvements = [entry['improvement'] for entry in self.replacement_history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, improvements, marker='o', linewidth=2, markersize=6)
        ax.axhline(y=self.performance_threshold, color='red', linestyle='--', alpha=0.7, label='Replacement Threshold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Performance Improvement')
        ax.set_title('Model Replacement History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def save_replacement_data(self, filename: str = 'model_replacement_data.json'):
        """Save replacement data to file"""
        replacement_data = {
            'current_model_info': self.current_model_info,
            'model_registry': {k: {**v, 'model': None} for k, v in self.model_registry.items()},  # Exclude model objects
            'replacement_history': self.replacement_history,
            'performance_threshold': self.performance_threshold,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(replacement_data, f, indent=2, default=str)


def create_streamlit_auto_replacement_dashboard():
    """Create Streamlit dashboard for auto model replacement"""
    st.title("🔄 Auto Model Replacement Dashboard")
    
    # Initialize auto replacement system
    if 'auto_replacement' not in st.session_state:
        st.session_state.auto_replacement = ModelAutoReplacement()
    
    auto_replacement = st.session_state.auto_replacement
    
    # Get dashboard data
    dashboard_data = auto_replacement.get_streamlit_dashboard_data()
    
    # Current model status
    st.subheader("📊 Current Model Status")
    
    if dashboard_data['current_model']:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", dashboard_data['current_model']['type'])
        
        with col2:
            st.metric("RMSE", f"{dashboard_data['current_model']['performance']['rmse']:.4f}")
        
        with col3:
            st.metric("R² Score", f"{dashboard_data['current_model']['performance']['r2']:.4f}")
        
        with col4:
            st.metric("Loaded At", dashboard_data['current_model']['loaded_at'][:19])
    else:
        st.warning("No model currently in production")
    
    # Model registry
    st.subheader("📝 Model Registry")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Candidate Models", dashboard_data['candidate_models'])
    
    with col2:
        st.metric("Production Models", dashboard_data['production_models'])
    
    with col3:
        st.metric("Total Registered", dashboard_data['total_models_registered'])
    
    # Replacement history
    if dashboard_data['replacement_history']:
        st.subheader("📈 Replacement History")
        
        # Show recent replacements
        for replacement in dashboard_data['replacement_history'][-5:]:
            with st.expander(f"Replacement - {replacement['timestamp'][:19]}"):
                st.write(f"**From:** {replacement['old_model']}")
                st.write(f"**To:** {replacement['new_model']}")
                st.write(f"**Improvement:** {replacement['improvement']:.2%}")
                st.write(f"**Backup Created:** {replacement['backup_created']}")
        
        # Show visualization
        fig = auto_replacement.create_streamlit_visualizations()
        if fig:
            st.pyplot(fig)
    
    # Manual model replacement
    st.subheader("🔄 Manual Model Replacement")
    
    # Show candidate models
    candidate_models = {k: v for k, v in auto_replacement.model_registry.items() if v['status'] == 'candidate'}
    
    if candidate_models:
        st.write("**Available Candidate Models:**")
        
        for model_id, model_info in candidate_models.items():
            with st.expander(f"{model_info['name']} ({model_info['type']})"):
                st.write(f"**Performance:**")
                st.write(f"- RMSE: {model_info['performance']['rmse']:.4f}")
                st.write(f"- MAE: {model_info['performance']['mae']:.4f}")
                st.write(f"- R²: {model_info['performance']['r2']:.4f}")
                st.write(f"**Registered:** {model_info['registered_at'][:19]}")
                
                # Check if should replace
                decision = auto_replacement.check_for_replacement(model_id)
                
                if decision['should_replace']:
                    st.success(f"✅ Recommended for replacement (Improvement: {decision['improvement']:.2%})")
                    
                    if st.button(f"Replace with {model_info['name']}", key=f"replace_{model_id}"):
                        result = auto_replacement.replace_model(model_id)
                        if result['success']:
                            st.success("Model replaced successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error replacing model: {result['error']}")
                else:
                    st.warning(f"❌ Not recommended (Improvement: {decision['improvement']:.2%})")
    else:
        st.info("No candidate models available")
    
    # Auto-replacement settings
    st.subheader("⚙️ Auto-Replacement Settings")
    
    new_threshold = st.slider(
        "Performance Improvement Threshold",
        min_value=0.01,
        max_value=0.20,
        value=auto_replacement.performance_threshold,
        step=0.01,
        format="%.1%"
    )
    
    if new_threshold != auto_replacement.performance_threshold:
        auto_replacement.performance_threshold = new_threshold
        st.success(f"Threshold updated to {new_threshold:.1%}")
    
    # Test auto-replacement
    if st.button("Test Auto-Replacement"):
        if candidate_models:
            # Test with first candidate
            first_candidate = list(candidate_models.keys())[0]
            result = auto_replacement.auto_replace_if_better(first_candidate)
            
            if result['auto_replaced']:
                st.success("Auto-replacement test completed - Model replaced!")
            else:
                st.info(f"Auto-replacement test completed - {result['reason']}")
        else:
            st.warning("No candidate models available for testing")


if __name__ == "__main__":
    # Test the auto replacement system
    print("🔄 Testing Auto Model Replacement System...")
    
    auto_replacement = ModelAutoReplacement()
    
    # Create a test model
    from sklearn.ensemble import GradientBoostingRegressor
    test_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Load some data for training
    data = pd.read_csv('data/clean_data.csv')
    target = 'life_expectancy'
    exclude_cols = ['country', 'year', 'status', target]
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols].fillna(data[feature_cols].median())
    y = data[target]
    
    # Train test model
    test_model.fit(X, y)
    
    # Register the model
    registration_result = auto_replacement.register_new_model(
        test_model, 
        "TestGradientBoosting", 
        "GradientBoosting",
        data
    )
    
    print(f"Registration result: {registration_result}")
    
    # Test auto-replacement
    auto_result = auto_replacement.auto_replace_if_better(registration_result['model_id'])
    print(f"Auto-replacement result: {auto_result}")
    
    # Save data
    auto_replacement.save_replacement_data()
    
    print("✅ Auto model replacement test completed!")