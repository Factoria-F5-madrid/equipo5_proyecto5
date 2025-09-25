"""
Data Drift Monitoring System for Life Expectancy ML Pipeline
Detects changes in data distribution over time and provides Streamlit integration
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import streamlit as st


class DataDriftMonitor:
    """
    Monitor data drift in the ML pipeline with Streamlit integration
    """
    
    def __init__(self, reference_data_path: str = 'data/clean_data.csv'):
        """
        Initialize Data Drift Monitor
        
        Args:
            reference_data_path: Path to reference dataset
        """
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.reference_stats = {}
        self.drift_threshold = 0.1  # 10% threshold for drift detection
        self.monitoring_history = []
        self.alert_history = []
        
        # Load reference data
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load and prepare reference data"""
        self.reference_data = pd.read_csv(self.reference_data_path)
        
        # Prepare features (same as in pipeline)
        target = 'life_expectancy'
        exclude_cols = ['country', 'year', 'status', target]
        feature_cols = [col for col in self.reference_data.columns if col not in exclude_cols]
        
        self.reference_features = self.reference_data[feature_cols]
        self.reference_target = self.reference_data[target]
        
        # Calculate reference statistics
        self._calculate_reference_stats()
    
    def _calculate_reference_stats(self):
        """Calculate reference statistics for drift detection"""
        # Numerical features statistics
        numerical_features = self.reference_features.select_dtypes(include=[np.number]).columns
        
        for feature in numerical_features:
            values = self.reference_features[feature].dropna()
            
            self.reference_stats[feature] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75),
                'skewness': values.skew(),
                'kurtosis': values.kurtosis()
            }
        
        # Target variable statistics
        target_values = self.reference_target.dropna()
        self.reference_stats['target'] = {
            'mean': target_values.mean(),
            'std': target_values.std(),
            'min': target_values.min(),
            'max': target_values.max(),
            'median': target_values.median()
        }
    
    def detect_drift(self, new_data: pd.DataFrame, feature_cols: List[str] = None) -> Dict:
        """
        Detect data drift in new data compared to reference
        
        Args:
            new_data: New data to check for drift
            feature_cols: List of feature columns to check
        
        Returns:
            Dict with drift detection results
        """
        if feature_cols is None:
            target = 'life_expectancy'
            exclude_cols = ['country', 'year', 'status', target]
            feature_cols = [col for col in new_data.columns if col not in exclude_cols]
        
        # Prepare new data
        new_features = new_data[feature_cols]
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'total_features_checked': len(feature_cols),
            'features_with_drift': 0,
            'drift_details': {},
            'overall_drift_detected': False,
            'drift_severity': 'none',
            'alert_level': 'green'
        }
        
        # Check each feature for drift
        for feature in feature_cols:
            if feature not in self.reference_stats:
                continue
            
            drift_info = self._check_feature_drift(feature, new_features[feature])
            drift_results['drift_details'][feature] = drift_info
            
            if drift_info['drift_detected']:
                drift_results['features_with_drift'] += 1
        
        # Determine overall drift
        drift_ratio = drift_results['features_with_drift'] / drift_results['total_features_checked']
        drift_results['drift_ratio'] = drift_ratio
        
        if drift_ratio > self.drift_threshold:
            drift_results['overall_drift_detected'] = True
            if drift_ratio > 0.3:
                drift_results['drift_severity'] = 'high'
                drift_results['alert_level'] = 'red'
            elif drift_ratio > 0.2:
                drift_results['drift_severity'] = 'medium'
                drift_results['alert_level'] = 'orange'
            else:
                drift_results['drift_severity'] = 'low'
                drift_results['alert_level'] = 'yellow'
        
        # Store in monitoring history
        self.monitoring_history.append(drift_results)
        
        # Create alert if drift detected
        if drift_results['overall_drift_detected']:
            self._create_alert(drift_results)
        
        return drift_results
    
    def _check_feature_drift(self, feature: str, new_values: pd.Series) -> Dict:
        """Check drift for a specific feature"""
        if feature not in self.reference_stats:
            return {'drift_detected': False, 'reason': 'Feature not in reference'}
        
        ref_stats = self.reference_stats[feature]
        new_values_clean = new_values.dropna()
        
        if len(new_values_clean) == 0:
            return {'drift_detected': False, 'reason': 'No valid values in new data'}
        
        drift_info = {
            'drift_detected': False,
            'drift_score': 0.0,
            'tests_performed': [],
            'details': {}
        }
        
        # Test 1: Kolmogorov-Smirnov test
        try:
            ks_statistic, ks_p_value = stats.ks_2samp(
                self.reference_features[feature].dropna(),
                new_values_clean
            )
            drift_info['tests_performed'].append('ks_test')
            drift_info['details']['ks_statistic'] = ks_statistic
            drift_info['details']['ks_p_value'] = ks_p_value
            
            if ks_p_value < 0.05:  # Significant difference
                drift_info['drift_score'] += 0.4
        except Exception as e:
            drift_info['details']['ks_error'] = str(e)
        
        # Test 2: Mean shift test
        try:
            new_mean = new_values_clean.mean()
            ref_mean = ref_stats['mean']
            ref_std = ref_stats['std']
            
            # Z-score for mean difference
            z_score = abs(new_mean - ref_mean) / ref_std
            drift_info['details']['mean_z_score'] = z_score
            
            if z_score > 2:  # More than 2 standard deviations
                drift_info['drift_score'] += 0.3
        except Exception as e:
            drift_info['details']['mean_test_error'] = str(e)
        
        # Test 3: Variance test
        try:
            new_std = new_values_clean.std()
            ref_std = ref_stats['std']
            
            # F-test for variance
            f_statistic = (new_std / ref_std) ** 2
            drift_info['details']['f_statistic'] = f_statistic
            
            if f_statistic > 1.5 or f_statistic < 0.67:  # Significant variance change
                drift_info['drift_score'] += 0.3
        except Exception as e:
            drift_info['details']['variance_test_error'] = str(e)
        
        # Determine if drift is detected
        drift_info['drift_detected'] = drift_info['drift_score'] > 0.5
        
        return drift_info
    
    def _create_alert(self, drift_results: Dict):
        """Create alert for drift detection"""
        alert = {
            'timestamp': drift_results['timestamp'],
            'alert_type': 'data_drift',
            'severity': drift_results['drift_severity'],
            'message': f"Data drift detected: {drift_results['features_with_drift']}/{drift_results['total_features_checked']} features affected",
            'details': drift_results
        }
        
        self.alert_history.append(alert)
    
    def get_streamlit_dashboard_data(self) -> Dict:
        """Get data formatted for Streamlit dashboard"""
        if not self.monitoring_history:
            return {
                'status': 'No data available',
                'last_check': 'Never',
                'drift_detected': False,
                'alert_level': 'green',
                'features_checked': 0,
                'features_with_drift': 0,
                'drift_ratio': 0.0,
                'recent_alerts': []
            }
        
        latest_result = self.monitoring_history[-1]
        
        return {
            'status': 'Monitoring active',
            'last_check': latest_result['timestamp'],
            'drift_detected': latest_result['overall_drift_detected'],
            'alert_level': latest_result['alert_level'],
            'features_checked': latest_result['total_features_checked'],
            'features_with_drift': latest_result['features_with_drift'],
            'drift_ratio': latest_result['drift_ratio'],
            'recent_alerts': self.alert_history[-5:] if self.alert_history else []
        }
    
    def create_streamlit_visualizations(self):
        """Create visualizations for Streamlit dashboard"""
        if not self.monitoring_history:
            return None
        
        # Create drift trend chart
        timestamps = [entry['timestamp'] for entry in self.monitoring_history]
        drift_ratios = [entry['drift_ratio'] for entry in self.monitoring_history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timestamps, drift_ratios, marker='o', linewidth=2, markersize=6)
        ax.axhline(y=self.drift_threshold, color='red', linestyle='--', alpha=0.7, label='Drift Threshold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Drift Ratio')
        ax.set_title('Data Drift Monitoring Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def save_monitoring_data(self, filename: str = 'data_drift_monitoring.json'):
        """Save monitoring data to file"""
        monitoring_data = {
            'reference_stats': self.reference_stats,
            'monitoring_history': self.monitoring_history,
            'alert_history': self.alert_history,
            'drift_threshold': self.drift_threshold,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(monitoring_data, f, indent=2, default=str)


def create_streamlit_drift_dashboard():
    """Create Streamlit dashboard for data drift monitoring"""
    st.title("🔍 Data Drift Monitoring Dashboard")
    
    # Initialize monitor
    if 'drift_monitor' not in st.session_state:
        st.session_state.drift_monitor = DataDriftMonitor()
    
    monitor = st.session_state.drift_monitor
    
    # Get dashboard data
    dashboard_data = monitor.get_streamlit_dashboard_data()
    
    # Status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", dashboard_data['status'])
    
    with col2:
        st.metric("Last Check", dashboard_data['last_check'][:19] if dashboard_data['last_check'] != 'Never' else 'Never')
    
    with col3:
        drift_status = "🚨 Drift Detected" if dashboard_data['drift_detected'] else "✅ No Drift"
        st.metric("Drift Status", drift_status)
    
    with col4:
        alert_color = {
            'green': '🟢',
            'yellow': '🟡', 
            'orange': '🟠',
            'red': '🔴'
        }.get(dashboard_data['alert_level'], '⚪')
        st.metric("Alert Level", f"{alert_color} {dashboard_data['alert_level'].title()}")
    
    # Drift metrics
    st.subheader("📊 Drift Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Features Checked", dashboard_data['features_checked'])
    
    with col2:
        st.metric("Features with Drift", dashboard_data['features_with_drift'])
    
    with col3:
        st.metric("Drift Ratio", f"{dashboard_data['drift_ratio']:.2%}")
    
    # Visualizations
    st.subheader("📈 Drift Trends")
    if monitor.monitoring_history:
        fig = monitor.create_streamlit_visualizations()
        if fig:
            st.pyplot(fig)
    else:
        st.info("No monitoring data available yet. Upload some data to start monitoring.")
    
    # Recent alerts
    if dashboard_data['recent_alerts']:
        st.subheader("🚨 Recent Alerts")
        for alert in dashboard_data['recent_alerts']:
            with st.expander(f"Alert - {alert['timestamp'][:19]} - {alert['severity'].title()}"):
                st.write(f"**Type:** {alert['alert_type']}")
                st.write(f"**Message:** {alert['message']}")
                st.write(f"**Severity:** {alert['severity']}")
    
    # Data upload for testing
    st.subheader("📤 Test Data Drift Detection")
    uploaded_file = st.file_uploader("Upload new data to test for drift", type=['csv'])
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            drift_results = monitor.detect_drift(new_data)
            
            st.success("Data drift analysis completed!")
            
            # Show results
            if drift_results['overall_drift_detected']:
                st.error(f"🚨 Data drift detected! {drift_results['features_with_drift']} features affected.")
                st.write(f"**Drift Severity:** {drift_results['drift_severity']}")
                st.write(f"**Drift Ratio:** {drift_results['drift_ratio']:.2%}")
            else:
                st.success("✅ No significant data drift detected.")
            
            # Show detailed results
            with st.expander("Detailed Drift Analysis"):
                for feature, details in drift_results['drift_details'].items():
                    if details['drift_detected']:
                        st.write(f"**{feature}:** Drift detected (Score: {details['drift_score']:.2f})")
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")


if __name__ == "__main__":
    # Test the drift monitor
    print("🔍 Testing Data Drift Monitor...")
    
    monitor = DataDriftMonitor()
    
    # Simulate new data with some drift
    reference_data = pd.read_csv('data/clean_data.csv')
    new_data = reference_data.copy()
    
    # Add some drift
    np.random.seed(42)
    new_data['gdp'] = new_data['gdp'] * (1 + np.random.normal(0, 0.1, len(new_data)))
    new_data['adult_mortality'] = new_data['adult_mortality'] * (1 + np.random.normal(0, 0.05, len(new_data)))
    
    # Detect drift
    drift_results = monitor.detect_drift(new_data)
    
    print(f"Drift detected: {drift_results['overall_drift_detected']}")
    print(f"Features with drift: {drift_results['features_with_drift']}")
    print(f"Drift ratio: {drift_results['drift_ratio']:.2%}")
    
    # Save monitoring data
    monitor.save_monitoring_data()
    
    print("✅ Data drift monitoring test completed!")
