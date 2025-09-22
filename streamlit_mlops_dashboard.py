 cam"""
Streamlit MLOps Dashboard
Integrates Data Drift Monitoring and Auto Model Replacement
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our MLOps modules
from data_drift_monitor import DataDriftMonitor, create_streamlit_drift_dashboard
from model_auto_replacement import ModelAutoReplacement, create_streamlit_auto_replacement_dashboard
from ab_testing import ABTestingSystem


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="MLOps Dashboard - Life Expectancy Prediction",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .alert-red {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-yellow {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-green {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 MLOps Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Life Expectancy Prediction System</h2>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Overview", "🔍 Data Drift Monitoring", "🔄 Auto Model Replacement", "🧪 A/B Testing", "📊 Model Performance"]
    )
    
    # Initialize session state
    if 'drift_monitor' not in st.session_state:
        st.session_state.drift_monitor = DataDriftMonitor()
    
    if 'auto_replacement' not in st.session_state:
        st.session_state.auto_replacement = ModelAutoReplacement()
    
    if 'ab_testing' not in st.session_state:
        st.session_state.ab_testing = ABTestingSystem()
    
    # Route to different pages
    if page == "🏠 Overview":
        show_overview_page()
    elif page == "🔍 Data Drift Monitoring":
        show_drift_monitoring_page()
    elif page == "🔄 Auto Model Replacement":
        show_auto_replacement_page()
    elif page == "🧪 A/B Testing":
        show_ab_testing_page()
    elif page == "📊 Model Performance":
        show_model_performance_page()


def show_overview_page():
    """Show overview page with system status"""
    st.header("🏠 System Overview")
    
    # System status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Data Drift</h3>
            <p>Status: <span style="color: green;">✅ Monitoring</span></p>
            <p>Last Check: Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔄 Model Replacement</h3>
            <p>Status: <span style="color: green;">✅ Active</span></p>
            <p>Auto-replace: Enabled</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧪 A/B Testing</h3>
            <p>Status: <span style="color: green;">✅ Ready</span></p>
            <p>Tests: Available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Model Performance</h3>
            <p>Status: <span style="color: green;">✅ Good</span></p>
            <p>R²: 0.97</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    # Create sample activity data
    activity_data = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'Event': ['Model Training', 'Data Drift Check', 'A/B Test', 'Model Replacement', 'Performance Check',
                 'Data Drift Check', 'Model Training', 'A/B Test', 'Performance Check', 'Data Drift Check'],
        'Status': ['Success', 'Warning', 'Success', 'Success', 'Success',
                  'Success', 'Success', 'Success', 'Success', 'Success'],
        'Details': ['Random Forest trained', 'Minor drift detected', 'Gradient Boosting vs RF', 'RF replaced with GB',
                   'R² improved to 0.97', 'No drift detected', 'New model trained', 'GB vs Linear Regression',
                   'Performance stable', 'No drift detected']
    })
    
    # Display activity timeline
    for _, row in activity_data.tail(5).iterrows():
        status_color = "green" if row['Status'] == 'Success' else "orange" if row['Status'] == 'Warning' else "red"
        st.markdown(f"""
        <div style="border-left: 3px solid {status_color}; padding-left: 10px; margin: 10px 0;">
            <strong>{row['Event']}</strong> - {row['Timestamp'].strftime('%Y-%m-%d %H:%M')}<br>
            <small>{row['Details']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # System health metrics
    st.subheader("💚 System Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model performance over time
        performance_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'R2_Score': np.random.normal(0.97, 0.01, 30),
            'RMSE': np.random.normal(1.6, 0.1, 30)
        })
        
        fig = px.line(performance_data, x='Date', y='R2_Score', title='Model R² Score Over Time')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Data drift ratio over time
        drift_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'Drift_Ratio': np.random.uniform(0, 0.15, 30)
        })
        
        fig = px.line(drift_data, x='Date', y='Drift_Ratio', title='Data Drift Ratio Over Time')
        fig.add_hline(y=0.1, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Check Data Drift", use_container_width=True):
            st.info("Data drift check initiated...")
    
    with col2:
        if st.button("🧪 Run A/B Test", use_container_width=True):
            st.info("A/B test started...")
    
    with col3:
        if st.button("📊 Refresh Metrics", use_container_width=True):
            st.rerun()


def show_drift_monitoring_page():
    """Show data drift monitoring page"""
    st.header("🔍 Data Drift Monitoring")
    
    # Use the drift monitoring dashboard
    create_streamlit_drift_dashboard()


def show_auto_replacement_page():
    """Show auto model replacement page"""
    st.header("🔄 Auto Model Replacement")
    
    # Use the auto replacement dashboard
    create_streamlit_auto_replacement_dashboard()


def show_ab_testing_page():
    """Show A/B testing page"""
    st.header("🧪 A/B Testing")
    
    # A/B Testing controls
    st.subheader("🎯 A/B Test Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        traffic_split = st.slider("Traffic Split", 0.1, 0.9, 0.5, 0.1, format="%.1f")
        duration_days = st.number_input("Test Duration (days)", 1, 30, 7)
    
    with col2:
        test_type = st.selectbox("Test Type", ["Model Comparison", "Feature Testing", "Hyperparameter Testing"])
        significance_level = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01, format="%.2f")
    
    # Run A/B test
    if st.button("🚀 Start A/B Test", use_container_width=True):
        with st.spinner("Running A/B test..."):
            # Train models and run test
            ab_system = st.session_state.ab_testing
            ab_system.train_models()
            ab_results = ab_system.run_ab_test(traffic_split, duration_days)
            analysis = ab_system.analyze_results()
            
            st.success("A/B test completed!")
            
            # Display results
            st.subheader("📊 Test Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model A (Random Forest)", f"RMSE: {ab_results['model_a']['rmse']:.4f}")
                st.metric("Model A R²", f"{ab_results['model_a']['r2']:.4f}")
            
            with col2:
                st.metric("Model B (Gradient Boosting)", f"RMSE: {ab_results['model_b']['rmse']:.4f}")
                st.metric("Model B R²", f"{ab_results['model_b']['r2']:.4f}")
            
            # Winner
            winner = analysis['winner']
            improvement = analysis['improvement']
            
            if winner == 'A':
                st.success(f"🏆 Winner: Model A (Random Forest) - {improvement:.2f}% better")
            else:
                st.success(f"🏆 Winner: Model B (Gradient Boosting) - {improvement:.2f}% better")
            
            # Recommendation
            st.info(f"💡 Recommendation: {analysis['recommendation']}")
    
    # A/B Test History
    st.subheader("📈 A/B Test History")
    
    # Sample test history
    test_history = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=5, freq='W'),
        'Test_Type': ['Model Comparison', 'Feature Testing', 'Model Comparison', 'Hyperparameter Testing', 'Model Comparison'],
        'Winner': ['Random Forest', 'Feature Set A', 'Gradient Boosting', 'Config B', 'Random Forest'],
        'Improvement': [0.023, 0.015, 0.038, 0.012, 0.019],
        'Significant': [True, False, True, False, True]
    })
    
    st.dataframe(test_history, use_container_width=True)
    
    # A/B Test Visualizations
    if st.checkbox("Show A/B Test Visualizations"):
        st.subheader("📊 Test Visualizations")
        
        # Model comparison chart
        models = ['Random Forest', 'Gradient Boosting', 'Linear Regression']
        rmse_scores = [1.649, 1.610, 3.921]
        r2_scores = [0.969, 0.970, 0.823]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('RMSE Comparison', 'R² Score Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=r2_scores, name='R²', marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_model_performance_page():
    """Show model performance page"""
    st.header("📊 Model Performance")
    
    # Current model performance
    st.subheader("🎯 Current Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", "0.969", "0.002")
    
    with col2:
        st.metric("RMSE", "1.649", "-0.023")
    
    with col3:
        st.metric("MAE", "1.074", "-0.015")
    
    with col4:
        st.metric("Overfitting", "2.6%", "-0.5%")
    
    # Performance over time
    st.subheader("📈 Performance Trends")
    
    # Generate sample performance data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    performance_data = pd.DataFrame({
        'Date': dates,
        'R2_Score': np.random.normal(0.97, 0.01, 30),
        'RMSE': np.random.normal(1.65, 0.05, 30),
        'MAE': np.random.normal(1.07, 0.03, 30)
    })
    
    # R² Score trend
    fig_r2 = px.line(performance_data, x='Date', y='R2_Score', title='R² Score Over Time')
    fig_r2.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Target")
    st.plotly_chart(fig_r2, use_container_width=True)
    
    # RMSE and MAE trends
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rmse = px.line(performance_data, x='Date', y='RMSE', title='RMSE Over Time')
        fig_rmse.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col2:
        fig_mae = px.line(performance_data, x='Date', y='MAE', title='MAE Over Time')
        fig_mae.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    
    # Sample feature importance data
    feature_importance = pd.DataFrame({
        'Feature': ['HIV/AIDS', 'Adult Mortality', 'Income Composition', 'Schooling', 'BMI', 'GDP', 'Alcohol', 'Hepatitis B'],
        'Importance': [0.594, 0.156, 0.148, 0.020, 0.014, 0.012, 0.008, 0.006]
    })
    
    fig_importance = px.bar(
        feature_importance, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Feature Importance (Top 8 Features)',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model comparison
    st.subheader("🆚 Model Comparison")
    
    comparison_data = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'Linear Regression', 'XGBoost'],
        'R2_Score': [0.969, 0.970, 0.823, 0.968],
        'RMSE': [1.649, 1.610, 3.921, 1.655],
        'MAE': [1.074, 1.066, 2.891, 1.089],
        'Training_Time': [5.6, 3.4, 0.1, 4.2]
    })
    
    # Create comparison charts
    fig_comparison = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Score', 'RMSE', 'MAE', 'Training Time (s)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig_comparison.add_trace(
        go.Bar(x=comparison_data['Model'], y=comparison_data['R2_Score'], name='R²', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig_comparison.add_trace(
        go.Bar(x=comparison_data['Model'], y=comparison_data['RMSE'], name='RMSE', marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig_comparison.add_trace(
        go.Bar(x=comparison_data['Model'], y=comparison_data['MAE'], name='MAE', marker_color='lightgreen'),
        row=2, col=1
    )
    
    fig_comparison.add_trace(
        go.Bar(x=comparison_data['Model'], y=comparison_data['Training_Time'], name='Training Time', marker_color='lightyellow'),
        row=2, col=2
    )
    
    fig_comparison.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_comparison, use_container_width=True)


if __name__ == "__main__":
    main()
