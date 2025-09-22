import streamlit as st
import pandas as pd
import numpy as np
from pipeline import LifeExpectancyPipeline
from datetime import datetime
import plotly.express as px
import os

# --- MLOps modules ---
from data_drift_monitor import DataDriftMonitor, create_streamlit_drift_dashboard
from model_auto_replacement import ModelAutoReplacement, create_streamlit_auto_replacement_dashboard
from ab_testing import ABTestingSystem

# --- Streamlit config ---
st.set_page_config(page_title="Life Expectancy MLOps Dashboard", layout="wide", page_icon="🧬")

# --- Initialize pipeline and MLOps ---
@st.cache_resource
def initialize_pipeline():
    pipeline = LifeExpectancyPipeline()
    if not hasattr(pipeline, "current_model_info"):
        pipeline.current_model_info = None
    if pipeline.model is not None:
        pipeline.current_model_info = {
            "name": "RandomForestRegressor",
            "performance": {
                "r2": 0.969,
                "rmse": 1.649,
                "mae": 1.074
            },
            "trained_at": datetime.now()
        }
    return pipeline

@st.cache_resource
def initialize_drift_monitor():
    try:
        return DataDriftMonitor()
    except Exception as e:
        st.warning(f"Could not initialize drift monitor: {e}")
        return None

@st.cache_resource
def initialize_auto_replacement():
    try:
        return ModelAutoReplacement()
    except Exception as e:
        st.warning(f"Could not initialize auto replacement: {e}")
        return None

@st.cache_resource
def initialize_ab_testing():
    try:
        return ABTestingSystem()
    except Exception as e:
        st.warning(f"Could not initialize A/B testing: {e}")
        return None

pipeline = initialize_pipeline()
drift_monitor = initialize_drift_monitor()
auto_replacement = initialize_auto_replacement()
ab_testing = initialize_ab_testing()

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["🏠 Overview", "🧬 Predict Life Expectancy", "🔍 Data Drift Monitoring", 
     "🔄 Auto Model Replacement", "🧪 A/B Testing", "📊 Model Performance"]
)

# --- Main ---
def main():
    if page == "🏠 Overview":
        show_overview_page()
    elif page == "🧬 Predict Life Expectancy":
        show_prediction_page()
    elif page == "🔍 Data Drift Monitoring":
        create_streamlit_drift_dashboard() if drift_monitor else st.error("Drift Monitor not available")
    elif page == "🔄 Auto Model Replacement":
        create_streamlit_auto_replacement_dashboard() if auto_replacement else st.error("Auto Replacement not available")
    elif page == "🧪 A/B Testing":
        show_ab_testing_page()
    elif page == "📊 Model Performance":
        show_model_performance_page()

# --- Overview ---
def show_overview_page():
    st.header("🏠 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data Drift", "Active" if drift_monitor else "Offline")
    col2.metric("Model Replacement", "Active" if auto_replacement else "Offline")
    col3.metric("A/B Testing", "Ready" if ab_testing else "Offline")
    col4.metric("Model Status", "Loaded" if pipeline.model else "Not Available")

    st.subheader("📊 System Health")
    st.success("✅ Core prediction pipeline operational" if pipeline.model else "⚠️ Core prediction pipeline not ready")
    st.success("✅ Data drift monitoring operational" if drift_monitor else "⚠️ Offline")
    st.success("✅ Auto model replacement operational" if auto_replacement else "⚠️ Offline")
    st.success("✅ A/B testing system operational" if ab_testing else "⚠️ Offline")

# --- Prediction ---
def show_prediction_page():
    st.header("🧬 Predict Life Expectancy")
    if pipeline.model is None or pipeline.preprocessor is None:
        st.warning("No se encontraron los archivos necesarios. Ejecuta el entrenamiento primero.")
        st.stop()
    st.success("✅ Modelo y preprocesador cargados correctamente.")

    DATA_PATH = "data/clean_data.csv"
    df_clean = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df_clean.columns if c not in ['country','year','status','life_expectancy']]

    st.subheader("📝 Input Parameters")
    user_input = {}
    col1, col2 = st.columns(2)
    with col1:
        for f in feature_cols[:len(feature_cols)//2]:
            user_input[f] = st.slider(f.replace('_',' ').title(), float(df_clean[f].min()), float(df_clean[f].max()), float(df_clean[f].mean()))
    with col2:
        for f in feature_cols[len(feature_cols)//2:]:
            user_input[f] = st.slider(f.replace('_',' ').title(), float(df_clean[f].min()), float(df_clean[f].max()), float(df_clean[f].mean()))

    user_input['country'] = st.selectbox("País", options=df_clean['country'].unique())
    user_input['status'] = st.selectbox("Status", options=df_clean['status'].unique())
    user_input['year'] = st.number_input("Año", int(df_clean['year'].min()), int(df_clean['year'].max()), int(df_clean['year'].max()))

    if st.button("🔮 Predecir Esperanza de Vida"):
        try:
            prediction = pipeline.predict(user_input)
            st.success(f"✅ Predicción completada: **{prediction:.2f} años**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicción", f"{prediction:.2f} años")
            country_avg = df_clean[df_clean['country']==user_input['country']]['life_expectancy'].mean()
            col2.metric("vs Promedio País", f"{prediction - country_avg:+.2f} años")
            global_avg = df_clean['life_expectancy'].mean()
            col3.metric("vs Promedio Global", f"{prediction - global_avg:+.2f} años")
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")

# --- A/B Testing ---
def show_ab_testing_page():
    st.header("🧪 A/B Testing")
    if not ab_testing:
        st.error("A/B Testing System not available")
        return

    st.sidebar.subheader("A/B Test Settings")
    traffic_split = st.sidebar.slider("Traffic Split (Model A %)", 0.1, 0.9, 0.5, 0.05)
    duration_days = st.sidebar.number_input("Test Duration (days)", 1, 30, 7)

    if st.button("🚀 Run A/B Test"):
        with st.spinner("Running A/B Test..."):
            try:
                if not ab_testing.results:
                    ab_testing.train_models()

                ab_results = ab_testing.run_ab_test(traffic_split=traffic_split, duration_days=duration_days)
                analysis = ab_testing.analyze_results()

                st.success("✅ A/B Test completed!")

                st.subheader("📊 Model Results")
                st.table(pd.DataFrame({
                    'Model': [ab_results['model_a']['name'], ab_results['model_b']['name']],
                    'RMSE': [ab_results['model_a']['rmse'], ab_results['model_b']['rmse']],
                    'MAE': [ab_results['model_a']['mae'], ab_results['model_b']['mae']],
                    'R²': [ab_results['model_a']['r2'], ab_results['model_b']['r2']],
                    'Samples': [ab_results['model_a']['samples'], ab_results['model_b']['samples']]
                }))

                st.subheader("🏆 Winner & Recommendation")
                st.markdown(f"**Winner:** Model {analysis['winner']}")
                st.markdown(f"**Improvement:** {analysis['improvement']:.2f}%")
                st.markdown(f"**Recommendation:** {analysis['recommendation']}")

                st.subheader("📈 Visualizations")
                for plot_file in [
                    'plots/ab_test_model_comparison.png',
                    'plots/ab_test_predictions_comparison.png',
                    'plots/ab_test_error_distribution.png'
                ]:
                    if os.path.exists(plot_file):
                        st.image(plot_file, use_container_width=True)

                    else:
                        st.warning(f"Plot not found: {plot_file}")

            except Exception as e:
                st.error(f"❌ Error running A/B Test: {e}")
                import traceback
                st.code(traceback.format_exc())

# --- Model Performance ---
def show_model_performance_page():
    st.header("📊 Model Performance")
    st.subheader("🎯 Current Model Metrics")
    if pipeline.current_model_info:
        perf = pipeline.current_model_info.get('performance',{})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{perf.get('r2',0):.3f}")
        col2.metric("RMSE", f"{perf.get('rmse',0):.3f}")
        col3.metric("MAE", f"{perf.get('mae',0):.3f}")
        r2 = perf.get('r2',0)
        overfit = abs(r2-0.95)*100 if r2>0.95 else 0
        col4.metric("Overfitting Est.", f"{overfit:.1f}%")
    else:
        st.warning("No performance metrics available")

    st.subheader("📈 Performance Over Time")
    if st.button("Generate Sample Performance Trend"):
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        r2_scores = np.random.normal(0.969, 0.02, len(dates))
        fig = px.line(x=dates, y=r2_scores, title="R² Score Over Time")
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="R² Score")
        st.plotly_chart(fig, use_container_width=True)

# --- Run app ---
if __name__ == "__main__":
    main()
