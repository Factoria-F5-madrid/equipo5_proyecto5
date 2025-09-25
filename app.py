import streamlit as st
import pandas as pd
import numpy as np
from pipeline import LifeExpectancyPipeline
from datetime import datetime
import plotly.express as px
import os
import sys

# Añadir backend/src al path de Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

# --- MLOps modules ---
from data_drift_monitor import DataDriftMonitor, create_streamlit_drift_dashboard
from model_auto_replacement import ModelAutoReplacement, create_streamlit_auto_replacement_dashboard
from ab_testing import ABTestingSystem

# --- Feedback integration with error handling ---
try:
    # Ahora debería poder importar directamente desde feedback_utils
    import feedback_utils
    save_feedback = feedback_utils.save_feedback
    FEEDBACK_ENABLED = True
    print("✅ Feedback module loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import feedback module: {e}")
    FEEDBACK_ENABLED = False
    
    # Función dummy para evitar errores
    def save_feedback(input_data: dict, prediction: float, feedback_text: str = None):
        raise Exception("Feedback module not available")
except Exception as e:
    print(f"Error loading feedback module: {e}")
    FEEDBACK_ENABLED = False
    
    def save_feedback(input_data: dict, prediction: float, feedback_text: str = None):
        raise Exception(f"Feedback module error: {e}")

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

# Show feedback status in sidebar
if not FEEDBACK_ENABLED:
    st.sidebar.warning("⚠️ Feedback module not available")
else:
    st.sidebar.success("✅ Feedback module loaded")

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
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Data Drift", "Active" if drift_monitor else "Offline")
    col2.metric("Model Replacement", "Active" if auto_replacement else "Offline")
    col3.metric("A/B Testing", "Ready" if ab_testing else "Offline")
    col4.metric("Model Status", "Loaded" if pipeline.model else "Not Available")
    col5.metric("Feedback System", "Active" if FEEDBACK_ENABLED else "Offline")

    st.subheader("📊 System Health")
    st.success("✅ Core prediction pipeline operational" if pipeline.model else "⚠️ Core prediction pipeline not ready")
    st.success("✅ Data drift monitoring operational" if drift_monitor else "⚠️ Offline")
    st.success("✅ Auto model replacement operational" if auto_replacement else "⚠️ Offline")
    st.success("✅ A/B testing system operational" if ab_testing else "⚠️ Offline")
    st.success("✅ Feedback system operational" if FEEDBACK_ENABLED else "⚠️ Offline")

# --- Prediction ---
def show_prediction_page():
    st.header("🧬 Predict Life Expectancy")
    if pipeline.model is None or pipeline.preprocessor is None:
        st.warning("No se encontraron los archivos necesarios. Ejecuta el entrenamiento primero.")
        st.stop()
    st.success("✅ Modelo y preprocesador cargados correctamente.")

    DATA_PATH = "data/clean_data.csv"
    try:
        df_clean = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"No se pudo cargar {DATA_PATH}: {e}")
        st.stop()

    feature_cols = [c for c in df_clean.columns if c not in ['country','year','status','life_expectancy']]

    st.subheader("📝 Input Parameters")
    user_input = {}
    col1, col2 = st.columns(2)
    with col1:
        for f in feature_cols[:len(feature_cols)//2]:
            user_input[f] = st.slider(
                f.replace('_',' ').title(), 
                float(df_clean[f].min()), 
                float(df_clean[f].max()), 
                float(df_clean[f].mean())
            )
    with col2:
        for f in feature_cols[len(feature_cols)//2:]:
            user_input[f] = st.slider(
                f.replace('_',' ').title(), 
                float(df_clean[f].min()), 
                float(df_clean[f].max()), 
                float(df_clean[f].mean())
            )

    user_input['country'] = st.selectbox("País", options=df_clean['country'].unique())
    user_input['status'] = st.selectbox("Status", options=df_clean['status'].unique())
    user_input['year'] = st.number_input(
        "Año", 
        int(df_clean['year'].min()), 
        int(df_clean['year'].max()), 
        int(df_clean['year'].max())
    )

    if st.button("🔮 Predecir Esperanza de Vida"):
        try:
            prediction = pipeline.predict(user_input)
            st.success(f"✅ Predicción completada: **{prediction:.2f} años**")

            # Métricas de comparación
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicción", f"{prediction:.2f} años")
            
            country_avg = df_clean[df_clean['country']==user_input['country']]['life_expectancy'].mean()
            col2.metric("vs Promedio País", f"{prediction - country_avg:+.2f} años")
            
            global_avg = df_clean['life_expectancy'].mean()
            col3.metric("vs Promedio Global", f"{prediction - global_avg:+.2f} años")

            # --- Feedback section ---
            if FEEDBACK_ENABLED:
                st.subheader("📝 Feedback del Usuario")
                feedback_text = st.text_area(
                    "¿Qué opinas de esta predicción?", 
                    placeholder="Comparte tu opinión sobre la predicción...",
                    key="feedback_text"
                )
                
                if st.button("💾 Guardar Feedback", key="save_feedback_btn"):
                    if feedback_text.strip():  # Solo guardar si hay texto
                        try:
                            save_feedback(user_input, float(prediction), feedback_text)
                            st.success("✅ Feedback guardado en la base de datos")
                        except Exception as e:
                            st.error(f"❌ No se pudo guardar el feedback: {e}")
                            st.code(f"Error details: {str(e)}")
                    else:
                        st.warning("⚠️ Por favor, escribe algo en el campo de feedback antes de guardarlo")
            else:
                st.info("ℹ️ La funcionalidad de feedback no está disponible en este momento")
                st.caption("Para habilitar el feedback, verifica que backend/src/feedback_utils.py y db_connect.py estén configurados correctamente")

        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")
            import traceback
            st.code(traceback.format_exc())

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
                results_df = pd.DataFrame({
                    'Model': [ab_results['model_a']['name'], ab_results['model_b']['name']],
                    'RMSE': [ab_results['model_a']['rmse'], ab_results['model_b']['rmse']],
                    'MAE': [ab_results['model_a']['mae'], ab_results['model_b']['mae']],
                    'R²': [ab_results['model_a']['r2'], ab_results['model_b']['r2']],
                    'Samples': [ab_results['model_a']['samples'], ab_results['model_b']['samples']]
                })
                st.table(results_df)

            except Exception as e:
                st.error(f"❌ Error running A/B Test: {e}")
                import traceback
                st.code(traceback.format_exc())

# --- Model Performance ---
def show_model_performance_page():
    st.header("📊 Model Performance")
    
    st.subheader("🎯 Current Model Metrics")
    if pipeline.current_model_info:
        perf = pipeline.current_model_info.get('performance', {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{perf.get('r2', 0):.3f}")
        col2.metric("RMSE", f"{perf.get('rmse', 0):.3f}")
        col3.metric("MAE", f"{perf.get('mae', 0):.3f}")
        
        r2 = perf.get('r2', 0)
        overfit = abs(r2-0.95)*100 if r2 > 0.95 else 0
        col4.metric("Overfitting Est.", f"{overfit:.1f}%")
        
        # Model information
        st.subheader("ℹ️ Model Information")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.info(f"**Model Type:** {pipeline.current_model_info.get('name', 'Unknown')}")
        with info_col2:
            trained_at = pipeline.current_model_info.get('trained_at', 'Unknown')
            if trained_at != 'Unknown':
                st.info(f"**Trained At:** {trained_at.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info(f"**Trained At:** {trained_at}")
    else:
        st.warning("No performance metrics available")

    st.subheader("📈 Performance Over Time")
    if st.button("Generate Sample Performance Trend"):
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        r2_scores = np.random.normal(0.969, 0.02, len(dates))
        r2_scores = np.clip(r2_scores, 0.9, 1.0)  # Ensure realistic values
        
        fig = px.line(
            x=dates, 
            y=r2_scores, 
            title="R² Score Over Time",
            labels={'x': 'Date', 'y': 'R² Score'}
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="R² Score",
            yaxis=dict(range=[0.9, 1.0])
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Run app ---
if __name__ == "__main__":
    main()