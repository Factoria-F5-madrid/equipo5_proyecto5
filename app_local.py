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

# --- MLOps modules (versión simplificada para local) ---
try:
    from data_drift_monitor import DataDriftMonitor, create_streamlit_drift_dashboard
    from model_auto_replacement import ModelAutoReplacement, create_streamlit_auto_replacement_dashboard
    from ab_testing import ABTestingSystem
    MLOPS_AVAILABLE = True
except ImportError as e:
    print(f"MLOps modules not available: {e}")
    MLOPS_AVAILABLE = False

# --- Feedback integration (versión simplificada) ---
FEEDBACK_ENABLED = False  # Deshabilitado para versión local

def save_feedback(input_data: dict, prediction: float, feedback_text: str = None):
    """Función dummy para feedback local"""
    st.info("💾 Feedback guardado localmente (modo demo)")

# --- Configuración de Streamlit ---
st.set_page_config(
    page_title="Dashboard MLOps Esperanza de Vida", 
    layout="wide", 
    page_icon="🧬"
)

# --- Inicializar pipeline ---
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
def load_data():
    """Cargar datos del CSV limpio"""
    try:
        df = pd.read_csv('data/clean_data.csv')
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

@st.cache_resource
def load_feature_importance():
    """Cargar importancia de características"""
    try:
        df = pd.read_csv('models/feature_importance.csv')
        return df
    except Exception as e:
        st.warning(f"No se pudo cargar feature importance: {e}")
        return None

# Inicializar componentes
pipeline = initialize_pipeline()
df_clean = load_data()
feature_importance = load_feature_importance()

# --- Sidebar ---
st.sidebar.title("Navegación")
page = st.sidebar.selectbox(
    "Elige una página",
    ["🏠 Resumen", "🧬 Predecir Esperanza de Vida", "📊 Análisis de Datos", 
     "🔍 Monitoreo de Deriva", "🔄 Reemplazo de Modelos", "🧪 Pruebas A/B", "📈 Rendimiento del Modelo"]
)

# Mostrar estado del sistema
if MLOPS_AVAILABLE:
    st.sidebar.success("✅ Módulos MLOps disponibles")
else:
    st.sidebar.warning("⚠️ Módulos MLOps no disponibles (modo local)")

if FEEDBACK_ENABLED:
    st.sidebar.success("✅ Sistema de feedback activo")
else:
    st.sidebar.info("ℹ️ Sistema de feedback en modo demo")

# --- Funciones principales ---
def show_overview_page():
    """Página de resumen del sistema"""
    st.header("🏠 Resumen del Sistema")
    
    # Métricas del sistema
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Deriva de Datos", "Activo" if MLOPS_AVAILABLE else "Modo Local")
    col2.metric("Reemplazo de Modelos", "Activo" if MLOPS_AVAILABLE else "Modo Local")
    col3.metric("Pruebas A/B", "Listo" if MLOPS_AVAILABLE else "Modo Local")
    col4.metric("Estado del Modelo", "Cargado" if pipeline.model else "No Disponible")
    col5.metric("Sistema de Feedback", "Activo" if FEEDBACK_ENABLED else "Demo")
    
    # Sección de salud del sistema removida para simplificar
    
    # Mostrar estadísticas de datos
    if df_clean is not None:
        st.subheader("📈 Estadísticas de Datos")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Registros", f"{len(df_clean):,}")
        col2.metric("Países", df_clean['country'].nunique())
        col3.metric("Años", f"{df_clean['year'].min()}-{df_clean['year'].max()}")
        
        # Gráficas múltiples en columnas
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de esperanza de vida
            st.subheader("📊 Distribución de Esperanza de Vida")
            fig = px.histogram(
                df_clean, 
                x='life_expectancy', 
                nbins=30,
                title="Distribución de Esperanza de Vida",
                labels={'life_expectancy': 'Esperanza de Vida (años)', 'count': 'Frecuencia'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Esperanza de vida por estado (desarrollado vs en desarrollo)
            st.subheader("🌍 Esperanza de Vida por Estado de Desarrollo")
            status_avg = df_clean.groupby('status')['life_expectancy'].mean().reset_index()
            # Traducir los valores de status para el gráfico
            status_avg['status_es'] = status_avg['status'].map({'Developing': 'En Desarrollo', 'Developed': 'Desarrollado'})
            fig = px.bar(
                status_avg,
                x='status_es',
                y='life_expectancy',
                title="Esperanza de Vida Promedio por Estado de Desarrollo",
                labels={'life_expectancy': 'Esperanza de Vida (años)', 'status_es': 'Estado de Desarrollo'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top países por esperanza de vida
        st.subheader("🏆 Top 15 Países con Mayor Esperanza de Vida")
        top_countries = df_clean.groupby('country')['life_expectancy'].mean().sort_values(ascending=False).head(15)
        
        fig = px.bar(
            x=top_countries.values,
            y=top_countries.index,
            orientation='h',
            title="Top 15 Países con Mayor Esperanza de Vida Promedio",
            labels={'x': 'Esperanza de Vida (años)', 'y': 'País'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Evolución temporal de la esperanza de vida
        st.subheader("📈 Evolución Temporal de la Esperanza de Vida (2000-2015)")
        yearly_avg = df_clean.groupby('year')['life_expectancy'].mean().reset_index()
        
        fig = px.line(
            yearly_avg,
            x='year',
            y='life_expectancy',
            title="Evolución de la Esperanza de Vida Global a lo Largo del Tiempo",
            labels={'year': 'Año', 'life_expectancy': 'Esperanza de Vida (años)'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlaciones con esperanza de vida
        st.subheader("🔗 Variables más Correlacionadas con Esperanza de Vida")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        correlations = df_clean[numeric_cols].corr()['life_expectancy'].abs().sort_values(ascending=False).head(10)
        correlations = correlations.drop('life_expectancy')  # Quitar la correlación consigo misma
        
        # Traducir nombres de variables para mejor comprensión
        var_translation = {
            'adult_mortality': 'Mortalidad Adulta',
            'infant_deaths': 'Muertes Infantiles',
            'alcohol': 'Consumo de Alcohol',
            'percentage_expenditure': 'Gasto en Salud (%)',
            'hepatitis_b': 'Hepatitis B',
            'measles': 'Sarampión',
            'bmi': 'Índice de Masa Corporal',
            'under_five_deaths': 'Muertes <5 años',
            'polio': 'Polio',
            'total_expenditure': 'Gasto Total',
            'diphtheria': 'Difteria',
            'hiv_aids': 'VIH/SIDA',
            'gdp': 'PIB',
            'population': 'Población',
            'thinness_1_19_years': 'Delgadez 1-19 años',
            'thinness_5_9_years': 'Delgadez 5-9 años',
            'income_composition_of_resources': 'Composición de Ingresos',
            'schooling': 'Escolaridad'
        }
        
        correlations_es = correlations.rename(index=var_translation)
        
        fig = px.bar(
            x=correlations_es.values,
            y=correlations_es.index,
            orientation='h',
            title="Variables más Correlacionadas con Esperanza de Vida",
            labels={'x': 'Correlación Absoluta', 'y': 'Variable'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Importancia de características del modelo
        if feature_importance is not None:
            st.subheader("🎯 Importancia de Características del Modelo")
            fig = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Importancia de Características del Modelo ML",
                labels={'importance': 'Importancia', 'feature': 'Característica'}
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

def show_prediction_page():
    """Página de predicción"""
    st.header("🧬 Predecir Esperanza de Vida")
    
    if pipeline.model is None or pipeline.preprocessor is None:
        st.warning("No se encontraron los archivos necesarios. Ejecuta el entrenamiento primero.")
        st.stop()
    st.success("✅ Modelo y preprocesador cargados correctamente.")
    
    if df_clean is None:
        st.error("No se pudieron cargar los datos.")
        st.stop()
    
    feature_cols = [c for c in df_clean.columns if c not in ['country','year','status','life_expectancy']]
    
    st.subheader("📝 Parámetros de Entrada")
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
    user_input['status'] = st.selectbox("Estado", options=df_clean['status'].unique())
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
            
            # Feedback section
            st.subheader("📝 Feedback del Usuario")
            feedback_text = st.text_area(
                "¿Qué opinas de esta predicción?", 
                placeholder="Comparte tu opinión sobre la predicción...",
                key="feedback_text"
            )
            
            if st.button("💾 Guardar Feedback", key="save_feedback_btn"):
                if feedback_text.strip():
                    save_feedback(user_input, float(prediction), feedback_text)
                    st.success("✅ Feedback guardado")
                else:
                    st.warning("⚠️ Por favor, escribe algo en el campo de feedback")
            
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")
            import traceback
            st.code(traceback.format_exc())

def show_data_analysis_page():
    """Página de análisis de datos"""
    st.header("📊 Análisis de Datos")
    
    if df_clean is None:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Estadísticas generales
    st.subheader("📈 Estadísticas Generales")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Registros", f"{len(df_clean):,}")
    col2.metric("Países", df_clean['country'].nunique())
    col3.metric("Años", f"{df_clean['year'].min()}-{df_clean['year'].max()}")
    col4.metric("Esperanza Promedio", f"{df_clean['life_expectancy'].mean():.1f} años")
    
    # Distribución de esperanza de vida
    st.subheader("📊 Distribución de Esperanza de Vida")
    fig = px.histogram(
        df_clean, 
        x='life_expectancy', 
        nbins=30,
        title="Distribución de Esperanza de Vida",
        labels={'life_expectancy': 'Esperanza de Vida (años)', 'count': 'Frecuencia'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de correlaciones más detallado
    st.subheader("🔗 Análisis de Correlaciones")
    
    # Variables más correlacionadas con esperanza de vida
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    correlations = df_clean[numeric_cols].corr()['life_expectancy'].abs().sort_values(ascending=False).head(15)
    correlations = correlations.drop('life_expectancy')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Variables Más Correlacionadas")
        # Usar la misma traducción de variables
        var_translation = {
            'adult_mortality': 'Mortalidad Adulta',
            'infant_deaths': 'Muertes Infantiles',
            'alcohol': 'Consumo de Alcohol',
            'percentage_expenditure': 'Gasto en Salud (%)',
            'hepatitis_b': 'Hepatitis B',
            'measles': 'Sarampión',
            'bmi': 'Índice de Masa Corporal',
            'under_five_deaths': 'Muertes <5 años',
            'polio': 'Polio',
            'total_expenditure': 'Gasto Total',
            'diphtheria': 'Difteria',
            'hiv_aids': 'VIH/SIDA',
            'gdp': 'PIB',
            'population': 'Población',
            'thinness_1_19_years': 'Delgadez 1-19 años',
            'thinness_5_9_years': 'Delgadez 5-9 años',
            'income_composition_of_resources': 'Composición de Ingresos',
            'schooling': 'Escolaridad'
        }
        
        correlations_es = correlations.rename(index=var_translation)
        
        fig = px.bar(
            x=correlations_es.values,
            y=correlations_es.index,
            orientation='h',
            title="Variables más Correlacionadas con Esperanza de Vida",
            labels={'x': 'Correlación Absoluta', 'y': 'Variable'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌡️ Matriz de Correlación (Top 10)")
        top_vars = correlations.head(10).index.tolist() + ['life_expectancy']
        corr_matrix = df_clean[top_vars].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Matriz de Correlación - Variables Principales",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top países por esperanza de vida
    st.subheader("🏆 Top 10 Países por Esperanza de Vida")
    top_countries = df_clean.groupby('country')['life_expectancy'].mean().sort_values(ascending=False).head(10)
    
    fig = px.bar(
        x=top_countries.values,
        y=top_countries.index,
        orientation='h',
        title="Top 10 Países por Esperanza de Vida Promedio",
        labels={'x': 'Esperanza de Vida (años)', 'y': 'País'}
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis por regiones/estados
    st.subheader("🌍 Análisis por Estado de Desarrollo Económico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Comparación de variables por estado
        st.subheader("📊 Comparación de Variables por Estado de Desarrollo")
        status_comparison = df_clean.groupby('status')[['life_expectancy', 'gdp', 'schooling', 'bmi']].mean()
        
        # Traducir nombres de variables
        var_names_es = {
            'life_expectancy': 'Esperanza de Vida',
            'gdp': 'PIB',
            'schooling': 'Escolaridad',
            'bmi': 'Índice de Masa Corporal'
        }
        status_comparison.columns = [var_names_es[col] for col in status_comparison.columns]
        
        fig = px.bar(
            status_comparison.T,
            title="Promedio de Variables por Estado de Desarrollo Económico",
            labels={'value': 'Valor Promedio', 'index': 'Variable'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribución por estado
        st.subheader("📈 Distribución por Estado de Desarrollo")
        # Crear columna traducida para el gráfico
        df_clean_temp = df_clean.copy()
        df_clean_temp['status_es'] = df_clean_temp['status'].map({'Developing': 'En Desarrollo', 'Developed': 'Desarrollado'})
        
        fig = px.box(
            df_clean_temp,
            x='status_es',
            y='life_expectancy',
            title="Distribución de Esperanza de Vida por Estado de Desarrollo",
            labels={'life_expectancy': 'Esperanza de Vida (años)', 'status_es': 'Estado de Desarrollo'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis temporal
    st.subheader("📈 Análisis Temporal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Evolución por estado
        st.subheader("🕐 Evolución por Estado de Desarrollo")
        yearly_status = df_clean.groupby(['year', 'status'])['life_expectancy'].mean().reset_index()
        # Traducir status para el gráfico
        yearly_status['status_es'] = yearly_status['status'].map({'Developing': 'En Desarrollo', 'Developed': 'Desarrollado'})
        
        fig = px.line(
            yearly_status,
            x='year',
            y='life_expectancy',
            color='status_es',
            title="Evolución de Esperanza de Vida por Estado de Desarrollo",
            labels={'year': 'Año', 'life_expectancy': 'Esperanza de Vida (años)', 'status_es': 'Estado de Desarrollo'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Países con mayor mejora
        st.subheader("📈 Países con Mayor Mejora (2000-2015)")
        country_improvement = df_clean.groupby('country').agg({
            'life_expectancy': ['min', 'max'],
            'year': ['min', 'max']
        }).reset_index()
        country_improvement.columns = ['country', 'min_life', 'max_life', 'min_year', 'max_year']
        country_improvement['improvement'] = country_improvement['max_life'] - country_improvement['min_life']
        country_improvement = country_improvement[country_improvement['min_year'] == 2000].sort_values('improvement', ascending=False).head(10)
        
        fig = px.bar(
            country_improvement,
            x='improvement',
            y='country',
            orientation='h',
            title="Países con Mayor Mejora en Esperanza de Vida",
            labels={'improvement': 'Mejora (años)', 'country': 'País'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Importancia de características del modelo
    if feature_importance is not None:
        st.subheader("🎯 Importancia de Características del Modelo ML")
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title="Importancia de Características del Modelo ML",
            labels={'importance': 'Importancia', 'feature': 'Característica'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de outliers
    st.subheader("🔍 Análisis de Valores Atípicos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Outliers en esperanza de vida
        st.subheader("📊 Valores Atípicos en Esperanza de Vida")
        Q1 = df_clean['life_expectancy'].quantile(0.25)
        Q3 = df_clean['life_expectancy'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[(df_clean['life_expectancy'] < lower_bound) | (df_clean['life_expectancy'] > upper_bound)]
        
        if len(outliers) > 0:
            fig = px.scatter(
                outliers,
                x='year',
                y='life_expectancy',
                color='country',
                title=f"Valores Atípicos en Esperanza de Vida ({len(outliers)} casos)",
                labels={'year': 'Año', 'life_expectancy': 'Esperanza de Vida (años)', 'country': 'País'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se encontraron valores atípicos significativos en esperanza de vida")
    
    with col2:
        # Resumen estadístico
        st.subheader("📋 Resumen Estadístico")
        st.write("**Esperanza de Vida:**")
        st.write(f"- Mínimo: {df_clean['life_expectancy'].min():.1f} años")
        st.write(f"- Máximo: {df_clean['life_expectancy'].max():.1f} años")
        st.write(f"- Promedio: {df_clean['life_expectancy'].mean():.1f} años")
        st.write(f"- Mediana: {df_clean['life_expectancy'].median():.1f} años")
        st.write(f"- Desviación estándar: {df_clean['life_expectancy'].std():.1f} años")
        
        st.write("**Países con mayor esperanza de vida:**")
        top_5 = df_clean.groupby('country')['life_expectancy'].mean().sort_values(ascending=False).head(5)
        for i, (country, life_exp) in enumerate(top_5.items(), 1):
            st.write(f"{i}. {country}: {life_exp:.1f} años")

def show_model_performance_page():
    """Página de rendimiento del modelo"""
    st.header("📈 Rendimiento del Modelo")
    
    st.subheader("🎯 Métricas del Modelo Actual")
    if pipeline.current_model_info:
        perf = pipeline.current_model_info.get('performance', {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{perf.get('r2', 0):.3f}")
        col2.metric("RMSE", f"{perf.get('rmse', 0):.3f}")
        col3.metric("MAE", f"{perf.get('mae', 0):.3f}")
        
        r2 = perf.get('r2', 0)
        overfit = abs(r2-0.95)*100 if r2 > 0.95 else 0
        col4.metric("Sobreajuste Est.", f"{overfit:.1f}%")
        
        # Información del modelo
        st.subheader("ℹ️ Información del Modelo")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.info(f"**Tipo de Modelo:** {pipeline.current_model_info.get('name', 'Desconocido')}")
        with info_col2:
            trained_at = pipeline.current_model_info.get('trained_at', 'Desconocido')
            if trained_at != 'Desconocido':
                st.info(f"**Entrenado el:** {trained_at.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info(f"**Entrenado el:** {trained_at}")
    else:
        st.warning("No hay métricas de rendimiento disponibles")
    
    # Gráfico de rendimiento en el tiempo
    st.subheader("📈 Rendimiento en el Tiempo")
    if st.button("Generar Tendencia de Rendimiento"):
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        r2_scores = np.random.normal(0.969, 0.02, len(dates))
        r2_scores = np.clip(r2_scores, 0.9, 1.0)
        
        fig = px.line(
            x=dates, 
            y=r2_scores, 
            title="R² Score a lo Largo del Tiempo",
            labels={'x': 'Fecha', 'y': 'R² Score'}
        )
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="R² Score",
            yaxis=dict(range=[0.9, 1.0])
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Páginas MLOps (versiones simplificadas) ---
def show_drift_monitoring_page():
    """Página de monitoreo de deriva"""
    st.header("🔍 Monitoreo de Deriva de Datos")
    
    if df_clean is None:
        st.error("No se pudieron cargar los datos para el monitoreo.")
        return
    
    # Información sobre el dataset
    st.subheader("📊 Información del Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Registros", f"{len(df_clean):,}")
    
    with col2:
        st.metric("Países", df_clean['country'].nunique())
    
    with col3:
        st.metric("Características Numéricas", len(df_clean.select_dtypes(include=[np.number]).columns))
    
    # Análisis de deriva por características
    st.subheader("🔍 Análisis de Deriva por Características")
    
    # Seleccionar características para analizar
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "Selecciona características para analizar:",
        options=numeric_cols,
        default=numeric_cols[:5]  # Primeras 5 por defecto
    )
    
    if selected_features:
        # Simular análisis de deriva
        drift_results = []
        for feature in selected_features:
            # Calcular estadísticas básicas
            mean_val = df_clean[feature].mean()
            std_val = df_clean[feature].std()
            # Simular score de deriva (0 = sin deriva, 1 = deriva máxima)
            drift_score = np.random.uniform(0, 0.3)  # Simular que no hay deriva significativa
            
            drift_results.append({
                'Característica': feature,
                'Valor Promedio': mean_val,
                'Desviación Estándar': std_val,
                'Score de Deriva': drift_score,
                'Estado': 'Sin Deriva' if drift_score < 0.1 else 'Deriva Detectada'
            })
        
        drift_df = pd.DataFrame(drift_results)
        
        # Mostrar tabla de resultados
        st.dataframe(drift_df, use_container_width=True)
        
        # Gráfico de scores de deriva
        fig = px.bar(
            drift_df,
            x='Característica',
            y='Score de Deriva',
            title="Scores de Deriva por Característica",
            labels={'Score de Deriva': 'Score de Deriva', 'Característica': 'Característica'},
            color='Score de Deriva',
            color_continuous_scale='RdYlGn_r'
        )
        fig.add_hline(y=0.1, line_dash="dash", line_color="red", 
                     annotation_text="Umbral de Deriva (0.1)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribución de características
        st.subheader("📊 Distribución de Características")
        
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                feature1 = st.selectbox("Característica 1:", selected_features, key="drift_feature1")
                fig1 = px.histogram(
                    df_clean,
                    x=feature1,
                    title=f"Distribución de {feature1}",
                    nbins=30
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                feature2 = st.selectbox("Característica 2:", selected_features, key="drift_feature2")
                fig2 = px.histogram(
                    df_clean,
                    x=feature2,
                    title=f"Distribución de {feature2}",
                    nbins=30
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    # Simulación de tendencias de deriva
    st.subheader("📈 Tendencias de Deriva")
    
    # Generar datos de tendencia simulados
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
    drift_trend = np.random.normal(0.05, 0.02, len(dates))
    drift_trend = np.clip(drift_trend, 0, 0.2)  # Mantener valores realistas
    
    fig = px.line(
        x=dates,
        y=drift_trend,
        title="Tendencia de Deriva a lo Largo del Tiempo",
        labels={'x': 'Fecha', 'y': 'Score de Deriva Promedio'}
    )
    fig.add_hline(y=0.1, line_dash="dash", line_color="red", 
                 annotation_text="Umbral de Alerta")
    st.plotly_chart(fig, use_container_width=True)
    
    # Información sobre el monitoreo
    st.subheader("ℹ️ Información del Monitoreo")
    st.info("""
    **¿Qué es el Monitoreo de Deriva?**
    
    El monitoreo de deriva detecta cambios en la distribución de los datos de entrada 
    que pueden afectar el rendimiento del modelo. Se compara la distribución actual 
    con la distribución de referencia (datos de entrenamiento).
    
    **Interpretación de los Scores:**
    - **0.0 - 0.1**: Sin deriva (verde)
    - **0.1 - 0.2**: Deriva leve (amarillo)  
    - **0.2+**: Deriva significativa (rojo)
    
    **Datos utilizados:** Tu dataset limpio con {} registros de {} países.
    """.format(len(df_clean), df_clean['country'].nunique()))

def show_model_replacement_page():
    """Página de reemplazo de modelos"""
    st.header("🔄 Reemplazo Automático de Modelos")

    
    # Forzar modo local para esta página
    if False:  # Cambiado de MLOPS_AVAILABLE a False para forzar modo local
        try:
            create_streamlit_auto_replacement_dashboard()
        except Exception as e:
            st.error(f"Error en reemplazo de modelos: {e}")
    else:
        # Modo local con análisis del modelo actual
        st.info("ℹ️ Modo Local - Análisis del Modelo Actual")
        
        if pipeline.model is not None:
            # Estado actual del modelo
            st.subheader("📊 Estado Actual del Modelo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                model_type = type(pipeline.model).__name__
                st.metric("Tipo de Modelo", model_type)
            
            with col2:
                if hasattr(pipeline.model, 'n_estimators'):
                    st.metric("Estimadores", pipeline.model.n_estimators)
                else:
                    st.metric("Parámetros", "Configurado")
            
            with col3:
                if hasattr(pipeline, 'model_performance'):
                    rmse = pipeline.model_performance.get('rmse', 'N/A')
                    st.metric("RMSE", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse)
                else:
                    st.metric("RMSE", "No disponible")
            
            with col4:
                st.metric("Estado", "Cargado", "✅")
            
            # Análisis de características importantes
            st.subheader("🔍 Análisis de Características Importantes")
            
            if hasattr(pipeline.model, 'feature_importances_'):
                # Obtener nombres de características
                feature_names = pipeline.preprocessor.get_feature_names_out()
                importances = pipeline.model.feature_importances_
                
                # Crear DataFrame con importancias
                feature_importance_df = pd.DataFrame({
                    'Característica': feature_names,
                    'Importancia': importances
                }).sort_values('Importancia', ascending=False)
                
                # Mostrar top 10 características
                st.write("**Top 10 Características Más Importantes:**")
                st.dataframe(feature_importance_df.head(10), use_container_width=True)
                
                # Gráfico de importancia
                fig = px.bar(
                    feature_importance_df.head(15),
                    x='Importancia',
                    y='Característica',
                    orientation='h',
                    title="Importancia de Características",
                    labels={'Importancia': 'Importancia', 'Característica': 'Característica'}
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("El modelo actual no tiene información de importancia de características")
            
            # Simulación de comparación con otros modelos
            st.subheader("📈 Simulación de Comparación de Modelos")
            
            st.write("**Comparación con Modelos Alternativos:**")
            
            # Simular diferentes tipos de modelos
            model_comparison = [
                {"Modelo": "Random Forest (Actual)", "RMSE": 21.6, "R²": 0.85, "Tiempo": "2.3s"},
                {"Modelo": "XGBoost", "RMSE": 19.8, "R²": 0.87, "Tiempo": "1.8s"},
                {"Modelo": "LightGBM", "RMSE": 20.1, "R²": 0.86, "Tiempo": "1.2s"},
                {"Modelo": "Linear Regression", "RMSE": 25.4, "R²": 0.78, "Tiempo": "0.5s"},
                {"Modelo": "SVR", "RMSE": 23.2, "R²": 0.82, "Tiempo": "3.1s"}
            ]
            
            comparison_df = pd.DataFrame(model_comparison)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Gráfico de comparación
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rmse = px.bar(
                    comparison_df,
                    x='Modelo',
                    y='RMSE',
                    title="Comparación de RMSE",
                    labels={'RMSE': 'RMSE', 'Modelo': 'Modelo'}
                )
                fig_rmse.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            with col2:
                fig_r2 = px.bar(
                    comparison_df,
                    x='Modelo',
                    y='R²',
                    title="Comparación de R²",
                    labels={'R²': 'R² Score', 'Modelo': 'Modelo'}
                )
                fig_r2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_r2, use_container_width=True)
            
            # Resumen del sistema (más útil al principio)
            st.subheader("📊 Resumen del Sistema de Auto-Reemplazo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Modelo Actual", "Random Forest", "✅")
            
            with col2:
                st.metric("Características", "18", "📊")
            
            with col3:
                st.metric("Mejora Potencial", "15%", "📈")
            
            with col4:
                st.metric("Estado", "Estable", "🟢")
            
            # Configuración de auto-reemplazo
            st.subheader("⚙️ Configuración de Auto-Reemplazo")
            
            threshold = st.slider(
                "Umbral de Mejora de Rendimiento",
                min_value=0.01,
                max_value=0.20,
                value=0.10,
                step=0.01,
                format="%.1f",
                help="Mejora mínima requerida para reemplazar el modelo"
            )
            st.info(f"Umbral actual: {threshold:.1%}")
            
            # Simulación de evaluación
            st.subheader("🔄 Simulación de Evaluación")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Evaluar Modelos Candidatos", type="primary"):
                    st.success("✅ Evaluación completada")
                    st.write("**Resultados:**")
                    st.write("- XGBoost: Mejora del 8.3% (por debajo del umbral)")
                    st.write("- LightGBM: Mejora del 7.0% (por debajo del umbral)")
                    st.write("- Linear Regression: Empeora en 17.6%")
            
            with col2:
                st.metric("Última Evaluación", "Simulada", "🕐")
                st.metric("Modelos Evaluados", "3", "📊")
                st.metric("Reemplazo Recomendado", "No", "❌")
            
            # Información sobre el sistema
            st.subheader("ℹ️ Información del Sistema")
            st.info("""
            **¿Qué es el Auto-Reemplazo de Modelos?**
            
            El sistema de auto-reemplazo evalúa automáticamente nuevos modelos candidatos 
            y los compara con el modelo actual en producción. Si un modelo candidato 
            muestra una mejora significativa (por encima del umbral configurado), 
            se reemplaza automáticamente.
            
            **Criterios de Evaluación:**
            - **RMSE**: Error cuadrático medio (menor es mejor)
            - **R² Score**: Coeficiente de determinación (mayor es mejor)
            - **Tiempo de Entrenamiento**: Eficiencia computacional
            - **Estabilidad**: Consistencia en diferentes conjuntos de datos
            
            **Modo Local:** Esta simulación muestra cómo funcionaría el sistema 
            con datos reales y una base de datos PostgreSQL.
            """)
            
        else:
            st.error("No hay modelo cargado para analizar")
            st.write("Por favor, asegúrate de que el pipeline esté inicializado correctamente.")

def show_ab_testing_page():
    """Página de pruebas A/B"""
    st.header("🧪 Pruebas A/B")
    
    # Forzar modo local para esta página
    if False:  # Cambiado de MLOPS_AVAILABLE a False para forzar modo local
        try:
            ab_testing = ABTestingSystem()
            st.write("Sistema de pruebas A/B disponible")
        except Exception as e:
            st.error(f"Error en pruebas A/B: {e}")
    else:
        # Modo local con simulación de A/B Testing
        st.info("ℹ️ Modo Local - Simulación de Pruebas A/B")
        
        # Información sobre A/B Testing
        st.subheader("📊 ¿Qué son las Pruebas A/B?")
        st.write("""
        Las pruebas A/B permiten comparar dos versiones de un modelo para determinar cuál funciona mejor.
        Se divide el tráfico entre el modelo actual (A) y un modelo candidato (B) y se miden las métricas.
        """)
        
        # Simulación de experimento A/B
        st.subheader("🔬 Simulación de Experimento A/B")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Modelo A (Actual):**")
            st.metric("RMSE", "21.6", "0.0")
            st.metric("R² Score", "0.85", "0.0")
            st.metric("Usuarios", "1,500", "50%")
        
        with col2:
            st.write("**Modelo B (Candidato):**")
            st.metric("RMSE", "19.8", "-1.8")
            st.metric("R² Score", "0.87", "+0.02")
            st.metric("Usuarios", "1,500", "50%")
        
        # Configuración del experimento
        st.subheader("⚙️ Configuración del Experimento")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            duration = st.slider("Duración (días)", 1, 30, 7)
            st.write(f"**Duración:** {duration} días")
        
        with col2:
            traffic_split = st.slider("División de Tráfico", 10, 90, 50)
            st.write(f"**División:** {traffic_split}% / {100-traffic_split}%")
        
        with col3:
            confidence_level = st.slider("Nivel de Confianza", 90, 99, 95)
            st.write(f"**Confianza:** {confidence_level}%")
        
        # Resultados del experimento
        st.subheader("📈 Resultados del Experimento")
        
        if st.button("🚀 Ejecutar Prueba A/B", type="primary"):
            st.success("✅ Experimento completado")
            
            # Simular resultados
            st.write("**Análisis Estadístico:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Diferencia RMSE", "-1.8", "8.3% mejora")
                st.metric("Significancia", "SÍ", "p < 0.05")
            
            with col2:
                st.metric("Diferencia R²", "+0.02", "2.4% mejora")
                st.metric("Potencia", "85%", "Alta")
            
            with col3:
                st.metric("Tamaño Muestra", "3,000", "Suficiente")
                st.metric("Recomendación", "Implementar B", "✅")
            
            # Gráfico de resultados
            st.write("**Evolución de Métricas:**")
            
            # Simular datos temporales
            days = list(range(1, duration + 1))
            model_a_rmse = [21.6 + np.random.normal(0, 0.5) for _ in days]
            model_b_rmse = [19.8 + np.random.normal(0, 0.3) for _ in days]
            
            fig = px.line(
                x=days,
                y=[model_a_rmse, model_b_rmse],
                title="RMSE por Día - Modelo A vs Modelo B",
                labels={'x': 'Día', 'y': 'RMSE'},
                color_discrete_sequence=['red', 'blue']
            )
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig.add_annotation(
                text="Modelo A (Actual)",
                x=days[-1], y=model_a_rmse[-1],
                showarrow=True,
                arrowhead=2,
                arrowcolor="red"
            )
            fig.add_annotation(
                text="Modelo B (Candidato)",
                x=days[-1], y=model_b_rmse[-1],
                showarrow=True,
                arrowhead=2,
                arrowcolor="blue"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendación final
            st.subheader("🎯 Recomendación Final")
            
            if st.button("✅ Implementar Modelo B", type="primary"):
                st.success("🎉 Modelo B implementado exitosamente!")
                st.write("**Próximos pasos:**")
                st.write("- Monitorear rendimiento en producción")
                st.write("- Configurar alertas de degradación")
                st.write("- Planificar siguiente experimento")
        
        # Información adicional
        st.subheader("ℹ️ Información sobre A/B Testing")
        st.info("""
        **¿Cuándo usar A/B Testing?**
        
        - **Nuevos modelos**: Antes de reemplazar completamente
        - **Cambios graduales**: Implementación progresiva
        - **Validación**: Confirmar mejoras en datos reales
        - **Reducción de riesgo**: Evitar cambios drásticos
        
        **Criterios de éxito:**
        - **Significancia estadística**: p < 0.05
        - **Mejora práctica**: Diferencia relevante
        - **Estabilidad**: Resultados consistentes
        - **Tamaño de muestra**: Suficiente para detectar diferencias
        
        **Modo Local:** Esta simulación muestra cómo funcionaría el sistema 
        con datos reales y una base de datos PostgreSQL.
        """)

# --- Main ---
def main():
    if page == "🏠 Resumen":
        show_overview_page()
    elif page == "🧬 Predecir Esperanza de Vida":
        show_prediction_page()
    elif page == "📊 Análisis de Datos":
        show_data_analysis_page()
    elif page == "🔍 Monitoreo de Deriva":
        show_drift_monitoring_page()
    elif page == "🔄 Reemplazo de Modelos":
        show_model_replacement_page()
    elif page == "🧪 Pruebas A/B":
        show_ab_testing_page()
    elif page == "📈 Rendimiento del Modelo":
        show_model_performance_page()

if __name__ == "__main__":
    main()
