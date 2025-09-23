import streamlit as st
import pandas as pd
import numpy as np
from pipeline import LifeExpectancyPipeline
import os

# --- Configuración de la página ---
st.set_page_config(page_title="Predicción de Esperanza de Vida", layout="wide")
st.title("🧬 Predicción de Esperanza de Vida")

# --- Inicializar Pipeline ---
pipeline = LifeExpectancyPipeline()

st.header("Cargando el Modelo y Preprocesador")
if pipeline.model is None or pipeline.preprocessor is None:
    st.warning("No se encontraron los archivos necesarios. Ejecuta el entrenamiento primero.")
    st.stop()
st.success("✅ Modelo y preprocesador cargados correctamente.")

# --- Cargar datos para sliders ---
# Detectar si estamos en Docker o localmente
if os.path.exists("/app/data/clean_data.csv"):
    # Estamos en Docker
    DATA_PATH = "/app/data/clean_data.csv"
    FEATURE_IMPORTANCE_PATH = "/app/models/feature_importance.csv"
else:
    # Estamos ejecutando localmente
    DATA_PATH = "data/clean_data.csv"
    FEATURE_IMPORTANCE_PATH = "models/feature_importance.csv"

try:
    df_clean = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"No se pudo cargar {DATA_PATH}: {e}")
    st.stop()

# --- Cargar top 5 features ---
try:
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    top_features = feature_importance['feature'].head(5).tolist()
except Exception as e:
    st.error(f"No se pudo cargar feature_importance.csv: {e}")
    st.stop()

# --- Inputs del usuario ---
user_input = {}
for feature in top_features:
    min_val = df_clean[feature].min()
    max_val = df_clean[feature].max()
    avg_val = df_clean[feature].mean()
    user_input[feature] = st.slider(
        f"Selecciona el valor para **{feature}**",
        float(min_val), float(max_val), float(avg_val)
    )

user_input['country'] = st.selectbox("País", options=pipeline.valid_countries)
user_input['status'] = st.selectbox("Status", options=pipeline.valid_status)
user_input['year'] = st.number_input("Año", min_value=2000, max_value=2025, value=2020)

# --- Botón de predicción ---
if st.button("Predecir Esperanza de Vida"):
    try:
        # Crear DataFrame completo
        # Separar columnas numéricas y categóricas
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        
        # Crear diccionario con valores por defecto
        default_values = {}
        
        # Para columnas numéricas, usar la media
        for col in numeric_columns:
            default_values[col] = df_clean[col].mean()
        
        # Para columnas categóricas, usar el valor más común (moda)
        for col in categorical_columns:
            mode_values = df_clean[col].mode()
            default_values[col] = mode_values[0] if len(mode_values) > 0 else 'Unknown'
        
        # Crear el DataFrame con valores por defecto
        df_input = pd.DataFrame([default_values])
        
        # Actualizar con los valores del usuario
        for feature in top_features:
            df_input[feature] = user_input[feature]
        df_input['country'] = user_input['country']
        df_input['status'] = user_input['status']
        df_input['year'] = user_input['year']

        # Remover la columna target si existe
        if 'life_expectancy' in df_input.columns:
            df_for_transform = df_input.drop(['life_expectancy'], axis=1)
        else:
            df_for_transform = df_input.copy()

        # Validación - Convertir a diccionario para evitar el error de Series
        input_dict = df_for_transform.iloc[0].to_dict()
        is_valid, message = pipeline.validate_data(input_dict)
        
        if not is_valid:
            st.error("❌ Error en los datos ingresados:")
            for m in message:
                st.write(f"- {m}")
        else:
            # Transformar datos
            transformed = pipeline.transform_data(input_dict)
            
            if transformed is not None:
                # Realizar predicción
                prediction = pipeline.predict(transformed)
                
                # Verificar que la predicción no sea None
                if prediction is None:
                    st.error("❌ Error: No se pudo hacer la predicción. Verifica que el modelo esté cargado correctamente.")
                else:
                    # Manejar tanto arrays como escalares
                    if isinstance(prediction, (list, np.ndarray)):
                        final_prediction = prediction[0]
                    else:
                        final_prediction = prediction
                    
                    st.success(f"✅ Predicción completada: **{final_prediction:.2f} años**")
            else:
                st.error("❌ Falló la transformación de los datos.")
                
    except Exception as e:
        st.error(f"❌ Error general: {str(e)}")
        import traceback
        st.code(traceback.format_exc())