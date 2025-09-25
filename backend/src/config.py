"""
Configuración central para el sistema MLOps
"""

import os
from typing import Dict, Any

# Configuración de base de datos
# Prioridad: Streamlit secrets > Environment variables > Defaults
def get_db_config():
    """Obtener configuración de BD con prioridad para Streamlit"""
    try:
        import streamlit as st
        # Si estamos en Streamlit, usar secrets (solo si existen)
        if hasattr(st, 'secrets') and hasattr(st.secrets, 'get') and st.secrets.get('db'):
            return {
                'host': st.secrets['db']['host'],
                'port': int(st.secrets['db']['port']),
                'database': st.secrets['db']['name'],
                'user': st.secrets['db']['user'],
                'password': st.secrets['db']['password']
            }
    except Exception:
        # Ignorar errores de secrets silenciosamente
        pass
    
    # Fallback a variables de entorno
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'healthdb'),
        'user': os.getenv('DB_USER', 'admin'),
        'password': os.getenv('DB_PASSWORD', 'admin')
    }

DATABASE_CONFIG = get_db_config()

# URL de conexión para SQLAlchemy
DATABASE_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"

# Configuración de modelos
MODEL_CONFIG = {
    'default_algorithm': 'RandomForestRegressor',
    'default_hyperparams': {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'model_storage_path': 'models/',
    'preprocessor_storage_path': 'models/preprocessor.pkl'
}

# Configuración de drift monitoring
DRIFT_CONFIG = {
    'threshold': float(os.getenv('DRIFT_THRESHOLD', '0.1')),
    'check_interval_hours': int(os.getenv('DRIFT_CHECK_INTERVAL', '24')),
    'alert_threshold': float(os.getenv('DRIFT_ALERT_THRESHOLD', '0.15')),
    'features_to_monitor': [
        'adult_mortality', 'infant_deaths', 'alcohol', 'gdp', 
        'schooling', 'hiv/aids', 'bmi', 'population'
    ]
}

# Configuración de A/B testing
AB_TESTING_CONFIG = {
    'min_sample_size': int(os.getenv('AB_MIN_SAMPLE_SIZE', '100')),
    'max_duration_days': int(os.getenv('AB_MAX_DURATION', '30')),
    'default_traffic_split': float(os.getenv('AB_DEFAULT_SPLIT', '0.5')),
    'significance_level': float(os.getenv('AB_SIGNIFICANCE_LEVEL', '0.05'))
}

# Configuración de logging
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': os.getenv('LOG_FILE', 'logs/mlops.log')
}

# Configuración de predicciones
PREDICTION_CONFIG = {
    'confidence_threshold': float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', '0.8')),
    'max_processing_time_ms': int(os.getenv('MAX_PROCESSING_TIME_MS', '5000')),
    'enable_feedback': os.getenv('ENABLE_FEEDBACK', 'true').lower() == 'true'
}

# Configuración de alertas
ALERT_CONFIG = {
    'email_enabled': os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true',
    'email_recipients': os.getenv('EMAIL_RECIPIENTS', '').split(','),
    'slack_enabled': os.getenv('SLACK_ALERTS_ENABLED', 'false').lower() == 'true',
    'slack_webhook': os.getenv('SLACK_WEBHOOK_URL', '')
}

# Configuración de la aplicación
APP_CONFIG = {
    'debug': os.getenv('DEBUG', 'false').lower() == 'true',
    'host': os.getenv('APP_HOST', '0.0.0.0'),
    'port': int(os.getenv('APP_PORT', '8501')),
    'reload': os.getenv('APP_RELOAD', 'false').lower() == 'true'
}

def get_config() -> Dict[str, Any]:
    """Obtener toda la configuración"""
    return {
        'database': DATABASE_CONFIG,
        'model': MODEL_CONFIG,
        'drift': DRIFT_CONFIG,
        'ab_testing': AB_TESTING_CONFIG,
        'logging': LOGGING_CONFIG,
        'prediction': PREDICTION_CONFIG,
        'alert': ALERT_CONFIG,
        'app': APP_CONFIG
    }

def validate_config() -> bool:
    """Validar configuración"""
    try:
        # Validar configuración de base de datos
        assert DATABASE_CONFIG['host'], "DB_HOST is required"
        assert DATABASE_CONFIG['database'], "DB_NAME is required"
        assert DATABASE_CONFIG['user'], "DB_USER is required"
        assert DATABASE_CONFIG['password'], "DB_PASSWORD is required"
        
        # Validar rangos de configuración
        assert 0 < DRIFT_CONFIG['threshold'] < 1, "DRIFT_THRESHOLD must be between 0 and 1"
        assert 0 < AB_TESTING_CONFIG['default_traffic_split'] < 1, "AB_DEFAULT_SPLIT must be between 0 and 1"
        assert 0 < PREDICTION_CONFIG['confidence_threshold'] < 1, "PREDICTION_CONFIDENCE_THRESHOLD must be between 0 and 1"
        
        return True
    except AssertionError as e:
        print(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error validating configuration: {e}")
        return False

if __name__ == "__main__":
    # Mostrar configuración actual
    config = get_config()
    print("=== MLOps Configuration ===")
    for section, settings in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in settings.items():
            if 'password' in key.lower():
                print(f"  {key}: {'*' * len(str(value))}")
            else:
                print(f"  {key}: {value}")
    
    # Validar configuración
    print(f"\nConfiguration valid: {validate_config()}")
