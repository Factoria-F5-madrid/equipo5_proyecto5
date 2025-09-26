"""
MLOps Backend Package
Sistema de backend para MLOps con PostgreSQL
"""

__version__ = "1.0.0"
__author__ = "Equipo 5 - Bootcamp IA"
from .db_connect import get_connection, get_cursor, engine
from .data_utils import (
    read_world_health,
    read_feedback,
    read_training_data,
    read_reference_data,
    save_training_data,
    save_reference_data
)
from .model_utils import (
    insert_model,
    get_active_model,
    get_model_by_id,
    update_model_status,
    get_all_models,
    save_feature_importance
)
from .prediction_utils import (
    save_prediction,
    get_predictions,
    get_prediction_stats,
    get_predictions_by_country
)
from .drift_utils import (
    log_drift,
    get_recent_drifts,
    get_drift_summary,
    create_drift_alert,
    get_active_alerts
)
from .experiments_utils import (
    create_experiment,
    log_experiment_result,
    get_experiment,
    get_active_experiments,
    update_experiment_status,
    get_experiment_results,
    get_experiment_summary
)
from .feedback_utils import save_feedback
from .config import get_config, validate_config

__all__ = [
    'get_connection',
    'get_cursor',
    'engine',
    'read_world_health',
    'read_feedback',
    'read_training_data',
    'read_reference_data',
    'save_training_data',
    'save_reference_data',
    'insert_model',
    'get_active_model',
    'get_model_by_id',
    'update_model_status',
    'get_all_models',
    'save_feature_importance',
    'save_prediction',
    'get_predictions',
    'get_prediction_stats',
    'get_predictions_by_country',
    'log_drift',
    'get_recent_drifts',
    'get_drift_summary',
    'create_drift_alert',
    'get_active_alerts',
    'create_experiment',
    'log_experiment_result',
    'get_experiment',
    'get_active_experiments',
    'update_experiment_status',
    'get_experiment_results',
    'get_experiment_summary',
    'save_feedback',
    'get_config',
    'validate_config'
]
