"""
Script para migrar datos de CSV a PostgreSQL
"""

import pandas as pd
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.src.data_utils import save_training_data, save_reference_data
from backend.src.config import DATABASE_CONFIG
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_csv_to_database():
    """Migrar datos de CSV a PostgreSQL"""
    csv_path = "data/clean_data.csv"
    if not os.path.exists(csv_path):
        return False

    try:
        df = pd.read_csv(csv_path)
        } filas, {len(df.columns)} columnas")
        }")
        }")
        .sum().sum()}")
        }")
        success_training = save_training_data(df)
        if success_training:
            else:
            return False
        reference_df = df.sample(n=min(1000, len(df)), random_state=42)
        success_reference = save_reference_data(reference_df)
        if success_reference:
            else:
            return False

        } filas")
        } filas")

        return True

    except Exception as e:
        logger.error(f"Migration error: {e}")
        return False

def verify_migration():
    """Verificar que la migración fue exitosa"""

    try:
        from backend.src.data_utils import read_training_data, read_reference_data
        training_data = read_training_data()
        if not training_data.empty:
            } filas")
        else:
            return False
        reference_data = read_reference_data()
        if not reference_data.empty:
            } filas")
        else:
            return False

        return True

    except Exception as e:
        return False

def main():
    """Función principal"""
    success = migrate_csv_to_database()

    if success:
        verify_success = verify_migration()

        if verify_success:
            else:
            else:
        sys.exit(1)

if __name__ == "__main__":
    main()
