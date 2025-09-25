#!/usr/bin/env python3
"""
Script para migrar datos de CSV a PostgreSQL
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from backend.src.data_utils import save_training_data, save_reference_data
from backend.src.config import DATABASE_CONFIG
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_csv_to_database():
    """Migrar datos de CSV a PostgreSQL"""
    
    print("🔄 Iniciando migración de datos CSV a PostgreSQL...")
    
    # Verificar que existe el archivo CSV
    csv_path = "data/clean_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: No se encontró el archivo {csv_path}")
        return False
    
    try:
        # Cargar datos desde CSV
        print("📂 Cargando datos desde CSV...")
        df = pd.read_csv(csv_path)
        print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        
        # Mostrar información básica
        print(f"\n📊 Información del dataset:")
        print(f"  - Filas: {len(df)}")
        print(f"  - Columnas: {len(df.columns)}")
        print(f"  - Valores faltantes: {df.isnull().sum().sum()}")
        print(f"  - Columnas: {list(df.columns)}")
        
        # Guardar como datos de entrenamiento
        print("\n💾 Guardando datos de entrenamiento...")
        success_training = save_training_data(df)
        if success_training:
            print("✅ Datos de entrenamiento guardados correctamente")
        else:
            print("❌ Error guardando datos de entrenamiento")
            return False
        
        # Crear datos de referencia (usar una muestra para drift monitoring)
        print("\n📈 Creando datos de referencia...")
        reference_df = df.sample(n=min(1000, len(df)), random_state=42)
        success_reference = save_reference_data(reference_df)
        if success_reference:
            print("✅ Datos de referencia guardados correctamente")
        else:
            print("❌ Error guardando datos de referencia")
            return False
        
        print("\n🎉 Migración completada exitosamente!")
        print(f"  - Datos de entrenamiento: {len(df)} filas")
        print(f"  - Datos de referencia: {len(reference_df)} filas")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la migración: {e}")
        logger.error(f"Migration error: {e}")
        return False

def verify_migration():
    """Verificar que la migración fue exitosa"""
    
    print("\n🔍 Verificando migración...")
    
    try:
        from backend.src.data_utils import read_training_data, read_reference_data
        
        # Verificar datos de entrenamiento
        training_data = read_training_data()
        if not training_data.empty:
            print(f"✅ Datos de entrenamiento: {len(training_data)} filas")
        else:
            print("❌ No se encontraron datos de entrenamiento")
            return False
        
        # Verificar datos de referencia
        reference_data = read_reference_data()
        if not reference_data.empty:
            print(f"✅ Datos de referencia: {len(reference_data)} filas")
        else:
            print("❌ No se encontraron datos de referencia")
            return False
        
        print("✅ Verificación completada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error durante la verificación: {e}")
        return False

def main():
    """Función principal"""
    
    print("=" * 60)
    print("🚀 MIGRACIÓN DE DATOS CSV A POSTGRESQL")
    print("=" * 60)
    
    # Mostrar configuración de base de datos
    print(f"\n📋 Configuración de base de datos:")
    print(f"  - Host: {DATABASE_CONFIG['host']}")
    print(f"  - Puerto: {DATABASE_CONFIG['port']}")
    print(f"  - Base de datos: {DATABASE_CONFIG['database']}")
    print(f"  - Usuario: {DATABASE_CONFIG['user']}")
    
    # Ejecutar migración
    success = migrate_csv_to_database()
    
    if success:
        # Verificar migración
        verify_success = verify_migration()
        
        if verify_success:
            print("\n🎉 ¡Migración completada exitosamente!")
            print("   Los datos están listos para usar en el sistema MLOps")
        else:
            print("\n⚠️ Migración completada pero hay problemas en la verificación")
    else:
        print("\n❌ La migración falló")
        sys.exit(1)

if __name__ == "__main__":
    main()
