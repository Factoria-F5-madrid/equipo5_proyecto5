#!/bin/bash
# Script para ejecutar la aplicación unificada con PYTHONPATH configurado

# Activar entorno virtual
source venv/bin/activate

# Configurar PYTHONPATH para incluir el directorio raíz
export PYTHONPATH=.

# Ejecutar la aplicación unificada
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
