# 🧬 Life Expectancy MLOps Dashboard

Sistema completo de Machine Learning Operations (MLOps) para predicción de esperanza de vida con monitoreo de deriva de datos, reemplazo automático de modelos y pruebas A/B.

## 📁 Estructura del Proyecto

```
equipo5_proyecto5/
├── app.py                         # Aplicación Streamlit unificada
├── ml/                            # Machine Learning
│   ├── pipeline.py                # Pipeline principal
│   ├── ml_modeling.py             # Modelado y entrenamiento
│   ├── cleaning.py                # Limpieza de datos
│   └── create_plots.py            # Generación de gráficos
├── mlops/                         # MLOps y Monitoreo
│   ├── data_drift_monitor.py      # Monitoreo de deriva
│   ├── model_auto_replacement.py  # Reemplazo automático
│   └── ab_testing.py              # Sistema de pruebas A/B
├── notebooks/                     # Jupyter Notebooks
│   ├── data_cleaning.ipynb        # Análisis y limpieza
│   ├── eda_visualizaciones.ipynb  # EDA y visualizaciones
│   └── ml_modeling.ipynb          # Modelado ML
├── backend/                       # Backend y Base de Datos
│   ├── src/                       # Código fuente backend
│   │   ├── config.py              # Configuración
│   │   ├── db_connect.py          # Conexión BD
│   │   ├── data_utils.py          # Utilidades datos
│   │   ├── model_utils.py         # Utilidades modelos
│   │   ├── prediction_utils.py    # Utilidades predicciones
│   │   ├── drift_utils.py         # Utilidades deriva
│   │   ├── experiments_utils.py   # Utilidades experimentos
│   │   └── feedback_utils.py      # Utilidades feedback
│   └── docker_postgree/           # Docker PostgreSQL
│       ├── docker-compose.yml     # Configuración Docker
│       ├── init.sql               # Script inicialización BD
│       └── Life_Expectancy_Data.csv
├── deployment/                    # Despliegue
│   ├── Dockerfile                 # Imagen Docker
│   ├── docker-compose.yml         # Orquestación contenedores
│   └── setup_database.py          # Script configuración BD
├── config/                        # Configuración
│   └── .streamlit/                # Configuración Streamlit
│       └── secrets.toml.example   # Ejemplo secrets
├── requirements.txt               # Dependencias Python
├── docs/                          # Documentación
│   ├── README.md                  # Este archivo
│   ├── README_VENV.md             # Guía entorno virtual
│   └── DEPLOYMENT.md              # Guía despliegue
├── data/                          # Datos
│   ├── clean_data.csv             # Datos limpios
│   └── Life Expectancy Data.csv   # Datos originales
├── models/                        # Modelos entrenados
│   ├── best_life_expectancy_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_importance.csv
│   ├── model_results.json
│   └── backups/                   # Respaldos modelos
├── plots/                         # Gráficos generados
├── tests/                         # Tests unitarios
└── venv/                          # Entorno virtual Python
```

## 🚀 Inicio Rápido

### Ejecutar Aplicación

```bash
# Ejecutar aplicación (detecta automáticamente si hay BD disponible)
./run_app.sh
# o
./run_app_local.sh  # Ambos scripts son equivalentes ahora
```

### Opción 3: Docker

```bash
# Construir y ejecutar con Docker
cd deployment
docker-compose up --build
```

## 📋 Requisitos

- Python 3.11+
- PostgreSQL (para versión con BD)
- Docker (opcional)

## 🛠️ Instalación

1. **Clonar repositorio:**
```bash
git clone <repository-url>
cd equipo5_proyecto5
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Ejecutar aplicación:**
```bash
./run_app.sh
```

## 🌐 Despliegue

### Streamlit Cloud

1. Configurar secrets en Streamlit Cloud:
```toml
[db]
host = "your-db-host"
port = 5432
name = "healthdb"
user = "your-user"
password = "your-password"
```

2. Conectar repositorio GitHub
3. Configurar archivo principal: `app.py`

### Base de Datos (Render/Railway)

Ver `docs/DEPLOYMENT.md` para instrucciones detalladas.

## 🧪 Testing

```bash
# Ejecutar todos los tests
python tests/run_all_tests.py

# Tests específicos
python -m pytest tests/test_model.py
python -m pytest tests/test_pipeline.py
```

## 📊 Características MLOps

- **Monitoreo de Deriva de Datos**: Detección automática de cambios en distribución
- **Reemplazo Automático de Modelos**: Actualización automática cuando se detecta degradación
- **Pruebas A/B**: Comparación de modelos en producción
- **Monitoreo de Rendimiento**: Seguimiento continuo de métricas
- **Feedback Loop**: Sistema de retroalimentación de usuarios

## 🔧 Configuración

### Variables de Entorno

```bash
# Base de datos
DB_HOST=localhost
DB_PORT=5432
DB_NAME=healthdb
DB_USER=admin
DB_PASSWORD=admin
```

### Streamlit Secrets

Ver `config/.streamlit/secrets.toml.example` para configuración de secrets.

## 📈 Uso

1. **Dashboard Principal**: Visualización general del sistema
2. **Análisis de Datos**: Exploración interactiva de datos
3. **Monitoreo de Deriva**: Análisis de cambios en datos
4. **Reemplazo de Modelos**: Gestión automática de modelos
5. **Pruebas A/B**: Comparación de modelos
6. **Rendimiento**: Monitoreo de métricas

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abrir Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👥 Equipo

- **Equipo 5** - Bootcamp IA
- **Proyecto 5** - MLOps Dashboard

## 📞 Soporte

Para soporte técnico o preguntas, contactar al equipo de desarrollo.
