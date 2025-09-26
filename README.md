# 🧬 Dashboard MLOps - Predicción de Esperanza de Vida

![Predicción de la Esperanza de Vida con Machine Learning](https://via.placeholder.com/800x400/FFE5E5/000000?text=PREDICCIÓN+DE+LA+ESPERANZA+DE+VIDA+CON+MACHINE+LEARNING)

## 🌐 Aplicación Desplegada

**🔗 [Ver Aplicación en Vivo](https://equipo5-proyecto5-1.onrender.com/)**

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de **Machine Learning Operations (MLOps)** para la predicción de esperanza de vida basado en indicadores socioeconómicos y de salud. La aplicación combina técnicas avanzadas de machine learning con un sistema híbrido de predicción que integra datos reales de países para generar predicciones más precisas y realistas.

## 🎯 Objetivo

Desarrollar un sistema MLOps robusto que prediga la esperanza de vida de países utilizando 18 características socioeconómicas y de salud, implementando las mejores prácticas de machine learning en producción.

## 🏗️ Arquitectura del Sistema

```
equipo5_proyecto5/
├── app.py                         # Aplicación Streamlit principal
├── ml/                            # Módulos de Machine Learning
│   ├── pipeline.py                # Pipeline principal de ML
│   ├── country_data.py            # Datos reales de países
│   ├── ml_modeling.py             # Modelado y entrenamiento
│   ├── cleaning.py                # Limpieza de datos
│   └── create_plots.py            # Generación de visualizaciones
├── mlops/                         # Módulos MLOps
│   ├── data_drift_monitor.py      # Monitoreo de deriva de datos
│   ├── model_auto_replacement.py  # Reemplazo automático de modelos
│   └── ab_testing.py              # Sistema de pruebas A/B
├── backend/                       # Lógica de backend y base de datos
│   ├── src/                       # Código fuente del backend
│   │   ├── config.py              # Configuración de la aplicación
│   │   ├── db_connect.py          # Conexión a la base de datos
│   │   └── feedback_utils.py      # Utilidades para feedback
│   └── docker_postgree/           # Configuración Docker para PostgreSQL
├── data/                          # Datasets
│   └── clean_data.csv             # Dataset limpio de esperanza de vida
├── models/                        # Modelos entrenados y preprocesadores
├── tests/                         # Pruebas unitarias
├── deployment/                    # Configuración de despliegue
└── requirements.txt               # Dependencias Python
```

## 🚀 Características Principales

### 🤖 Modelo de Machine Learning
- **Algoritmo**: Gradient Boosting Regressor con 200 estimadores
- **Preprocesamiento**: Imputación de valores faltantes y escalado estándar
- **Validación**: Cross-validation con 5 folds
- **Sistema Híbrido**: Combina 70% datos reales + 30% predicción ML

### 📊 Análisis Exploratorio de Datos (EDA)
- **Visualizaciones interactivas** con Plotly
- **Análisis de correlaciones** entre variables
- **Distribución de la variable objetivo** (esperanza de vida)
- **Comparación entre países desarrollados y en desarrollo**
- **Análisis temporal** de tendencias por país

### 🔧 Sistema MLOps Avanzado

#### 📈 Monitoreo de Data Drift
- **Detección automática** de cambios en la distribución de datos
- **Alertas en tiempo real** cuando se detecta deriva
- **Métricas estadísticas** (KS test, PSI) para comparar distribuciones
- **Dashboard interactivo** para visualizar el estado del modelo

#### 🔄 Auto-Reemplazo de Modelos
- **Evaluación continua** del rendimiento del modelo en producción
- **Reentrenamiento automático** cuando el rendimiento cae
- **Validación de métricas** antes del despliegue
- **Rollback automático** si el nuevo modelo no mejora

#### 🧪 A/B Testing
- **Comparación de modelos** en tiempo real
- **Métricas de rendimiento** para cada variante
- **Análisis estadístico** de diferencias significativas
- **Implementación automática** del mejor modelo

### 💾 Gestión de Datos
- **Base de datos PostgreSQL** para almacenamiento persistente
- **Sistema de feedback** para recopilar datos de usuarios
- **Pipeline de ingestión** para datos nuevos
- **Backup automático** de modelos y datos

## 🛠️ Tecnologías Utilizadas

### Machine Learning
- **scikit-learn**: Algoritmos de ML y preprocesamiento
- **pandas & numpy**: Manipulación de datos
- **joblib**: Serialización de modelos

### Visualización y Análisis
- **Plotly**: Visualizaciones interactivas
- **Matplotlib & Seaborn**: Gráficos estáticos
- **Streamlit**: Interfaz de usuario

### MLOps y Producción
- **PostgreSQL**: Base de datos relacional
- **Docker**: Containerización
- **Streamlit Cloud**: Despliegue en la nube
- **SQLAlchemy**: ORM para base de datos

### Testing y Calidad
- **pytest**: Framework de testing
- **unittest**: Pruebas unitarias

## 📈 Métricas de Rendimiento

### Métricas de Regresión
- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto medio
- **R²**: Coeficiente de determinación
- **Cross-validation**: Validación cruzada con 5 folds

### Métricas MLOps
- **Data Drift Score**: Medida de deriva de datos
- **Model Performance**: Rendimiento en tiempo real
- **A/B Test Results**: Comparación de modelos
- **Feedback Quality**: Calidad de datos de usuario

## 🎯 Niveles de Entrega Implementados

### 🟢 Nivel Esencial ✅
- ✅ **Modelo ML funcional** que predice esperanza de vida
- ✅ **EDA completo** con visualizaciones de regresión
- ✅ **Overfitting < 5%** mediante validación cruzada
- ✅ **Productivización** con Streamlit
- ✅ **Informe de rendimiento** con métricas detalladas

### 🟡 Nivel Medio ✅
- ✅ **Ensemble methods** (Gradient Boosting)
- ✅ **Validación cruzada** (K-Fold)
- ✅ **Optimización de hiperparámetros** con validación
- ✅ **Sistema de feedback** para monitoreo
- ✅ **Pipeline de ingestión** de datos nuevos

### 🟠 Nivel Avanzado ✅
- ✅ **Dockerización** completa del sistema
- ✅ **Base de datos PostgreSQL** para persistencia
- ✅ **Despliegue en Render** - [Ver aplicación](https://equipo5-proyecto5-1.onrender.com/)
- ✅ **Test unitarios** para validación

### 🔴 Nivel Experto ✅
- ✅ **A/B Testing** para comparar modelos
- ✅ **Monitoreo de Data Drift** en tiempo real
- ✅ **Auto-reemplazo de modelos** con validación
- ✅ **Sistema MLOps completo** con todas las funcionalidades

## 🚀 Instalación y Uso

### 🌐 Acceso Rápido
**🔗 [Ver Aplicación Desplegada](https://equipo5-proyecto5-1.onrender.com/)**

### Requisitos Previos
- Python 3.11+
- PostgreSQL (opcional, para modo completo)
- Docker (opcional, para containerización)

### Instalación
```bash
# Clonar el repositorio
git clone <repository-url>
cd equipo5_proyecto5

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

### Uso con Docker
```bash
# Construir imagen
docker build -t life-expectancy-app .

# Ejecutar contenedor
docker run -p 8501:8501 life-expectancy-app
```

## 📊 Datasets

### Dataset Principal
- **Archivo**: `data/clean_data.csv`
- **Registros**: 2,939 países-año
- **Características**: 18 variables socioeconómicas y de salud
- **Período**: 2000-2015
- **Países**: 193 países

### Variables Principales
- **Salud**: Mortalidad adulta, muertes infantiles, vacunaciones
- **Económicas**: PIB per cápita, gasto en salud, composición de ingresos
- **Sociales**: Escolaridad, IMC, consumo de alcohol
- **Demográficas**: Población, estado de desarrollo

## 🔧 Configuración

### Variables de Entorno
```bash
# Base de datos (opcional)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=healthdb
DB_USER=admin
DB_PASSWORD=admin
```

### Configuración de Streamlit Cloud
Crear archivo `.streamlit/secrets.toml`:
```toml
[db]
host = "your-db-host"
port = 5432
name = "your-db-name"
user = "your-username"
password = "your-password"
```

## 🧪 Testing

### Ejecutar Tests
```bash
# Todos los tests
python -m pytest tests/

# Test específico
python -m pytest tests/test_pipeline.py

# Con cobertura
python -m pytest --cov=ml tests/
```

### Tests Incluidos
- **Test de pipeline**: Validación del modelo ML
- **Test de datos**: Validación de preprocesamiento
- **Test de métricas**: Validación de rendimiento mínimo
- **Test de integración**: Validación end-to-end

## 📈 Monitoreo y Mantenimiento

### Dashboard de Monitoreo
- **Estado del modelo**: Rendimiento en tiempo real
- **Data Drift**: Alertas de cambios en datos
- **A/B Testing**: Comparación de modelos
- **Feedback**: Calidad de datos de usuario

### Mantenimiento Automático
- **Reentrenamiento**: Cuando el rendimiento cae
- **Actualización de datos**: Ingestión automática
- **Backup**: Respaldo automático de modelos
- **Alertas**: Notificaciones de problemas

## 🤝 Contribución

### Estructura del Código
- **Modular**: Cada funcionalidad en su módulo
- **Documentado**: Docstrings en todas las funciones
- **Testeable**: Cobertura de tests > 80%
- **Mantenible**: Código limpio y organizado

### Flujo de Trabajo
1. Fork del repositorio
2. Crear rama feature
3. Implementar cambios
4. Ejecutar tests
5. Crear pull request



## 👥 Equipo

- **Desarrollo**: Maribel Gutierrez, Mónica Gómez y Bárbara Sánchez
- **Mentoría**: Factoria F5 Madrid


## 📞 Contacto

Para preguntas o sugerencias sobre el proyecto, contactar al equipo de desarrollo.

---

**Nota**: Este proyecto implementa un sistema MLOps completo siguiendo las mejores prácticas de la industria, desde el análisis exploratorio hasta el despliegue en producción con monitoreo continuo.