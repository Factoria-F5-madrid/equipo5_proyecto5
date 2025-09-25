![Portada](https://drive.google.com/uc?export=view&id=1kkxRMWHfNrVrHebpBDSEFEqxEGpp7GnV)


## Descripción
Este proyecto busca predecir la esperanza de vida a partir de un conjunto de datos obtenido en Kaggle.
Se exploran relaciones entre factores socioeconómicos, sanitarios y demográficos, aplicando Machine Learning para construir un modelo de regresión que permita entender e inferir la variable objetivo.

## Objetivos
- Desarrollar un modelo de regresión para predecir la esperanza de vida
- Realizar análisis exploratorio de datos (EDA) completo
- Implementar técnicas de optimización de hiperparámetros
- Crear una aplicación web para productivizar el modelo

## Tecnologías Utilizadas
- **Análisis de datos**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Optimización**: Optuna
- **Visualización**: Matplotlib, Seaborn, Plotly
- **Aplicación web**: Streamlit
- **Notebooks**: Jupyter

## 📁 Estructura del Proyecto
```
├── data/                   # Datasets y archivos de datos
├── models/                 # Modelos entrenados guardados
├── notebooks/              # Jupyter notebooks de análisis
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```
---

## Dataset  
- **Fuente:** [Kaggle - Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)  
- **Observaciones:** 2938 filas, 22 columnas.  
- **Variable objetivo:** `Life expectancy` (años).  
- **Principales features:**  
  - Salud: *Adult Mortality, infant deaths, HIV/AIDS, Hepatitis B, Measles, BMI, Polio, Diphtheria, thinness, under-five deaths*  
  - Socioeconómicas: *GDP, Population, Income composition of resources, Schooling*  
  - Otros: *Alcohol, percentage expenditure, Total expenditure, Status (Developed/Developing)*  

---
## 🔍 Exploración inicial de los datos (EDA preliminar)  
- **Valores nulos:** varias columnas presentan missing values (ej. *GDP, Population, Hepatitis B, BMI*).  
- **Tipos de variables:**  
  - Categóricas → `Country`, `Status`  
  - Numéricas → 20 variables (años, porcentajes, tasas, PIB, etc.).  
- **Estadísticas básicas:**  
  - `Life expectancy` varía aprox. entre **36 y 90 años**, con media cercana a **70 años**.  
  - Alta correlación positiva con `Schooling` e `Income composition of resources`.  
  - Alta correlación negativa con `Adult Mortality` y `HIV/AIDS`.  

---

## Metodología  
1. **Análisis Exploratorio de Datos (EDA):**  
   - Distribución de variables.  
   - Correlaciones y gráficos explicativos.  
   - Identificación de outliers y datos faltantes.  

2. **Preprocesamiento:**  
   - Limpieza de valores nulos.  
   - Codificación de variables categóricas (`Status`).  
   - Estandarización/normalización de variables.  
   - Selección de features.  

3. **Entrenamiento del Modelo:**  
   - Modelos base: regresión lineal y regularizada.  
   - Modelos ensemble: Random Forest, XGBoost, Gradient Boosting.  
   - Validación cruzada (K-Fold).  

4. **Optimización:**  
   - Ajuste de hiperparámetros
     
5. **Evaluación:**  
    

6. **Productivización:**  
   - Aplicación web interactiva en Streamlit  
   - Despliegue de back en la nube Railway  

---

## 📸 Evidencia 
- Visualización de porcentajes del rendimiento de modelo en tiempo real en el front.  

---

## Instalación y Uso

### 1. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar notebooks
```bash
jupyter notebook
```

---

## Competencias demostradas en este proyecto: 
- **1:** Evaluar conjuntos de datos con análisis y visualización.  
- **2:** Aplicar algoritmos de ML según el problema, resolviendo retos clásicos de Inteligencia Artificial.

  ## Niveles alcanzados  
- **🟢 Esencial** → Modelo base + EDA + métricas + aplicación sencilla.  
- **🟡 Medio** → Ensembles, validación cruzada, optimización, pipeline de datos.  
- **🟠 Avanzado** → Dockerización, almacenamiento en BD, despliegue cloud, test unitarios.  
- **🔴 Experto** → MLOps con A/B Testing, monitoreo de drift, auto-reemplazo de modelos.  



___
## 👥 Equipo
- Bárbara Sánchez
- Mónica Gómez
- Azul Fayos
- Maribel Gutiérrez Ramírez
