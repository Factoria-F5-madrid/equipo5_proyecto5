# 🚀 Guía de Despliegue - Life Expectancy MLOps

## 📋 Resumen del Proyecto
- **Frontend:** Streamlit Cloud
- **Base de Datos:** PostgreSQL en Render
- **Código:** GitHub

## 🗄️ PASO 1: Configurar Base de Datos en Render

### 1.1 Crear cuenta en Render
1. Ve a https://render.com
2. Regístrate con tu cuenta de GitHub

### 1.2 Crear base de datos PostgreSQL
1. **Dashboard** → **New** → **PostgreSQL**
2. **Configuración:**
   - **Name:** `life-expectancy-db`
   - **Database:** `healthdb`
   - **User:** `admin`
   - **Password:** (generar automáticamente)
   - **Region:** `Oregon (US West)`

### 1.3 Obtener credenciales
Render te dará algo como:
```
Host: dpg-xxxxx-a.oregon-postgres.render.com
Port: 5432
Database: healthdb
User: admin
Password: xxxxxxxx
```

### 1.4 Poblar la base de datos
```bash
# Configurar variables de entorno
export DB_HOST="tu-host-de-render"
export DB_PORT="5432"
export DB_NAME="healthdb"
export DB_USER="admin"
export DB_PASSWORD="tu-password-de-render"

# Ejecutar script de configuración
python setup_database.py
```

## 🌐 PASO 2: Desplegar en Streamlit Cloud

### 2.1 Preparar repositorio
```bash
# Subir cambios a GitHub
git add .
git commit -m "Add deployment configuration"
git push origin feature/advanced-mlops
```

### 2.2 Crear aplicación en Streamlit Cloud
1. Ve a https://share.streamlit.io
2. **New app** → **From GitHub repo**
3. **Configuración:**
   - **Repository:** `tu-usuario/equipo5_proyecto5`
   - **Branch:** `feature/advanced-mlops`
   - **Main file path:** `app.py`

### 2.3 Configurar secrets
En Streamlit Cloud, ve a **Settings** → **Secrets** y agrega:

```toml
[db]
host = "tu-host-de-render"
port = 5432
name = "healthdb"
user = "admin"
password = "tu-password-de-render"

[app]
debug = false
host = "0.0.0.0"
port = 8501

[mlops]
drift_threshold = 0.1
ab_min_sample_size = 100
enable_feedback = true
```

### 2.4 Deploy
1. **Deploy** → La aplicación se construirá automáticamente
2. **URL:** `https://tu-app-name.streamlit.app`

## ✅ PASO 3: Verificar Funcionalidades

### Funcionalidades disponibles:
- 🏠 **Overview** - Estado del sistema
- 🧬 **Predict Life Expectancy** - Predicciones con feedback
- 🔍 **Data Drift Monitoring** - Monitoreo de deriva
- 🔄 **Auto Model Replacement** - Reemplazo automático
- 🧪 **A/B Testing** - Comparación de modelos
- 📊 **Model Performance** - Métricas de rendimiento

### Verificar:
1. ✅ Aplicación carga correctamente
2. ✅ Predicciones funcionan
3. ✅ Feedback se guarda en BD
4. ✅ MLOps features operativas
5. ✅ Base de datos conectada

## 🔧 Troubleshooting

### Error de conexión a BD:
- Verificar credenciales en secrets
- Comprobar que la BD esté activa en Render
- Revisar logs en Streamlit Cloud

### Error de importación:
- Verificar que todos los archivos estén en GitHub
- Comprobar requirements.txt

### Error de permisos:
- Verificar que la BD permita conexiones externas
- Comprobar configuración de firewall

## 📊 Niveles Cumplidos

### 🟢 Nivel Esencial ✅
- Modelo ML funcional
- EDA con visualizaciones
- Overfitting < 5%
- Productivización (Streamlit)
- Informe de rendimiento

### 🟡 Nivel Medio ✅
- Técnicas ensemble
- Validación cruzada
- Optimización hiperparámetros
- Sistema feedback
- Pipeline ingestión datos

### 🟠 Nivel Avanzado ✅
- Dockerización (en otra rama)
- Base de datos PostgreSQL
- Tests unitarios

### 🔴 Nivel Experto ✅
- A/B Testing
- Data Drift Monitoring
- Auto-reemplazo modelos

## 🎯 Próximos Pasos

1. **Ejecutar setup_database.py** con credenciales de Render
2. **Subir código a GitHub**
3. **Desplegar en Streamlit Cloud**
4. **Configurar secrets**
5. **Verificar funcionalidades**
6. **Capturar screenshots para entrega**

## 📞 Soporte

Si tienes problemas:
1. Revisar logs en Streamlit Cloud
2. Verificar configuración de BD en Render
3. Comprobar que todos los archivos estén en GitHub
