FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando por defecto
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
