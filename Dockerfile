# Usamos una imagen oficial de Python como base
FROM python:3.10-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Asegurarnos de que el directorio del modelo esté disponible
ENV PYTHONPATH=/app:$PYTHONPATH

# Exponer el puerto 3004 donde corre Flask
EXPOSE 3004

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]