#!/bin/bash

# Nombre de la imagen Docker
IMAGE_NAME="flask-denoising-app"
CONTAINER_NAME="denoising-app"
PORT=3004

echo "🏗️ Construyendo la imagen Docker..."
docker build -t $IMAGE_NAME .

echo "🚀 Iniciando el contenedor en el puerto $PORT..."
docker run -d \
  --name $CONTAINER_NAME \
  -p $PORT:$PORT \
  -e FLASK_RUN_PORT=$PORT \
  $IMAGE_NAME

echo "✅ Aplicación corriendo en http://localhost:$PORT"