import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from NN import FeedForwardNet
from rng import rng
import pandas as pd
import os
from aux import clean_data
from statsmodels.tsa.filters.hp_filter import hpfilter
import yfinance as yf


###################################################
###################################################
###################################################


# Cargar datos multiactivo
data = pd.read_csv("multistock_timeseries.csv", index_col=0, parse_dates=True)
data = data.dropna(axis=1)  # Eliminar columnas con valores faltantes


# Parámetros
input_size = 100
num_samples = 5000
hp_lambda = 1600

# Preparar señales limpias y ruidosas
clean_signals = []
noisy_signals = []

for col in data.columns:
    series = data[col].dropna().values
    if len(series) < input_size:
        continue
    for i in range(0, len(series) - input_size):
        window = series[i:i+input_size]
        _, trend = hpfilter(window, lamb=hp_lambda)
        clean_signals.append(trend)
        noisy_signals.append(window)
        if len(clean_signals) >= num_samples:
            break
    if len(clean_signals) >= num_samples:
        break

inputs = torch.tensor(noisy_signals, dtype=torch.float32)
targets = torch.tensor(clean_signals, dtype=torch.float32)

# Normalizar datos
inputs = (inputs - inputs.mean()) / inputs.std()
targets = (targets - targets.mean()) / targets.std()



###################################################
###################################################
###################################################

# Inicialización del modelo con parámetros de dilatación y traslación
dilation = 1.0  # Parámetro de dilatación
translation = 0.0  # Parámetro de traslación
model = FeedForwardNet(input_size, dilation, translation)

weights_path = "model_weights.pth"

if os.path.exists(weights_path):
    # Cargar pesos previamente entrenados
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    print("Pesos cargados desde archivo.")
else:
    # Entrenar modelo desde cero
    print("Entrenando modelo desde cero...")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 5000

    def smoothness_penalty(output):
        return torch.mean((output[:, 1:] - output[:, :-1]) ** 2)

    losses = []
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}')

        optimizer.zero_grad()
        outputs = model(inputs)
        mse_loss = torch.mean((outputs - targets) ** 2)
        smooth_loss = smoothness_penalty(outputs)
        loss = mse_loss + smooth_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Guardar pesos para uso futuro
    torch.save(model.state_dict(), weights_path)
    print(f"Entrenamiento completado. Pesos guardados en {weights_path}.")



###################################################
###################################################
###################################################

# ===== CONFIGURACIÓN =====
ticker = "BTC-USD"               # Activo (Yahoo Finance)
start_date = "2016-01-01"        # Fecha de inicio
end_date = "2024-01-01"          # Fecha de fin
window_size = input_size         # Tamaño de ventana (igual al tamaño de entrada para la red)
stride = 1                       # Paso entre ventanas

# ===== DESCARGAR DATOS =====
df = yf.download(ticker, start=start_date, end=end_date, group_by='None', progress=False)

df.columns = [col[1] for col in df.columns]  # extrae solo 'Open', 'Close', etc.

if df.empty or "Close" not in df:
    raise ValueError(f"No se pudo descargar datos válidos para {ticker}")

close_prices = df["Close"].dropna().values.astype(np.float32)

# ===== NORMALIZACIÓN =====
series_mean = close_prices.mean()
series_std = close_prices.std()
normalized_series = (close_prices - series_mean) / series_std

# ===== PREPARAR PREDICCIÓN POR TRAMOS =====
denoised_full = np.zeros_like(normalized_series)
count = np.zeros_like(normalized_series)

# ===== VENTANAS DESLIZANTES =====
for start in range(0, len(normalized_series) - window_size + 1, stride):
    end = start + window_size
    window = normalized_series[start:end]
    input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)[0].numpy()
    denoised_full[start:end] += output
    count[start:end] += 1

# ===== RECONSTRUIR SEÑAL =====
count[count == 0] = 1  # evitar divisiones por cero
denoised_full /= count
denoised_full = denoised_full * series_std + series_mean  # desnormalizar

# ===== GRAFICAR RESULTADO =====
plt.figure(figsize=(14,6))
plt.plot(close_prices, label=f"{ticker} original", alpha=0.4)
plt.plot(denoised_full, label="Red neuronal (por tramos)", alpha=0.8)
plt.title(f"{ticker}: comparación de eliminación de ruido (completo)")
plt.xlabel("Índice temporal")
plt.ylabel("Precio")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
