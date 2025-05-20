from datetime import datetime, timedelta
from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import base64
from model.NN import FeedForwardNet  # Asegúrate de tener esta clase

app = Flask(__name__)

# Parámetros del modelo
input_size = 100
dilation = 1.0
translation = 0.0

# Inicializa la red y carga pesos si existen
model = FeedForwardNet(input_size, dilation, translation)
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device("cpu")))
model.eval()


def generate_plot(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, group_by='None', progress=False)
    
    df.columns = [col[1] for col in df.columns]  # extrae solo 'Open', 'Close', etc.

    if df.empty or "Close" not in df:
        raise ValueError(f"No se pudo descargar datos válidos para {ticker}")

    close_prices = df["Close"].dropna().values.astype(np.float32)

    mean = close_prices.mean()
    std = close_prices.std()
    normalized = (close_prices - mean) / std

    denoised = np.zeros_like(normalized)
    count = np.zeros_like(normalized)

    for start in range(0, len(normalized) - input_size + 1, 1):
        end = start + input_size
        window = normalized[start:end]
        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)[0].numpy()
        denoised[start:end] += output
        count[start:end] += 1

    count[count == 0] = 1
    denoised /= count
    denoised = denoised * std + mean

    # Graficar
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close_prices, label="Original", alpha=0.5)
    ax.plot(denoised, label="Denoised", alpha=0.8)
    ax.set_title(f"{ticker} - Red Neuronal Denoising")
    ax.legend()
    ax.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return img_base64


@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    error = None

    today = datetime.today()
    ten_years_ago = today - timedelta(days=365 * 10)
    
    default_start = ten_years_ago.strftime("%Y-%m-%d")
    default_end = today.strftime("%Y-%m-%d")

    if request.method == "POST":
        ticker = request.form.get("ticker")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        if not ticker or not start_date or not end_date:
            error = "Por favor completa todos los campos."
        else:
            plot_url = generate_plot(ticker, start_date, end_date)
            if plot_url is None:
                error = "Datos inválidos"

    return render_template("index.html", plot_url=plot_url, error=error,
                           default_start=default_start, default_end=default_end)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3004)
