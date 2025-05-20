# download_multistock.py
import yfinance as yf
import pandas as pd

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "BABA", "NVO",
    "SAP", "TM", "SIE.DE", "MELI", "VALE", "RDS-A",
    "PTR", "INFY"
]

start_date = "2010-01-01"
end_date = "2024-01-01"

data = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date,
    progress=True,
    group_by='ticker',
    auto_adjust=True
)

close_data = pd.DataFrame()
for ticker in tickers:
    if ticker in data.columns.levels[0]:
        close_data[ticker] = data[ticker]['Close']
    else:
        print(f"Advertencia: No se encontraron datos para {ticker}")

close_data = close_data.dropna(how='all')
close_data.to_csv("multistock_timeseries.csv")
print("✅ Datos descargados y guardados en multistock_timeseries.csv")
