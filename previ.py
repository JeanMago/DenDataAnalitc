import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Carregar os dados do arquivo CSV
file_path = 'dados_ijui.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Preparar os dados para o modelo SARIMA
data['Data'] = pd.to_datetime(data['Ano'].astype(str) + '-' + data['Mes'].astype(str) + '-01')
time_series = data.set_index('Data')['Confirmados']

# Ajustar o modelo SARIMA
model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Fazer previsões para os próximos 5 anos (60 meses)
forecast = results.get_forecast(steps=60)
forecast_index = pd.date_range(time_series.index[-1], periods=61, freq='M')  # +1 para incluir o mês atual
forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

# Plotar a série temporal e as previsões
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Dados Históricos')
plt.plot(forecast_series, label='Previsão', color='red')
plt.title('Previsão de Casos Confirmados para os Próximos 5 Anos')
plt.xlabel('Data')
plt.ylabel('Casos Confirmados')
plt.legend()
plt.show()

# Exibir as previsões para os próximos 5 anos
print("Previsão de Casos Confirmados para os Próximos 5 Anos:")
print(forecast_series.tail(60))  # Exibir as últimas 60 previsões
