import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Carregar os dados do arquivo CSV
file_path = 'dados_agrupados_mensal.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Criar uma coluna de data a partir de Ano e Mes
data['Data'] = pd.to_datetime(data['Ano'].astype(str) + '-' + data['Mes'].astype(str) + '-01')

# Definir a série temporal
time_series = data.set_index('Data')['Confirmados']

# Ajustar o modelo SARIMA
model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_invertibility=False)
results = model.fit()

# Fazer previsões para os próximos 60 meses (5 anos)
forecast = results.get_forecast(steps=60)

# Extrair o índice das previsões (mensalmente)
forecast_index = pd.date_range(time_series.index[-1], periods=60, freq='M')

# Garantir que a série de previsões tenha o mesmo tamanho que o índice de previsão
forecast_series = forecast.predicted_mean[:len(forecast_index)]

# Criar DataFrame com datas e previsões
forecast_df = pd.DataFrame({
    'Data': forecast_index,
    'Previsao_Confirmados': forecast_series
})

# Salvar previsão em um arquivo CSV
forecast_csv_path = 'previsao_Confirmados.csv'
forecast_df.to_csv(forecast_csv_path, index=False)

# Plotar apenas as previsões
plt.figure(figsize=(12, 6))
plt.plot(forecast_index, forecast_series, label='Confirmados', color='red')
plt.title('Previsão de Confirmados de Surtos')
plt.xlabel('Data')
plt.ylabel('Número de Confirmados')
plt.legend()
plt.grid(True)
plt.show()

# Exibir as previsões
print(forecast_df)
print(f"A previsão foi salva em '{forecast_csv_path}'.")
