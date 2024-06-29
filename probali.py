import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import csv  # Importar biblioteca csv


# Carregar o arquivo CSV com codificação ISO-8859-1
file_path = 'dados_agrupados_mensal.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Criar uma coluna de data a partir do ano e da semana epidemiológica
data['Data'] = pd.to_datetime(data['Ano'].astype(str) + data['Mes'].astype(str) + '0', format='%Y%U%w', errors='coerce')

# Extrair o mês e o ano dessa data
data['Ano'] = data['Data'].dt.year
data['Mes'] = data['Data'].dt.month

# Agrupar os dados por 'Ano' e 'Mes' e somar as colunas de interesse
grouped_data = data.groupby(['Ano', 'Mes']).agg({
    'Confirmados': 'sum',
    'Obitos': 'sum',
    'Notificacoes': 'sum'
}).reset_index()

# Criar uma coluna de data a partir de Ano e Mes
grouped_data['Data'] = pd.to_datetime(grouped_data['Ano'].astype(str) + '-' + grouped_data['Mes'].astype(str) + '-01')

# Definir a série temporal
time_series = grouped_data.set_index('Data')['Notificacoes']

# Ajustar o modelo SARIMA
model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Fazer previsões para os próximos 5 anos (60 meses)
forecast = results.get_forecast(steps=12)
forecast_index = pd.date_range(time_series.index[-1], periods=10, freq='M')
forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

# Plotar a série temporal e as previsões
plt.figure(figsize=(12, 7))
plt.plot(time_series, label='Dados Históricos')
plt.plot(forecast_series, label='Previsão', color='red')
plt.title('Previsão de Notificações de Surtos')
plt.xlabel('Data')
plt.ylabel('Número de Notificações')
plt.legend()
plt.show()

# Exibir os meses com maior probabilidade de ocorrências de surtos
print(forecast_series)
# Criar DataFrame com Previsões
previsoes_df = pd.DataFrame({
    'Data': forecast_index,
    'Previsão': forecast_series
})

# Salvar DataFrame como CSV
previsoes_df.to_csv('previsoes_surtos.csv', index=True)

# ... (Visualização e exibição dos resultados)