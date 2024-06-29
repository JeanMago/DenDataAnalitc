import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Carregar o arquivo CSV
file_path = 'dados_agrupados_mensal.csv'
data = pd.read_csv(file_path)

# Verificar dados ausentes
if data.isnull().sum().sum() > 0:
    # Preenchendo valores ausentes com a média da coluna (pode ser ajustado conforme necessário)
    data.fillna(data.mean(), inplace=True)

# Criar coluna de data e ajustar o formato corretamente
data['Data'] = pd.to_datetime(data['Ano'].astype(str) + '-' + data['Mes'].astype(str) + '-01')
data.set_index('Data', inplace=True)
data.drop(columns=['Ano', 'Mes'], inplace=True)

# Dividir em conjunto de treino (até 2021) e teste (a partir de 2022)
train_data = data.loc[:'2021']
test_data = data.loc['2022':]

# Separar features e targets
X_train = np.array([i.toordinal() for i in train_data.index]).reshape(-1, 1)
X_test = np.array([i.toordinal() for i in test_data.index]).reshape(-1, 1)
y_train_confirmados = train_data['Confirmados']
y_test_confirmados = test_data['Confirmados']
y_train_obitos = train_data['Obitos']
y_test_obitos = test_data['Obitos']
y_train_notificacoes = train_data['Notificacoes']
y_test_notificacoes = test_data['Notificacoes']

# Ajustar modelo Random Forest para cada target
rf_confirmados = RandomForestRegressor(n_estimators=100, random_state=42)
rf_obitos = RandomForestRegressor(n_estimators=100, random_state=42)
rf_notificacoes = RandomForestRegressor(n_estimators=100, random_state=42)

rf_confirmados.fit(X_train, y_train_confirmados)
rf_obitos.fit(X_train, y_train_obitos)
rf_notificacoes.fit(X_train, y_train_notificacoes)

# Avaliação do modelo
y_pred_confirmados = rf_confirmados.predict(X_test)
y_pred_obitos = rf_obitos.predict(X_test)
y_pred_notificacoes = rf_notificacoes.predict(X_test)

rmse_confirmados = mean_squared_error(y_test_confirmados, y_pred_confirmados, squared=False)
rmse_obitos = mean_squared_error(y_test_obitos, y_pred_obitos, squared=False)
rmse_notificacoes = mean_squared_error(y_test_notificacoes, y_pred_notificacoes, squared=False)

print(f'RMSE Confirmados: {rmse_confirmados}')
print(f'RMSE Óbitos: {rmse_obitos}')
print(f'RMSE Notificações: {rmse_notificacoes}')

# Criar datas para previsão (2024 e 2025)
future_dates = pd.date_range(start='2024-01-01', end='2025-12-01', freq='MS')
future_ordinal = np.array([i.toordinal() for i in future_dates]).reshape(-1, 1)

# Fazer previsões
pred_confirmados = rf_confirmados.predict(future_ordinal)
pred_obitos = rf_obitos.predict(future_ordinal)
pred_notificacoes = rf_notificacoes.predict(future_ordinal)

# Criar DataFrame com previsões
future_data = pd.DataFrame({
    'Data': future_dates,
    'Confirmados': pred_confirmados,
    'Obitos': pred_obitos,
    'Notificacoes': pred_notificacoes
})

# Salvar previsões em arquivos CSV
future_data_confirmados = future_data[['Data', 'Confirmados']]
future_data_obitos = future_data[['Data', 'Obitos']]
future_data_notificacoes = future_data[['Data', 'Notificacoes']]

future_data_confirmados.to_csv('previsoes_confirmados.csv', index=False)
future_data_obitos.to_csv('previsoes_obitos.csv', index=False)
future_data_notificacoes.to_csv('previsoes_notificacoes.csv', index=False)

# Exibir previsões
print(future_data)

# Definir função para plotar e salvar gráficos
def plot_and_save_predictions(dates, predictions, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, predictions, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Plotar e salvar gráficos para Confirmados, Óbitos e Notificações
plot_and_save_predictions(future_data['Data'], future_data['Confirmados'], 'Previsões de Confirmados (2024-2025)', 'Confirmados', 'previsoes_confirmados.png')
plot_and_save_predictions(future_data['Data'], future_data['Obitos'], 'Previsões de Óbitos (2024-2025)', 'Óbitos', 'previsoes_obitos.png')
plot_and_save_predictions(future_data['Data'], future_data['Notificacoes'], 'Previsões de Notificações (2024-2025)', 'Notificações', 'previsoes_notificacoes.png')

