from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Carregar os dados
file_path = 'dados_agrupados_semanal.csv'
dados = pd.read_csv(file_path)

# Exibir as primeiras linhas do dataframe para entender sua estrutura
dados.head()


# Selecionar as características (features) e os alvos (targets)
X = dados[['Ano', 'Semana Epidemiologica']]
y_confirmados = dados['Confirmados']
y_obitos = dados['Obitos']
y_notificacoes = dados['Notificacoes']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train_confirmados, y_test_confirmados = train_test_split(X, y_confirmados, test_size=0.2, random_state=42)
_, _, y_train_obitos, y_test_obitos = train_test_split(X, y_obitos, test_size=0.2, random_state=42)
_, _, y_train_notificacoes, y_test_notificacoes = train_test_split(X, y_notificacoes, test_size=0.2, random_state=42)

# Treinar os modelos de Regressão Linear
modelo_confirmados = LinearRegression()
modelo_obitos = LinearRegression()
modelo_notificacoes = LinearRegression()

modelo_confirmados.fit(X_train, y_train_confirmados)
modelo_obitos.fit(X_train, y_train_obitos)
modelo_notificacoes.fit(X_train, y_train_notificacoes)

# Fazer previsões nos dados de teste
y_pred_confirmados = modelo_confirmados.predict(X_test)
y_pred_obitos = modelo_obitos.predict(X_test)
y_pred_notificacoes = modelo_notificacoes.predict(X_test)

# Garantir que as previsões não sejam negativas
y_pred_confirmados = np.maximum(y_pred_confirmados, 0)
y_pred_obitos = np.maximum(y_pred_obitos, 0)
y_pred_notificacoes = np.maximum(y_pred_notificacoes, 0)

# Criar DataFrames com os resultados previstos e reais
resultados_confirmados = pd.DataFrame({'Real': y_test_confirmados, 'Previsto': y_pred_confirmados})
resultados_obitos = pd.DataFrame({'Real': y_test_obitos, 'Previsto': y_pred_obitos})
resultados_notificacoes = pd.DataFrame({'Real': y_test_notificacoes, 'Previsto': y_pred_notificacoes})



# Plotar os resultados previstos vs. reais
plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
plt.plot(resultados_confirmados['Real'].values, label='Real')
plt.plot(resultados_confirmados['Previsto'].values, label='Previsto', linestyle='--')
plt.xlabel('Amostra')
plt.ylabel('Confirmados')
plt.title('Previsão de Casos Confirmados')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(resultados_obitos['Real'].values, label='Real')
plt.plot(resultados_obitos['Previsto'].values, label='Previsto', linestyle='--')
plt.xlabel('Amostra')
plt.ylabel('Óbitos')
plt.title('Previsão de Óbitos')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(resultados_notificacoes['Real'].values, label='Real')
plt.plot(resultados_notificacoes['Previsto'].values, label='Previsto', linestyle='--')
plt.xlabel('Amostra')
plt.ylabel('Notificações')
plt.title('Previsão de Notificações')
plt.legend()

plt.tight_layout()
plt.show()

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train_confirmados, y_test_confirmados = train_test_split(X, y_confirmados, test_size=0.2, random_state=42)
_, _, y_train_obitos, y_test_obitos = train_test_split(X, y_obitos, test_size=0.2, random_state=42)
_, _, y_train_notificacoes, y_test_notificacoes = train_test_split(X, y_notificacoes, test_size=0.2, random_state=42)

# Treinar os modelos de Regressão Linear
modelo_confirmados = LinearRegression()
modelo_obitos = LinearRegression()
modelo_notificacoes = LinearRegression()

modelo_confirmados.fit(X_train, y_train_confirmados)
modelo_obitos.fit(X_train, y_train_obitos)
modelo_notificacoes.fit(X_train, y_train_notificacoes)

# Fazer previsões nos dados de teste
y_pred_confirmados = modelo_confirmados.predict(X_test)
y_pred_obitos = modelo_obitos.predict(X_test)
y_pred_notificacoes = modelo_notificacoes.predict(X_test)

# Garantir que as previsões não sejam negativas
y_pred_confirmados = np.maximum(y_pred_confirmados, 0)
y_pred_obitos = np.maximum(y_pred_obitos, 0)
y_pred_notificacoes = np.maximum(y_pred_notificacoes, 0)

# Criar DataFrames com os resultados previstos e reais
resultados_confirmados = pd.DataFrame({'Real': y_test_confirmados, 'Previsto': y_pred_confirmados})
resultados_obitos = pd.DataFrame({'Real': y_test_obitos, 'Previsto': y_pred_obitos})
resultados_notificacoes = pd.DataFrame({'Real': y_test_notificacoes, 'Previsto': y_pred_notificacoes})

# Salvar os resultados previstos em arquivos CSV
resultados_confirmados.to_csv('resultados_confirmados_previstos.csv', index=False)
resultados_obitos.to_csv('resultados_obitos_previstos.csv', index=False)
resultados_notificacoes.to_csv('resultados_notificacoes_previstos.csv', index=False)

# Plotar os resultados previstos vs. reais
plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
plt.plot(resultados_confirmados['Real'].values, label='Real')
plt.plot(resultados_confirmados['Previsto'].values, label='Previsto', linestyle='--')
plt.xlabel('Amostra')
plt.ylabel('Confirmados')
plt.title('Previsão de Casos Confirmados')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(resultados_obitos['Real'].values, label='Real')
plt.plot(resultados_obitos['Previsto'].values, label='Previsto', linestyle='--')
plt.xlabel('Amostra')
plt.ylabel('Óbitos')
plt.title('Previsão de Óbitos')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(resultados_notificacoes['Real'].values, label='Real')
plt.plot(resultados_notificacoes['Previsto'].values, label='Previsto', linestyle='--')
plt.xlabel('Amostra')
plt.ylabel('Notificações')
plt.title('Previsão de Notificações')
plt.legend()

plt.tight_layout()
plt.show()

resultados_confirmados.head(), resultados_obitos.head(), resultados_notificacoes.head()
