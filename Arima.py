import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

def carregar_dados(file_path):
    """Carrega os dados do arquivo CSV e realiza a formatação necessária."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return None
    except pd.errors.EmptyDataError:
        print("Erro: Arquivo CSV está vazio.")
        return None
    
    # Verificação de colunas necessárias
    required_columns = ['Ano', 'Semana Epidemiologica', 'Confirmados']
    if not all(col in data.columns for col in required_columns):
        print("Erro: Colunas necessárias faltando no arquivo CSV.")
        return None
    
    # Criar uma data a partir do Ano e Semana Epidemiológica
    data['Data'] = pd.to_datetime(data['Ano'].astype(str) + data['Semana Epidemiologica'].astype(str) + '0', format='%Y%U%w')
    
    # Definir a coluna 'Data' como índice
    data.set_index('Data', inplace=True)
    
    return data

def plotar_serie_temporal(series, title, xlabel, ylabel):
    """Plotar uma série temporal."""
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Casos Confirmados')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def dividir_dados(series, train_end_date):
    """Divide a série temporal em dados de treino e teste."""
    train_data = series[:train_end_date]
    test_data = series[train_end_date:]
    return train_data, test_data

def ajustar_modelo_sarima(train_data, order, seasonal_order):
    """Ajusta um modelo SARIMA aos dados de treino."""
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    return model_fit

def plotar_previsoes(train_data, test_data, predictions):
    """Plotar os dados de treino, teste e previsões."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label='Dados de Treino')
    plt.plot(test_data.index, test_data, label='Dados de Teste')
    plt.plot(test_data.index, predictions, label='Previsões', color='red')
    plt.title('Previsões do Modelo SARIMA')
    plt.xlabel('Data')
    plt.ylabel('Número de Casos Confirmados')
    plt.legend()
    plt.grid(True)
    plt.show()

def fazer_previsoes_futuras(model_fit, steps):
    """Faz previsões para um número determinado de passos no futuro."""
    future_predictions = model_fit.forecast(steps=steps)
    future_predictions = future_predictions.apply(lambda x: max(0, x))  # Evitar valores negativos
    return future_predictions

def plotar_previsoes_futuras(series, future_predictions, start_date):
    """Plotar previsões futuras."""
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Dados Históricos')
    future_dates = pd.date_range(start=start_date, periods=len(future_predictions), freq='W')
    plt.plot(future_dates, future_predictions, label='Previsões Futuras', color='red')
    plt.title('Previsões Futuras do Modelo SARIMA')
    plt.xlabel('Data')
    plt.ylabel('Número de Casos Confirmados')
    plt.legend()
    plt.grid(True)
    plt.show()

def salvar_previsoes_csv(future_predictions, start_date, file_name='previsoes_futuras.csv'):
    """Salvar previsões futuras em um arquivo CSV."""
    future_dates = pd.date_range(start=start_date, periods=len(future_predictions), freq='W')
    future_df = pd.DataFrame({'Data': future_dates, 'Previsões': future_predictions})
    future_df.to_csv(file_name, index=False)
    print(f'Previsões futuras salvas em {file_name}')

def treinar_modelo(train_data, order, seasonal_order):
    """Função para treinar o modelo SARIMA."""
    try:
        model_fit = ajustar_modelo_sarima(train_data, order, seasonal_order)
        return model_fit
    except Exception as e:
        print(f"Erro ao ajustar o modelo SARIMA: {e}")
        return None

def main(file_path):
    data = carregar_dados(file_path)
    if data is None:
        return
    
    print(data.head())

    series_confirmados = data['Confirmados']
    print(series_confirmados.head())

    plotar_serie_temporal(series_confirmados, 'Série Temporal de Casos Confirmados', 'Data', 'Número de Casos Confirmados')

    train_end_date = '2024-05-30'
    train_data, test_data = dividir_dados(series_confirmados, train_end_date)

    print(f'Tamanho dos dados de treino: {len(train_data)}')
    print(f'Tamanho dos dados de teste: {len(test_data)}')

    plotar_serie_temporal(train_data, 'Dados de Treino da Série Temporal', 'Data', 'Número de Casos Confirmados')
    plotar_serie_temporal(test_data, 'Dados de Teste da Série Temporal', 'Data', 'Número de Casos Confirmados')

    orders = [(5, 1, 0), (1, 1, 1), (2, 1, 2)]
    seasonal_orders = [(1, 1, 1, 52), (0, 1, 1, 52), (1, 1, 0, 52)]

    best_model = None
    best_mse = float('inf')

    for order in orders:
        for seasonal_order in seasonal_orders:
            model_fit = treinar_modelo(train_data, order, seasonal_order)
            if model_fit is not None:
                try:
                    predictions = model_fit.forecast(steps=len(test_data))
                    predictions = predictions.apply(lambda x: max(0, x))  # Evitar valores negativos
                    mse = mean_squared_error(test_data, predictions)
                    print(f'Order: {order}, Seasonal Order: {seasonal_order}, MSE: {mse}')
                    if mse < best_mse:
                        best_mse = mse
                        best_model = model_fit
                except Exception as exc:
                    print(f'O modelo com Order: {order} e Seasonal Order: {seasonal_order} gerou uma exceção: {exc}')

    if best_model:
        predictions = best_model.forecast(steps=len(test_data))
        predictions = predictions.apply(lambda x: max(0, x))  # Evitar valores negativos

        plotar_previsoes(train_data, test_data, predictions)

        mse = mean_squared_error(test_data, predictions)
        print(f'Melhor Erro Quadrático Médio (MSE): {mse}')

        future_steps = 104
        future_predictions = fazer_previsoes_futuras(best_model, future_steps)

        plotar_previsoes_futuras(series_confirmados, future_predictions, test_data.index[-1])
        salvar_previsoes_csv(future_predictions, test_data.index[-1])

if __name__ == "__main__":
    file_path = 'dados_agrupados_semanal.csv'
    main(file_path)
