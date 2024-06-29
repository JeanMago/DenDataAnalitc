import pandas as pd

# Carregar o arquivo CSV com codificação ISO-8859-1
file_path = 'dados_ijui.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Agrupar os dados por 'Ano' e 'Semana Epidemiologica' e somar as colunas de interesse
grouped_data = data.groupby(['Ano', 'Semana Epidemiologica']).agg({
    'Confirmados': 'sum',
    'Obitos': 'sum',
    'Notificacoes': 'sum'
}).reset_index()

# Salvar o resultado em um novo arquivo CSV
output_path = 'dados_agrupados_semanal.csv'
grouped_data.to_csv(output_path, index=False)

# Caminho do arquivo gerado
print(output_path)
