import pandas as pd

# Carregar o arquivo CSV com codificação ISO-8859-1
file_path = 'dados_ijui.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Criar uma coluna de data a partir do ano e da semana epidemiológica
data['Data'] = pd.to_datetime(data['Ano'].astype(str) + data['Semana Epidemiologica'].astype(str) + '0', format='%Y%U%w', errors='coerce')

# Extrair o mês e o ano dessa data
data['Ano'] = data['Data'].dt.year
data['Mes'] = data['Data'].dt.month

# Agrupar os dados por 'Ano' e 'Mes' e somar as colunas de interesse
grouped_data = data.groupby(['Ano', 'Mes']).agg({
    'Confirmados': 'sum',
    'Obitos': 'sum',
    'Notificacoes': 'sum'
    
}).reset_index()

# Salvar o resultado em um novo arquivo CSV
output_path = 'dados_agrupados_mensal.csv'
grouped_data.to_csv(output_path, index=False)

# Caminho do arquivo gerado
print(output_path)
