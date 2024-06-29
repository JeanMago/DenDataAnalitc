import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo CSV com codificação ISO-8859-1
file_path = 'dados_ijui.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Agrupar os dados por 'Ano' e somar as colunas de interesse
grouped_data = data.groupby('Ano').agg({
    'Confirmados': 'sum',
    'Obitos': 'sum',
    'Notificacoes': 'sum'
}).reset_index()

# Configurações do gráfico
plt.figure(figsize=(10, 6))
width = 0.25  # Largura das barras
x = grouped_data['Ano']

# Criação das barras
plt.bar(x - width, grouped_data['Confirmados'], width=width, label='Confirmados', color='blue', alpha=0.6)
plt.bar(x, grouped_data['Obitos'], width=width, label='Óbitos', color='red', alpha=0.6)
plt.bar(x + width, grouped_data['Notificacoes'], width=width, label='Notificações', color='green', alpha=0.6)

# Adicionar títulos e rótulos
plt.title('Dados Agrupados por Ano')
plt.xlabel('Ano')
plt.ylabel('Quantidade')
plt.legend()

# Exibir o gráfico
plt.xticks(x)
plt.show()
