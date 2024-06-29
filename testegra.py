import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Carregar os dados do arquivo CSV
file_path = 'dados_agrupados_mensal.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Preparar dados para o gráfico
anos = data['Ano']
meses = data['Mes']
confirmados = data['Confirmados']

# Criar uma figura e um eixo 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotar o gráfico de barras em 3D
ax.bar3d(anos, meses, [0] * len(confirmados), 0.5, 0.5, confirmados, color='skyblue')

# Configurar rótulos e título
ax.set_xlabel('Ano')
ax.set_ylabel('Mês')
ax.set_zlabel('Casos Confirmados')
ax.set_title('Casos Confirmados por Ano e Mês')

plt.show()
