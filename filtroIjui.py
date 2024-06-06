import pandas as pd

# Caminho do arquivo CSV fornecido
arquivo_csv = 'dengue_resid_csv.csv'

# Leitura do arquivo CSV com a codificação adequada
df = pd.read_csv(arquivo_csv, encoding='ISO-8859-1', header=None, 
                 names=['Ano', 'Semana Epidemiologica', 'Cid IBGE', 'CRS', 'Nome Municipio', 'Sexo', 'Faixa Etaria', 'Raca Cor', 'Obitos', 'Investigacao', 'Descartados', 'Inconclusivos', 'Autoctones', 'Confirmados', 'Notificacoes'])

# Corrigir a codificação do caractere especial para 'Ijuï¿½'
df['Nome Municipio'] = df['Nome Municipio'].str.replace('Ijuï¿½', 'Ijuí')

# Filtrar os dados para o município de Ijuí
df_ijui = df[df['Nome Municipio'] == 'Ijuí']

# Exibir os resultados filtrados
print(df_ijui)

# Opcional: salvar os resultados filtrados em um novo arquivo CSV
df_ijui.to_csv('dados_ijui.csv', index=False, encoding='ISO-8859-1')
