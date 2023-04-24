# %%
import rpy2.robjects as robjects
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from rpy2.robjects import pandas2ri
from mpl_toolkits.mplot3d import Axes3D
# %%
# Configurando o arquivo RData
pandas2ri.activate()
# Carregando o arquivo RData
robjects.r['load']('paises.RData')
# %%
# Lendo o dataframe carregado acima.
pandas2ri.activate()
dados = robjects.r['paises']
# Ajustando o dataframe para que seja apresentado corretamente.
df = pd.DataFrame.from_records(dados).transpose()
# %%
# Renomeando os nomes das colunas.
df = df.rename(columns={0 : "pais",1 : "cpi", 2 : "idade", 3 : "horas"})
# Arrumando os tipos de variáveis do df
df['cpi'] = df['cpi'].astype(float)
df['idade'] = df['idade'].astype(int)
df['horas'] = df['horas'].astype(float)
df
# %%
df.describe()
# %%
# Plotando Gráfico 3D das várias do Dataset Paises.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['cpi'], df['idade'], df['horas'])
ax.set_xlabel('CPI')
ax.set_ylabel('Idade')
ax.set_zlabel('Horas')
plt.show()
# %%
dados = df[['cpi', 'idade', 'horas']]
matriz_corr = dados.corr(method='pearson')
print(matriz_corr.round(2))
# %%
# Plotando gráficos de correlação das variáveis
sns.pairplot(df)
# %%
# plotando o gráfico de matriz de correlação com Seaborn
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm')
# exibindo o gráfico
plt.show()
# %%
# Definindo as variáveis independentes e dependentes
X = df[['idade', 'horas']]
Y = df['cpi']

# Adicionando a constante ao modelo
X = sm.add_constant(X)

# Criando o modelo de regressão múltipla
model = sm.OLS(Y, X).fit()

# Imprimindo os resultados
print(model.summary())
# %%
# Salvando o dados do modelo na coluna cpifit no dataframe original
df['cpifit'] = model.predict(X)
df
# %%
