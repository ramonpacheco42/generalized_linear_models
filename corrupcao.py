# %%
import rpy2.robjects as robjects
import pandas as pd
import statsmodels.api as sm
from rpy2.robjects import pandas2ri
# %%
# Configurando o arquivo RData
pandas2ri.activate()
# Carregando o arquivo RData
robjects.r['load']('corrupcao.RData')
# %%
# Lendo o dataframe carregado acima.
pandas2ri.activate()
dados = robjects.r['corrupcao']
df = pd.DataFrame.from_records(dados)
df = df.transpose()
df = df.rename(columns={0 : "pais", 1 : "cpi", 2 : "regiao"})
# %%
df.loc[df['regiao'] == 1, 'regiao'] = "america do sul"
df.loc[df['regiao'] == 2, 'regiao'] = "oceania"
df.loc[df['regiao'] == 3, 'regiao'] = "europa"
df.loc[df['regiao'] == 4, 'regiao'] = "eua e canada"
df.loc[df['regiao'] == 5, 'regiao'] = "asia"
df['cpi'] = df['cpi'].astype(float)
df
# %%
# cria as variáveis dummy para a coluna "regiao"
dummies = pd.get_dummies(df['regiao'], prefix='regiao')

# concatena o dataframe original com as variáveis dummy
df = pd.concat([df, dummies], axis=1)
df
# %%
# Estimando o modelo apartir da variável europa(Categoria de referencia)
# selecionar as variáveis independentes
X = df[['regiao_america do sul', 'regiao_asia', 'regiao_eua e canada', 'regiao_oceania']]

# adicionar uma constante
X = sm.add_constant(X)

# selecionar a variável dependente
Y = df['cpi']

# ajustar o modelo de regressão múltipla
model = sm.OLS(Y, X).fit()

# imprimir os resultados
print(model.summary())
# %%
# Estimando o modelo apartir da variável america do sul(Categoria de referencia)
# selecionar as variáveis independentes
X = df[['regiao_europa', 'regiao_asia', 'regiao_eua e canada', 'regiao_oceania']]

# adicionar uma constante
X = sm.add_constant(X)

# selecionar a variável dependente
Y = df['cpi']

# ajustar o modelo de regressão múltipla
model = sm.OLS(Y, X).fit()

# imprimir os resultados
print(model.summary())
# %%
