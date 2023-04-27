# %%
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro, norm
# %%
# Configurando o arquivo RData
robjects.pandas2ri.activate()
# Carregando o arquivo RData
robjects.r['load']('bebes.RData')
# %%
# Lendo o dataframe carregado acima.
pandas2ri.activate()
dados = robjects.r['bebes']
# Ajustando o dataframe para que seja apresentado corretamente.
df = pd.DataFrame.from_records(dados).transpose()
df = df.rename(columns={0: 'comprimento', 1 : 'idade'})
df
# %%
# Plotando um gráfico de dispersão para observar a relação entre as duas variáveis
plt.scatter(df['idade'],df['comprimento'])
plt.xlabel('Idade')
plt.ylabel('Comprimento')
plt.show()
# %%
# Criando modelo Linear:
x = sm.add_constant(df['idade'])
model_lin = sm.OLS(df['comprimento'], x).fit()
predict_lin = model_lin.predict(x)

# Criando modelo loess:
lowess = sm.nonparametric.lowess(df['comprimento'], df['idade'], frac=0.3)
x_loess = lowess[:, 0]
y_loess = lowess[:, 1]

plt.scatter(df['idade'],df['comprimento'])
plt.xlabel('Idade')
plt.ylabel('Comprimento')
plt.plot(df['idade'], predict_lin, color='green', label='Linear', linewidth=3)
plt.plot(x_loess, y_loess, color='red', label='Loess', linewidth=3)
plt.legend()
plt.show()
# %%
print(model_lin.summary())
# %%
# Teste de verificação da aderência dos resíduos à normalidade shapiro-francia
residuos = model_lin.resid
stat, p = shapiro(residuos)
print(p) 
print(stat)
# %%
# Plotando um histograma da variável residuos
plt.hist(residuos, bins=30)
plt.title('Histograma dos Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frequência')
# # Calculando a média e o desvio padrão dos resíduos
# media_residuos = np.mean(residuos)
# desvio_padrao_residuos = np.std(residuos)
# # Gerando a curva normal teórica
# curva_normal = np.random.normal(media_residuos, desvio_padrao_residuos, 1000)
# # Plotando a curva normal teórica
# plt.plot(curva_normal, 
#          norm.pdf(curva_normal, loc=media_residuos, scale=desvio_padrao_residuos), 
#          color='red', 
#          label='Curva Normal Teórica')
plt.show()
# %%
