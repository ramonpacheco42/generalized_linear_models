# %%
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro, norm, boxcox
from scipy import stats
from rpy2.robjects import pandas2ri
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
plt.show()
# %%
# Plotando um gráfico dos resíduos
plt.scatter(predict_lin, residuos)
plt.axhline(y=0, color='r', linestyle='-', linewidth=3)
plt.xlabel('Fitted Values')
plt.ylabel('Resíduos')
plt.title('Distribuição dos Resíduos vs Fitted Values')
plt.show()
# %%
# Usando a função "boxcox" para transformar a variável "comprimento"
comprimento_boxcox, lambda_boxcox = stats.boxcox(df['comprimento'])

# Adicionando os valors obtidos no cálculo do Box-Cox no datafram.
df['bc_comprimento'] = comprimento_boxcox
df
# %%
# Estimando um novo modelo OLS com variável dependente transformada por Box-Cox
# Criando modelo para as resíduos:
model_bc = sm.OLS(df['bc_comprimento'], x).fit()
print(model_bc.summary())
# %%
# Teste de Shapiro-Francia para os resíduos do modeo_bc
fitted_data, lambda_bc = boxcox(np.array(df['comprimento']))
residuos_bc = model_bc.resid
stat, p = shapiro(residuos_bc)
print(p) 
print(stat)
print(lambda_bc)
# %%
# Plotando um histograma da variável residuos
plt.hist(residuos_bc, bins=30)
plt.title('Histograma dos Residuos_bc')
plt.xlabel('Residuos_bc')
plt.ylabel('Frequência')
plt.show()
# %%
# Plotando um gráfico dos resíduos
plt.scatter(fitted_data, residuos_bc)
plt.axhline(y=0, color='r', linestyle='-', linewidth=3)
plt.xlabel('Fitted Values')
plt.ylabel('Resíduos')
plt.title('Distribuição dos Resíduos vs Fitted Values')
plt.show()
# %%
# Gravando o modelo Box Cox no Dataset
df['predict_bc'] = (((fitted_data*lambda_bc)+1)**(1/lambda_bc))
df['predict'] = predict_lin
df
# %%
# 
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['comprimento'], df['predict'], color="yellow", alpha=0.6, s=50)
ax.scatter(df['comprimento'], df['predict_bc'], color="#440154FF", alpha=0.6, s=50)
ax.plot(df['comprimento'], df['comprimento'], color='gray', lw=2, ls='--')

ax.set_xlabel('Comprimento')
ax.set_ylabel('Fitted Values')
ax.set_title('Valores Previstos (fitted values) X Valores Reais')

ax.legend(labels=['Modelo Linear', 'Modelo Box-Cox'])
plt.show()
# %%
