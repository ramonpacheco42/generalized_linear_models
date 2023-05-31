# %%
import rpy2.robjects as robjects
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2, shapiro, boxcox
from scipy import stats
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from rpy2.robjects import pandas2ri
# %%
# Configurando o arquivo RData
robjects.pandas2ri.activate()
# Carregando o arquivo RData
robjects.r['load']('empresas.RData')
# %%
# Lendo o dataframe carregado acima.
pandas2ri.activate()
dados = robjects.r['empresas']
# Ajustando o dataframe para que seja apresentado corretamente.
df = pd.DataFrame.from_records(dados).transpose()
df = df.rename(columns={0: 'empresa', 1 : 'retorno',2 : 'disclosure', 3 : 'endividamento', 4 : 'ativos', 5 : 'liquídez'})
# %%
# Tratando as colunad do df
df['retorno'] = df['retorno'].astype(float)
df['disclosure'] = df['disclosure'].astype(int)
df['endividamento'] = df['endividamento'].astype(float)
df['ativos'] = df['ativos'].astype(int)
df['liquídez'] = df['liquídez'].astype(float)
# %%
# Visualizando o dataframe
df.head(10)
# %%
# Plotando gráfico para conhecer a correlação de pearson entre as variáveis
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# %%
# Plotando Gráfico de distribuição das variáveis
sns.pairplot(df,corner=True)
plt.show()
# %%
# Outra forma de apresentar a distribuição das variáveis
sns.pairplot(df, kind="kde")
plt.show()
# %%
# Estimando modelo de regressão linear múltiplo
# Organizando as variáveis Y(Alpha) e X(Betha)
y = df["retorno"]
X = df[["disclosure", "endividamento", "ativos", "liquídez"]]
# Ajustando as variáveis independentes
X = sm.add_constant(X)
# Ajustando a regressão linear
model = sm.OLS(y, X).fit()
# Imprimindo os resultados
# Observe que a variável individamento não passa em um nível de confiança de 95%
print(model.summary())
# %%
# Estimando outro modelo sem a variável individamento(que não passou no teste de confiança de 95%)
# Ajustando a regressão linear
model2 = sm.OLS(y, X).fit()
# Imprimindo os resultados
print(model2.summary())
# Básicamente o professor refaz o modelo retirando as demais variáveis para mostrar que
# ao tirar uma variável muda totalmente o teste estatistico f-value.
# Optei por não repetir esses passos e ir direto para o modelo stepwise
# %%
# Para fim didaticos vamos mostrar calculo do K realizado na aula, porém na função SFS
# sabemos que k = 2 quando confiança é 95% e igual a 3 quando é 90%
# Calculando o k
valor_critico = chi2.ppf(q=0.95, df=1)
probabilidade_acumulada = chi2.cdf(x=valor_critico, df=1)
# Criando uma instância do modelo de regressão linear
lr = LinearRegression()
# Definindo as variáveis explicativas (X) e a variável de resposta (y)
X = df[['disclosure', 'endividamento', 'ativos', 'liquídez']]
y = df['retorno']
# Stepwise
# Criando um objeto SFS para realizar a seleção sequencial de variáveis
sfs = SFS(lr, 
          k_features=2)
# Executando o SFS
sfs = sfs.fit(X, y)
# Resultado
print('Variáveis selecionadas: {}'.format(sfs.k_feature_names_))
# %%
# Gerando modelo de regressão linear múlplico final
y = df["retorno"]
X = df[["disclosure", "liquídez"]]
# Ajustando as variáveis independentes
X = sm.add_constant(X)
# Ajustando a regressão linear
model_final = sm.OLS(y, X).fit()
print(model_final.summary())
# %%
# Teste de verificação da aderência dos resíduos à normalidade shapiro-francia
residuos = model_final.resid
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
# Usando a função "boxcox" para transformar a variável "comprimento"
retorno_boxcox, lambda_boxcox = stats.boxcox(df['retorno'])
# %%
df['predict_bc'] = (((retorno_boxcox*lambda_boxcox)+1)**(1/lambda_boxcox))
# %%
lambda_boxcox
# %%
# 03:07:00