#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#%%
df_salarios = pd.read_csv('data/salarios.csv')
df_salarios

# %%
# Caracteristicas do dataset
df_salarios.info()
# %%
# Estatisticas Descritivas
df_salarios.describe()
# %%
corr1 = df_salarios[['rh1','econometria1']].corr()
corr1

plt.figure(figsize=(15,10))
sns.heatmap(corr1, annot= True, cmap= plt.cm.viridis,
            annot_kws={'size':27})
# %%
# Correlação Perfeita
modelo_1 = sm.OLS.from_formula('salario ~ rh1 + econometria1',
                               df_salarios).fit()
# %%
# Parametros do modelo
modelo_1.summary()
# %%
# Correlação Baixa

corr3 = df_salarios[['rh3','econometria3']].corr()
corr3

plt.figure(figsize=(15,10))
sns.heatmap(corr3, annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':27})
# %%
# Diagnóstico de multicolinearidade (Variance Inflation Factor e Tolerance)

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df_salarios[['rh3','econometria3']]
X = sm.add_constant(X)

vif = pd.Series([variance_inflation_factor(X.values, i)
                 for i in range(X.shape[1])],index=X.columns)
vif

tolerance = 1/vif
tolerance

pd.concat([vif,tolerance], axis=1, keys=['VIF', 'Tolerance'])
# %%
#CORRELAÇÃO MUITO ALTA, PORÉM NÃO PERFEITA:

corr2 = df_salarios[['rh2','econometria2']].corr()
corr2

plt.figure(figsize=(15,10))
sns.heatmap(corr2, annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':27})
# %%
# Estimando um modelo com variáveis preditoras com correlação quase perfeita
modelo_2 = sm.OLS.from_formula('salario ~ rh2 + econometria2',
                               df_salarios).fit()

# Parâmetros do modelo
modelo_2.summary()
# %%
# Diagnóstico de multicolinearidade (Variance Inflation Factor e Tolerance)

X = df_salarios[['rh2','econometria2']]
X = sm.add_constant(X)

vif = pd.Series([variance_inflation_factor(X.values, i)
                 for i in range(X.shape[1])],index=X.columns)
vif

tolerance = 1/vif
tolerance

pd.concat([vif,tolerance], axis=1, keys=['VIF', 'Tolerance'])
# %%
#DIAGNÓSTICO DE HETEROCEDASTICIDADE EM MODELOS DE REGRESSÃO
df_saeb_rend = pd.read_csv("data/saeb_rend.csv", delimiter=',')
df_saeb_rend
# %%
#Características das variáveis do dataset
df_saeb_rend.info()
# %%
#Estatísticas univariadas
df_saeb_rend.describe()
# %%
#Tabela de frequências absolutas das variáveis 'uf' e rede'
df_saeb_rend['uf'].value_counts()
df_saeb_rend['rede'].value_counts()
# %%
#Plotando 'saeb' em função de 'rendimento', com linear fit
x = df_saeb_rend['rendimento']
y = df_saeb_rend['saeb']
plt.plot(x, y, 'o', color='#FDE725FF', markersize=5, alpha=0.6)
sns.regplot(x="rendimento", y="saeb", data=df_saeb_rend)
plt.title('Dispersão dos dados com linear fit')
plt.xlabel('rendimento')
plt.ylabel('saeb')
plt.show()
# %%
# Plotando 'saeb' em função de 'rendimento',
#com destaque para 'rede' escolar

sns.scatterplot(x="rendimento", y="saeb", data=df_saeb_rend,
                hue="rede", alpha=0.6, palette = "viridis")
plt.title('Dispersão dos dados por rede escolar')
plt.xlabel('rendimento')
plt.ylabel('saeb')
plt.show()
# %%
# Plotando 'saeb' em função de 'rendimento',
#com destaque para 'rede' escolar e linear fits

sns.lmplot(x="rendimento", y="saeb", data=df_saeb_rend,
           hue="rede", ci=None, palette="viridis")
plt.title('Dispersão dos dados por rede escolar e com linear fits')
plt.xlabel('rendimento')
plt.ylabel('saeb')
plt.show()
# %%
# Estimação do modelo de regressão e diagnóstico de heterocedasticidade

# Estimando o modelo
modelo_saeb = sm.OLS.from_formula('saeb ~ rendimento', df_saeb_rend).fit()

# Parâmetros do modelo
modelo_saeb.summary()
# %%
# Função para o teste de Breusch-Pagan para a elaboração
# de diagnóstico de heterocedasticidade

# Criação da função 'breusch_pagan_test'

from scipy import stats
import numpy as np

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value
# %%
# Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_saeb)
#Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

#H0 do teste: ausência de heterocedasticidade.
#H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de
#variável relevante!
# %%
# Dummizando a variável 'uf'

df_saeb_rend_dummies = pd.get_dummies(df_saeb_rend, columns=['uf'],
                                      drop_first=True)

df_saeb_rend_dummies.head(10)
# %%
# Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_saeb_rend_dummies.drop(columns=['municipio',
                                                        'codigo',
                                                        'escola',
                                                        'rede',
                                                        'saeb']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "saeb ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

modelo_saeb_dummies_uf = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_saeb_rend_dummies).fit()

#Parâmetros do modelo
modelo_saeb_dummies_uf.summary()
# %%
# Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_saeb_dummies_uf'

breusch_pagan_test(modelo_saeb_dummies_uf)
# %%
# Plotando 'saeb' em função de 'rendimento',
#com destaque para UFs e linear fits

sns.lmplot(x="rendimento", y="saeb", data=df_saeb_rend,
           hue="uf", ci=None, palette="viridis")
plt.title('Dispersão dos dados por UF e com linear fits')
plt.xlabel('rendimento')
plt.ylabel('saeb')
plt.show()
# %%
#############################################################################
#               REGRESSÃO NÃO LINEAR MÚLTIPLA COM DUMMIES                   #
#               EXEMPLO 08 - CARREGAMENTO DA BASE DE DADOS                  #
#############################################################################

df_planosaude = pd.read_csv("data/planosaude.csv", delimiter=',')
df_planosaude

#Características das variáveis do dataset
df_planosaude.info()

#Estatísticas univariadas
df_planosaude.describe()
# %%
# Transformação da variável 'plano' para o tipo categórico

df_planosaude['plano'] = df_planosaude['plano'].astype('category')
df_planosaude['plano']
# %%
# Tabela de frequências absolutas da variável 'plano'

df_planosaude['plano'].value_counts()
# %%
# Distribuições das variáveis, scatters, valores das correlações e suas
#respectivas significâncias
def corrfunc(x, y, **kws):
    (r, p) = pearson(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

plt.figure(figsize=(15,10))
graph = sns.pairplot(df_planosaude.loc[:,"despmed":"renda"], diag_kind="kde")
graph.map(corrfunc)
plt.show()
# %%
# Dummizando a variável 'plano'

df_planosaude_dummies = pd.get_dummies(df_planosaude, columns=['plano'],
                                      drop_first=True)

df_planosaude_dummies.head(100)
# %%
# Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_planosaude_dummies.drop(columns=['id',
                                                         'despmed']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "despmed ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

modelo_planosaude = sm.OLS.from_formula(formula_dummies_modelo,
                                        df_planosaude_dummies).fit()

#Parâmetros do modelo
modelo_planosaude.summary()
# %%
# Procedimento Stepwise

# Instalação e carregamento da função 'stepwise' do pacote
#'stepwise_process.statsmodels'
#pip install "stepwise-process==2.5"
# Autores: Helder Prado Santos e Luiz Paulo Fávero
from stepwise_process.statsmodels import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_planosaude = stepwise(modelo_planosaude, pvalue_limit=0.05)
# %%
# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiroFrancia' do pacote
#'sfrancia'
# Autores: Luiz Paulo Fávero e Helder Prado Santos
#pip install sfrancia==1.0.8
from sfrancia import shapiroFrancia
shapiroFrancia(modelo_step_planosaude.resid)
# %%
# Plotando os resíduos do 'modelo_step_planosaude',
#com curva normal teórica

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_planosaude.resid, fit=norm, kde=True, bins=15)
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequências', fontsize=16)
plt.show()
# %%
# Kernel density estimation (KDE) - forma não-paramétrica para estimar
#a função densidade de probabilidade de uma variável aleatória

plt.figure(figsize=(15,10))
sns.kdeplot(data=modelo_step_planosaude.resid, multiple="stack",
            color='#55C667FF')
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequências', fontsize=16)
plt.show()
# %%
# Função para o teste de Breusch-Pagan para a elaboração
# de diagnóstico de heterocedasticidade

# Criação da função 'breusch_pagan_test'

#from scipy import stats

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value
# %%
# Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_step_planosaude)
#Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

#H0 do teste: ausência de heterocedasticidade.
#H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de
#variável relevante!
# %%
