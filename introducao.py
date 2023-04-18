#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
# %%
df = pd.read_excel('tempodist.xls')
df
# %%
# Calculando a reta de regressão
z = np.polyfit(df['Distância'], df['Tempo'], 1)
p = np.poly1d(z)

# Prevendo os valores
y_pred = p(df['Distância'])

# Calculando o R²
r2 = r2_score(df['Tempo'], y_pred)

# Plotando o gráfico de dispersão
plt.scatter(df['Distância'], df['Tempo'], c=df['Distância'], cmap='viridis')

# Adicionando uma reta de regressão
plt.plot(df['Distância'], p(df['Distância']), "r--")

# Configurando o título e os rótulos dos eixos
plt.title(f'R²={r2:.2f}')
plt.xlabel('Distância')
plt.ylabel('Tempo')
plt.xlim([0, max(df['Distância'])])
plt.ylim([0, max(df['Tempo'])])

# Exibindo o gráfico
plt.show()
# %%
# Análise descritiva dos dados
df.describe()
# %%
# Separando as variáveis independentes e dependentes
X = df[['Distância']]
y = df['Tempo']

# Criando um objeto de regressão linear
reg = LinearRegression()

# Treinando o modelo com os dados
reg.fit(X, y)

# Imprimindo os coeficientes da reta de regressão
print(f'Beta: {reg.coef_} Alfa: {reg.intercept_} R²: {r2:.4f}')
# %%
# Adicionando colunas para y^ e erro no dataframe
df['y^'] = y_pred
df['erro'] = df['Tempo'] - y_pred

# Exibindo o dataframe com as colunas adicionadas
print(df)
# %%
corr = df['Tempo'].corr(df['Distância'], method='pearson')
print(f'A correlação entre a variável Tempo e Distância é de {corr}')
