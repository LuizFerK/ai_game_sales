import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

# Carrega os dados do arquivo CSV
data = pd.read_csv('assets/Video_Games_Sales_as_at_22_Dec_2016.csv')

# Aplica formatação 2.a
# Retira a primeira coluna (nome do jogo) pois é insignificante para análise
data = data.iloc[:, 1:]

# User score tem 2425 linhas com valor "tbd" (to be determined), eu substituo aqui por valores nulos
data.replace('tbd', np.nan, inplace=True)

# Aplica formatação 2.b
imputer = SimpleImputer(strategy='mean')

colunas_nulas = ['Year_of_Release', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
data[colunas_nulas] = imputer.fit_transform(data[colunas_nulas])

# Aplica formatação 2.c
kbins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

colunas_continuas = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
data[colunas_continuas] = kbins.fit_transform(data[colunas_continuas])

# Aplica formatação 2.d
onehotencoder = OneHotEncoder(sparse_output=False, drop='first')

string_columns = ['Platform', 'Publisher', 'Developer', 'Rating']
result = onehotencoder.fit_transform(data[string_columns])

encoded_df = pd.DataFrame(result, columns=onehotencoder.get_feature_names_out(string_columns))
data = pd.concat([data, encoded_df], axis=1)
data = data.drop(string_columns, axis=1)

# Separa rótulo do dataframe
# a coluna Genre tem dois valores nulos no dataset então eu dou um dropna pra limpar eles
label = 'Genre'
data = data.dropna()
X = data.drop([label], axis=1)
y  = data[label]

# Separo 30% do dataframe para teste e 70% para treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

# Teste
y_pred = mlp.predict(X_test)

# Acurácia
accuracy = accuracy_score(y_test, y_pred)
print("\nAcurácia do modelo no conjunto de teste:", accuracy)
