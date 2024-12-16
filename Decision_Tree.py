# -*- coding: utf-8 -*-
# In[0.1]: Instalação dos pacotes

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install statstests

# In[0.2]: Importação dos pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
#from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
                                                        #logístico multinomial
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#%% 
#############################################################################
#                      ARVORE DE DECISAO                                    #                  
#               EXEMPLO 1 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################

df_downtime = pd.read_csv('TOP_40000Completo.csv',delimiter=',')
df_downtime = pd.read_excel('top_10000.xlsx')
df_downtime


# Características das variáveis do dataset
df_downtime.info()

# Estatísticas univariadas
describe = df_downtime.describe()

#%% Criando variavel downtime 0 ou 1

df_downtime.loc[df_downtime['TotalDownTimeInSec'] > 300 , 'downtime'] = 1
df_downtime.loc[df_downtime['TotalDownTimeInSec'] <= 300, 'downtime'] = 0
df_downtime['downtime'] = df_downtime['downtime'].astype('int64')

df_downtime = df_downtime.dropna(axis=1)

df_downtime = df_downtime.loc[:, (df_downtime != 0).any(axis=0)]

#df_downtime.to_csv('top_1000_semnull.csv', index=False)
#%% Tabela de frequências absolutas da variável 'downtime'

df_downtime['downtime'].value_counts().sort_index()

#%% Criando função do validador de treino e teste

def executar_validador(X, y):
  validador = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
  for treino_id, teste_id in validador.split(X, y):
    X_train, X_test = X[treino_id], X[teste_id]
    y_train, y_test = y[treino_id], y[teste_id]
  return X_train, X_test, y_train, y_test

#%% Criando função Classificador 

def executar_classificador(classificador, X_train, X_test, y_train):
  arvore = classificador.fit(X_train, y_train)
  y_pred = arvore.predict(X_test)
  return y_pred

#%% Criando função para salvar a arvore gerada

def salvar_arvore(classificador, nome):
  plt.figure(figsize=(200,100))
  tree.plot_tree(classificador, filled=True, fontsize=14)
  plt.savefig(nome)
  plt.close()

#%% Criando função para validar os resultados da arvore

def validar_arvore(y_test, y_pred):
  print("Accuracy" , accuracy_score(y_test, y_pred))
  print("Precision" , precision_score(y_test, y_pred))
  print("Recall" ,  recall_score(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))

#%% Gerando dados de treino e teste

X = df_downtime.drop(['downtime','LineRunDownTimeInSec', 'TotalDownTimeInSec','DownTimeInSec', 'ThreadDownTimeInSec', 'TailoutDownTimeInSec', 'RestockDownTimeInSec' ], axis=1).values
y = df_downtime.iloc[:,-1]

X_train,X_test, y_train, y_test = executar_validador(X, y)
     

#%% Executando e validando

#execucao do classificador DecisionTreeClassifier
classificador_arvore_decisao = tree.DecisionTreeClassifier(max_depth=10, random_state=0)

y_pred_arvore_decisao = executar_classificador(classificador_arvore_decisao, X_train, X_test, y_train)
     

#criacao da figura da arvore de decisao
salvar_arvore(classificador_arvore_decisao, "arvore_decisao1.png")
     

#validacao arvore de decisao
validar_arvore(y_test, y_pred_arvore_decisao)


