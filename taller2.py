# ---------- IMPORTS ----------
#General
import tensorflow
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification

#Keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb


X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

#Cambio para taller de analítica
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=12, stratify=y)

T = 20
ada = adaboost(X_train, y_train, X_val, y_val, T)
ada_metrics = adaboost_predict(ada["modelos"], ada["alphas"], X_train, y_train, X_val, y_val, ada["D"])

def entrena_stumps(X,y,D):
  stump = DecisionTreeClassifier(max_depth=1,max_leaf_nodes=2,random_state=12)
  stump.fit(X,y,sample_weight=D)
  return stump

def adaboost(X,y,X_val,y_val,T):
  N = len(y)
  D = np.ones(N)/N
  alpha = np.zeros(T)
  stumps = []
  accuracy = np.zeros(T)
  accuracy_val = np.zeros(T)
  error_pes = np.zeros(T)
  error = np.zeros(T)
  D_total = []
  for i in range(T):
    D_total.append(D)
    error_temp = 0
    stump = entrena_stumps(X,y,D)
    y_out = stump.predict(X)
    y_val_out = stump.predict(X_val)
    accuracy[i] = accuracy_score(y, y_out)
    accuracy_val[i] = accuracy_score(y_val, y_val_out)
    stumps.append(stump)
    eta = 0
    for j in range(N):
      error_temp += D[j]*np.exp(-y[j]*y_out[j])
      if y_out[j] != y[j]:
        eta += D[j]
    error_pes[i] = error_temp
    error[i] = eta
    alpha[i] = 0.5*np.log((1-eta)/eta)
    for j in range(N):
      D[j] = D[j]*np.exp(-alpha[i]*y[j]*y_out[j])
    D = D/np.sum(D)
  return {"modelos":stumps,"alphas":alpha,"precision":accuracy,"precision_val":accuracy_val,"error":error,"error_pes":error_pes,"D":D_total}

def adaboost_predict(models, alphas, X, y, X_val, y_val,D_vec):
  N = X.shape[0]
  N_val = X_val.shape[0]
  T = len(models)
  margen_train = np.zeros(N)
  margen_val = np.zeros(N_val)
  train_acc = np.zeros(T)
  val_acc = np.zeros(T)
  train_error = np.zeros(T)
  margen_raw = np.zeros(N)
  cont = 0
  for i in range(T):
    stump = models[i]
    alpha = alphas[i]
    margen_train += alpha*stump.predict(X)
    margen_val += alpha*stump.predict(X_val)
    pred_train = np.sign(margen_train)
    pred_val = np.sign(margen_val)
    train_acc[i] = np.mean(pred_train == y)
    val_acc[i] = np.mean(pred_val == y_val)
    D = D_vec[i]
    pred = np.sign(pred_train)
    train_error[i] = np.sum(D*(pred_train != y))

  return {"modelo":np.sign(margen_train),"modelo_val":np.sign(margen_val),"error":train_error,"precision":train_acc,"precision_val":val_acc}

def adaboost_best(X,y,T):
  N = len(y)
  D = np.ones(N)/N
  alpha = np.zeros(T)
  stumps = []
  accuracy = np.zeros(T)
  accuracy_val = np.zeros(T)
  error_pes = np.zeros(T)
  error = np.zeros(T)
  D_total = []
  for i in range(T):
    D_total.append(D)
    error_temp = 0
    stump = entrena_stumps(X,y,D)
    y_out = stump.predict(X)
    accuracy[i] = accuracy_score(y, y_out)
    stumps.append(stump)
    eta = 0
    for j in range(N):
      error_temp += D[j]*np.exp(-y[j]*y_out[j])
      if y_out[j] != y[j]:
        eta += D[j]
    error_pes[i] = error_temp
    error[i] = eta
    alpha[i] = 0.5*np.log((1-eta)/eta)
    for j in range(N):
      D[j] = D[j]*np.exp(-alpha[i]*y[j]*y_out[j])
    D = D/np.sum(D)
  return {"modelos":stumps,"alphas":alpha,"precision":accuracy,"error":error,"error_pes":error_pes,"D":D_total}
print("Adaboost implementation functions")



clf = RandomForestClassifier(random_state=12,bootstrap=True,n_estimators=499,min_samples_leaf=2)
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]]))

xg_clf = xgb.XGBClassifier(n_estimators = 461,eta = 0.005 ,use_label_encoder=False, eval_metric='logloss', random_state=12, n_jobs=-1,tree_method="hist")
xg_clf.fit(X, y)
print(xg_clf.predict([[0, 0, 0, 0]]))



