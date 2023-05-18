import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_absolute_error
import util
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import os
from sklearn.decomposition import PCA

absolute_path = os.path.dirname(__file__)
relative_path, r_path = "train.csv", "test.csv"
train_path, test_path = os.path.join(absolute_path, relative_path), os.path.join(absolute_path, r_path) 
traindata, testdata = pd.read_csv(train_path, sep = ","), pd.read_csv(test_path, sep = ",")

y = traindata.pop('Danceability')
useless = ['Uri', 'Url_spotify', 'Url_youtube', 'Description', 'Title']
X, Xtest = traindata.drop(useless, axis=1), testdata.drop(useless, axis=1)

ONEHOTarr = ['Album_type','Licensed','official_video']
for feature in ONEHOTarr:
    tmp1, tmp2 = util.OneHotEncoding(X.pop(feature), y, Xtest.pop(feature))
    X, Xtest = pd.concat([X, tmp1], axis=1), pd.concat([Xtest, tmp2], axis=1)
    
Catarr = ['Track', 'Album', 'Channel', 'Composer', 'Artist']
for feature in Catarr:
    X[feature], Xtest[feature] = util.CatBoostEncoding(X[feature], y, Xtest[feature])

X, Xtest = util.fillna(X, y, Xtest)

X, Xtest = util.Standardize(X, Xtest)

'''models'''
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
print(mae)

model = LogisticRegression(random_state=0, max_iter=100).fit(X, y)
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
print(mae)
y_pred = model.predict(Xtest)
util.OutputCSV(y_pred) # 2.21