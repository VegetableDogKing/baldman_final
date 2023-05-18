import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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


y = traindata['Danceability']
traindata.pop("Danceability")
X, Xtest = traindata, testdata

ONEHOTarr = ['Album_type','Licensed','official_video']
for feature in ONEHOTarr:
    X[feature], Xtest[feature] = util.OneHotEncoding(X[feature], y, Xtest[feature])
Catarr = ['Track', 'Album', 'Channel', 'Composer', 'Artist']
for feature in ONEHOTarr:
    X[feature], Xtest[feature] = util.CatBoostEncoding(X[feature], y, Xtest[feature])

X, Xtest = util.fillna(traindata, testdata)
