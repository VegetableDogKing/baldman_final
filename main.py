import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import util
from sklearn import metrics
import lightgbm as lgb
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
'''
linear_model = util.LinearReg(X, y)
util.CVScore(X, y, linear_model)

# util.OutputCSV(y_pred)  # 2.21 on logistic

gbr_model = util.GBR(X, y)
util.CVScore(X, y, gbr_model)
# util.ModelPrediction(Xtest, None, gbr_model, is_testdata=True, file_name='GBR_1100_4')

hgbr_model = util.HGBR(X, y)
util.CVScore(X, y, hgbr_model) # 1.1
'''
gbdt_model = util.GBDT(X, y)
util.CVScore(X, y, gbdt_model)
util.ModelPrediction(Xtest, None, gbdt_model, is_testdata=True, file_name='GBDT_1100_4')