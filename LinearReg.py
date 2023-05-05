import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import mean_absolute_error
from util import preprocess
from sklearn.model_selection import cross_val_score
from sklearn import metrics

model_name = 'linear_regression'  # <----------------記得改檔名
data = pd.read_csv("ML\FinalProject\\train.csv", sep = ",")


y = data['Danceability']
data.pop("Danceability")
arr = []
X, arr = preprocess(data, arr, y)

model = LinearRegression().fit(X, y) # 可改
joblib.dump(model, 'ML\FinalProject\\models\\' + model_name)  
y_pred = model.predict(X)
mae_on_train = mean_absolute_error(y, y_pred)

scores = -cross_val_score(model, X, y, cv=5, scoring = 'neg_mean_absolute_error')


print("%0.8f MAE with a standard deviation of %0.8f" % (scores.mean(), scores.std()))