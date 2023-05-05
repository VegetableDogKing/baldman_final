import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_absolute_error
from util import preprocess
from sklearn.model_selection import cross_val_score
from sklearn import metrics

data = pd.read_csv("ML\FinalProject\\train.csv", sep = ",")
model_name = 'GBDT_100_5' # <----------------記得改檔名

y = data['Danceability']
data.pop("Danceability")
arr = []
X, arr = preprocess(data, arr, y)

joblib.dump(arr, 'ML\FinalProject\\EncodeArray')
x_train, x_test, y_train, y_test = train_test_split(X, y)

gbdt = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, n_estimators=100, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=5
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )

model = gbdt.fit(x_train, y_train)
joblib.dump(model, 'ML\FinalProject\\' + model_name)  
y_train_pred = gbdt.predict(x_train)
acc_train = gbdt.score(x_train, y_train)
mae_on_train = mean_absolute_error(y_train, y_train_pred)
y_test_pred = gbdt.predict(x_test)
acc_test = gbdt.score(x_test, y_test)
mae_on_test = mean_absolute_error(y_test, y_test_pred)

print(f'Accuracy on train data = {acc_train}')
print(f'MAE on train data = {mae_on_train}')
print(f'MAE on val data = {mae_on_test}')

''' n = 100 depth = 5
Accuracy on train data = 0.7452046284072377
MAE on train data = 0.7736273976857964
Accuracy on val data = 0.7452046284072377
MAE on val data = 1.87724202189611
'''