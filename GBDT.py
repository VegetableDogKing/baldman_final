import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_absolute_error
import util
from sklearn.model_selection import cross_val_score
from sklearn import metrics

data = pd.read_csv("ML\FinalProject\\train.csv", sep = ",")
model_name = 'GBR_100_5' # <----------------記得改檔名

y = data['Danceability']
data.pop("Danceability")
arr = []
X, arr = util.preprocess(data, arr, y)
joblib.dump(arr, 'ML\FinalProject\\EncodeArray')

# plot
features = ['Danceability','Energy','Key','Loudness','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Duration_ms','Views','Likes','Stream',
            'Album_type','Licensed','official_video', 'Track', 'Album', 'Channel', 'Composer', 'Artist']
dtwhole = pd.DataFrame(np.column_stack((X, y)), columns=features)
util.correlation_plot(dtwhole)


x_train, x_test, y_train, y_test = train_test_split(X, y)
gbdt = GradientBoostingRegressor(loss='absolute_error', learning_rate=0.1, n_estimators=2000, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=5
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )

model = gbdt.fit(X, y)
joblib.dump(model, 'ML\FinalProject\\models\\' + model_name)  
y_train_pred = gbdt.predict(x_train)
acc_train = gbdt.score(x_train, y_train)
mae_on_train = mean_absolute_error(y_train, y_train_pred)
y_test_pred = gbdt.predict(x_test)
acc_test = gbdt.score(x_test, y_test)
mae_on_test = mean_absolute_error(y_test, y_test_pred)

print(f'Accuracy on train data = {acc_train}')
print(f'MAE on train data = {mae_on_train}')
print(f'MAE on val data = {mae_on_test}')

scores = -cross_val_score(model, X, y, cv=5, scoring = 'neg_mean_absolute_error')
print("%0.8f MAE with a standard deviation of %0.8f" % (scores.mean(), scores.std()))


'''
GBDT
n = 100 depth = 5
Accuracy on train data = 0.7452046284072377
MAE on train data = 0.7736273976857964
Accuracy on val data = 0.7452046284072377
MAE on val data = 1.87724202189611


GBR

'''