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
relative_path = "train.csv"
train_path = os.path.join(absolute_path, relative_path) 

data = pd.read_csv(train_path, sep = ",")
model_name = 'GBR_1100_10_pca' # <----------------記得改檔名

y = data['Danceability']
data.pop("Danceability")
arr = []
X, arr = util.preprocess(data, arr, y)
joblib.dump(arr, absolute_path + '//EncodeArray')

# # plot
pca = PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)
print(X)

x_train, x_test, y_train, y_test = train_test_split(X, y)
gbdt = GradientBoostingRegressor(loss='absolute_error', learning_rate=0.01, n_estimators=1100, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=10
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )

model = gbdt.fit(X, y)
model_path = os.path.join(absolute_path, "models\\")
joblib.dump(model, model_path + model_name)  
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

file_name = 'GBR_1100_10_pca' # 把模型輸在這裡
absolute_path = os.path.dirname(__file__)
output_path = os.path.join(absolute_path, "outputs\\") 
model_path = os.path.join(absolute_path, "models\\")
enc_path = os.path.join(absolute_path, "EncodeArray")
test_path = os.path.join(absolute_path, "test.csv") 
train_path = os.path.join(absolute_path, "train.csv") 

#產出CSV檔                
def OutputCSV(data: pd.DataFrame):  
    Result = output_path + file_name + '.csv'
    data.to_csv(Result, index=False)
    print('Export Succeeded: ' + Result)

'''讀檔'''
loaded_model = joblib.load(model_path + file_name)  
enc_arr = joblib.load(enc_path)
data = pd.read_csv(test_path, sep = ",") #測試檔案
data_train = pd.read_csv(train_path, sep = ",")

y_train = data_train['Danceability']
data_train.pop("Danceability")
arr = []
X_train, arr = util.preprocess(data_train, arr, y_train)

pca = PCA(n_components=10)
pca.fit(X_train)
X_train = pca.transform(X_train)

X = util.preprocess(data, enc_arr)[0]
X = pca.transform(X)


''''輸出.csv'''
y_pred = loaded_model.predict(X)
label = range(17170,23485)
df = {"id":label, "Danceability": y_pred}
df = pd.DataFrame(df)
OutputCSV(df)


'''
GBDT
n = 100 depth = 5
Accuracy on train data = 0.7452046284072377
MAE on train data = 0.7736273976857964
Accuracy on val data = 0.7452046284072377
MAE on val data = 1.87724202189611


GBR

'''