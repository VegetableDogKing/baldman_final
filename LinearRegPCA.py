import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import mean_absolute_error
import util
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import os
from sklearn.decomposition import PCA

model_name = 'linear_regression_PCA'  # <----------------記得改檔名
absolute_path = os.path.dirname(__file__)
relative_path = "train.csv"
train_path = os.path.join(absolute_path, relative_path) 

data = pd.read_csv(train_path, sep = ",")


y = data['Danceability']
data.pop("Danceability")
arr = []
X, arr = util.preprocess(data, arr, y)

pca = PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)

print(pca.explained_variance_ratio_)

model = LinearRegression().fit(X, y) # 可改
model_path = os.path.join(absolute_path, "models\\")
joblib.dump(model, model_path + model_name)  
y_pred = model.predict(X)
mae_on_train = mean_absolute_error(y, y_pred)

scores = -cross_val_score(model, X, y, cv=5, scoring = 'neg_mean_absolute_error')


print("%0.8f MAE with a standard deviation of %0.8f" % (scores.mean(), scores.std()))


file_name = 'linear_regression_PCA' # 把模型輸在這裡
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
X = util.preprocess(data, enc_arr)[0]
X = pca.transform(X)

''''輸出.csv'''
y_pred = loaded_model.predict(X)
label = range(17170,23485)
df = {"id":label, "Danceability": y_pred}
df = pd.DataFrame(df)
OutputCSV(df)

