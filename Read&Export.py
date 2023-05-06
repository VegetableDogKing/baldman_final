import numpy as np
import joblib
import pandas as pd
from util import preprocess
file_name = 'GBR_1100_10' # 把模型輸在這裡

#產出CSV檔                
def OutputCSV(data: pd.DataFrame):   
    Result ='ML\FinalProject\\outputs\\' + file_name + '.csv'
    data.to_csv(Result, index=False)
    print('Export Succeeded: ' + Result)

'''讀檔'''
loaded_model = joblib.load('ML\FinalProject\\models\\' + file_name)  
enc_arr = joblib.load('ML\FinalProject\\EncodeArray')
data = pd.read_csv("ML\FinalProject\\test.csv", sep = ",") #測試檔案
data_train = pd.read_csv("ML\FinalProject\\train.csv", sep = ",")
X = preprocess(data, enc_arr)[0]

''''輸出.csv'''
y_pred = loaded_model.predict(X)
label = range(17170,23485)
df = {"id":label, "Danceability": y_pred}
df = pd.DataFrame(df)
OutputCSV(df)