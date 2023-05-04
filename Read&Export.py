import numpy as np
import joblib
import pandas as pd
#產出CSV檔                
def OutputCSV(data):   
    Result ='ML\FinalProject\\output.csv'
    data.to_csv(Result, index=False)
    print('Export Succeeded: ' + Result)

'''讀檔'''
loaded_model = joblib.load('GBDT2')  # 把模型輸在這裡
data = pd.read_csv("ML\FinalProject\\test.csv", sep = ",") #測試檔案

tags = data.columns
for i in range(13):
    data[tags[i]].fillna(np.mean(data[tags[i]]), inplace = True)
data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream']] = data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream']].round(0)
X = data[tags[0:12]]

''''輸出.csv'''
y_pred = loaded_model.predict(X)
label = range(17170,23485)
df = {"id":label, "Danceability": y_pred}
df = pd.DataFrame(df)
OutputCSV(df)