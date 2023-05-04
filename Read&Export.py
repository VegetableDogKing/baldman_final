import numpy as np
import joblib
import pandas as pd
import category_encoders as ce

#產出CSV檔                
def OutputCSV(data):   
    Result ='ML\FinalProject\\20features.csv'
    data.to_csv(Result, index=False)
    print('Export Succeeded: ' + Result)

'''讀檔'''
loaded_model = joblib.load('ML\FinalProject\GBDTwith20features')  # 把模型輸在這裡
data = pd.read_csv("ML\FinalProject\\test.csv", sep = ",") #測試檔案
data_train = pd.read_csv("ML\FinalProject\\train.csv", sep = ",")

target_encode = ['Album_type','Licensed','official_video', 'Track', 'Album', 'Channel', 'Composer', 'Artist']
data['Album_type'].fillna('album', inplace = True)
data['Licensed'].fillna(True, inplace = True)
data['official_video'].fillna(True, inplace = True)
for i in target_encode:
    target_enc = ce.TargetEncoder(cols=i, drop_invariant=True)
    target_enc.fit(data_train[i], data_train['Danceability'])
    data[i] = target_enc.transform(data[i])

tags = data.columns
for i in range(13):
    data[tags[i]].fillna(np.mean(data[tags[i]]), inplace = True)
data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream']] = data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream']].round(0)
X = data[tags[0:12]].join(data[target_encode])
y = data[tags[0]]

''''輸出.csv'''
y_pred = loaded_model.predict(X)
label = range(17170,23485)
df = {"id":label, "Danceability": y_pred}
df = pd.DataFrame(df)
OutputCSV(df)