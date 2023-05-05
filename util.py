import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("ML\FinalProject\\train.csv", sep = ",")

def preprocess(data: pd.DataFrame, encode_arr = [], y = None):
    target_encode = ['Album_type','Licensed','official_video', 'Track', 'Album', 'Channel', 'Composer', 'Artist']
    data['Album_type'].fillna('album', inplace = True)
    data['Licensed'].fillna(True, inplace = True)
    data['official_video'].fillna(True, inplace = True)
    tags = data.columns
    for i in range(13):
        data[tags[i]].fillna(np.mean(data[tags[i]]), inplace = True)
    data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream']] = data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream']].round(0)
    scaler = StandardScaler()
    if len(encode_arr) == 0:
        for i in target_encode:
            target_enc = ce.TargetEncoder(cols=i, drop_invariant=True)
            target_enc.fit(data[i], y)
            data[i] = target_enc.transform(data[i])
            encode_arr.append(target_enc)  
        X = data[tags[0:12]].join(data[target_encode])
        X = scaler.fit_transform(X)     
    else:
        for i in range(len(target_encode)):
            data[target_encode[i]] = encode_arr[i].transform(data[target_encode[i]])
        X = data[tags[0:12]].join(data[target_encode])
        X = scaler.fit_transform(X) 
    
    return X, encode_arr
    
def performance():
    pass

'''
print(collections.Counter(data['Album_type'])) # {'album': 10379, 'single': 3732, nan: 2560, 'compilation': 499}
print(collections.Counter(data['Licensed']))  # {True: 10260, False: 4317, nan: 2593}
print(collections.Counter(data['official_video'])) # {True: 11491, False: 3064, nan: 2615}
print(len(collections.Counter(data['Track'])))
print(len(collections.Counter(data['Album'])))
print(len(collections.Counter(data['Channel'])))
print(len(collections.Counter(data['Composer'])))
print(len(collections.Counter(data['Artist'])))
'''