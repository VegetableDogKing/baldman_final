import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
absolute_path = os.path.dirname(__file__)
relative_path = "train.csv"
file_path = os.path.join(absolute_path, relative_path) 
data = pd.read_csv(file_path, sep = ",")

def preprocess(data: pd.DataFrame, encode_arr = [], y = None):
    target_encode = ['Album_type','Licensed','official_video', 'Track', 'Album', 'Channel', 'Composer', 'Artist']
    tags = data.columns
    for i in range(13):
        data[tags[i]].fillna(np.mean(data[tags[i]]), inplace = True)  
    data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream']] = data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream']]
    scaler = StandardScaler()
    if len(encode_arr) == 0:
        for i in target_encode:
            target_enc = ce.TargetEncoder(cols=i, drop_invariant=True)
            target_enc.fit(data[i], y)
            data[i] = target_enc.transform(data[i])
            encode_arr.append(target_enc)  
        X = data[tags[0:13]].join(data[target_encode])
        X.pop("Track")
        X.pop("Album")
        X.pop("Channel")
        X.pop("Composer")
        X.pop("Artist")
        X = scaler.fit_transform(X)     
    else:
        for i in range(len(target_encode)):
            data[target_encode[i]] = encode_arr[i].transform(data[target_encode[i]])
        X = data[tags[0:13]].join(data[target_encode])
        X.pop("Track")
        X.pop("Album")
        X.pop("Channel")
        X.pop("Composer")
        X.pop("Artist")
        X = scaler.fit_transform(X) 

    return X, encode_arr

def fillna(X: pd.DataFrame, y: pd.DataFrame, Xtest:pd.DataFrame):
    return X, Xtest

def OneHotEncoding(X: pd.DataFrame, y: pd.DataFrame, Xtest:pd.DataFrame): 
    oneHotEncoding = ce.OneHotEncoder()
    oneHotEncoding.fit(X, y)
    X = oneHotEncoding.transform(X)
    Xtest = oneHotEncoding.transform(Xtest)
    return X, Xtest

def CatBoostEncoding(X: pd.DataFrame, y: pd.DataFrame, Xtest:pd.DataFrame):
    oneHotEncoding = ce.CatBoostEncoder()
    oneHotEncoding.fit(X, y)
    X = oneHotEncoding.transform(X)
    Xtest = oneHotEncoding.transform(Xtest)
    return X, Xtest
    
def Standardize(X: pd.DataFrame, Xtest:pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xtest = scaler.fit_transform(Xtest)
    return X, Xtest

def correlation(df_feat: pd.DataFrame):
    corr_df = df_feat.corr()
    fig = plt.figure(figsize=(20, 8))
    sns.heatmap(corr_df,vmax = 1, annot=True,cmap='Blues',fmt='.2f')
    plt.xticks(rotation = 90, fontsize = 8)
  
    # plt.show()
    
    corr_df = corr_df[0:1]
    tags = corr_df.columns
    for i in range(len(tags)):
        if(abs(corr_df[tags[i]]['Danceability'])) < 0.09 or (abs(corr_df[tags[i]]['Danceability'])) > 0.7:
            df_feat.pop(tags[i])
    tags = df_feat.columns
    df_feat = np.array(df_feat)
    
    return df_feat, tags

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