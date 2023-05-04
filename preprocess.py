import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import category_encoders as ce
data = pd.read_csv("ML\FinalProject\\train.csv", sep = ",")

target_encode = ['Album_type','Licensed','official_video', 'Track', 'Album', 'Channel', 'Composer', 'Artist']
data['Album_type'].fillna('album', inplace = True)
data['Licensed'].fillna(True, inplace = True)
data['official_video'].fillna(True, inplace = True)
for i in target_encode:
    target_enc = ce.TargetEncoder(cols=i, drop_invariant=True)
    target_enc.fit(data[i], data['Danceability'])
    data[i] = target_enc.transform(data[i])

tags = data.columns
for i in range(14):
    data[tags[i]].fillna(np.mean(data[tags[i]]), inplace = True)
data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Danceability']] = data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Danceability']].round(0)
X = data[tags[1:13]].join(data[target_encode])
y = data[tags[0]]

print(len(X.loc[0]))

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