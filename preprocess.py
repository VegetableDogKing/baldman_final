import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("ML\FinalProject\\train.csv", sep = ",")

target_encode = ['Album_type','Licensed','official_video']
# Create new columns in clicks using preprocessing.LabelEncoder()
encoder = LabelEncoder()
for feature in target_encode:
    encoded = encoder.fit_transform(data[feature])
    data[feature] = encoded

tags = data.columns
for i in range(17):
    data[tags[i]].fillna(np.mean(data[tags[i]]), inplace = True)
data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Danceability']] = data[['Key', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Danceability']].round(0)


print(collections.Counter(data['Album_type'])) # {'album': 10379, 'single': 3732, nan: 2560, 'compilation': 499}
'''print(collections.Counter(data['Licensed']))  # {True: 10260, False: 4317, nan: 2593}
print(collections.Counter(data['official_video'])) # {True: 11491, False: 3064, nan: 2615}
print(collections.Counter(data['Track']))
print(collections.Counter(data['Album']))
print(collections.Counter(data['Channel']))
print(collections.Counter(data['Composer']))
print(collections.Counter(data['Artist']))'''

X = data[tags[1:16]]
y = data[tags[0]]
print(X.loc[0])