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