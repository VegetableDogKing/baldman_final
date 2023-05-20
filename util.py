import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import category_encoders as ce
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
pd.set_option('mode.chained_assignment', None)
file_name = 'GBR_test'

absolute_path = os.path.dirname(__file__)
relative_path = "train.csv"
file_path = os.path.join(absolute_path, relative_path) 
outputs_path = os.path.join(absolute_path, 'outputs//')
data = pd.read_csv(file_path, sep = ",")

def fillna(X: pd.DataFrame, y: pd.DataFrame, Xtest:pd.DataFrame):
    for feature in X.columns:
        X[feature].fillna(np.mean(X[feature]), inplace = True)
        Xtest[feature].fillna(np.mean(Xtest[feature]), inplace = True)    
    return X, Xtest

def fill(data: pd.DataFrame, train_columns = []):
    for i in train_columns:
        if i not in data.columns:
            train_columns.remove(i)
            
    if data[train_columns[0]].isnull().sum() != 0:
        length = len(train_columns)
        train = data[train_columns]
        test = train[train[train_columns[0]].isnull()]
        for i in range(1, length, 1):
            if train_columns[i] in test.columns:
                test = test[test[train_columns[i]].notnull()]
        for i in range(0, length, 1):
            if train_columns[i] in train.columns:
                train = train[train[train_columns[i]].notnull()]
        
        train_y = train[train_columns[0]]
        train_X = train.drop(train_columns[0], axis=1)
        test_X = test.drop(train_columns[0], axis=1)
        
        if test_X.empty == False:
            lr = LinearRegression()
            lr.fit(train_X, train_y)
            y_pred = lr.predict(test_X)
            
            index = test.loc[test[train_columns[0]].isnull(), train_columns[0]].index
            for i in range(len(index)):
                data[train_columns[0]][index[i]] = y_pred[i]
                
    return data

def fillna_LinReg(data: pd.DataFrame):      
    Energy_columns = ['Energy', 'Loudness', 'Acousticness', 'Valence']
    Loudness_columns = ['Loudness', 'Danceability', 'Energy', 'Acousticness', 'Instrumentalness', 'Valence']
    Speechiness_columns = ['Speechiness', 'Danceability']
    Acousticness_columns = ['Acousticness', 'Danceability', 'Energy', 'Loudness', 'Instrumentalness', 'Valence']
    Instrumentalness_columns = ['Instrumentalness', 'Danceability', 'Loudness', 'Acousticness', 'Valence']
    Valence_columns = ['Valence', 'Danceability', 'Energy', 'Loudness', 'Acousticness', 'Instrumentalness']
           
    data = fill(data, Energy_columns)
    data = fill(data, Loudness_columns)
    data = fill(data, Speechiness_columns)
    data = fill(data, Acousticness_columns)
    data = fill(data, Instrumentalness_columns)
    data = fill(data, Valence_columns)
    
    return data

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

def Normalize(X: pd.DataFrame, Xtest:pd.DataFrame):
    pass

def _PCA(X: pd.DataFrame, Xtest:pd.DataFrame, components):
    pca = PCA(n_components=components)
    pca.fit(X)
    X, Xtest = pca.transform(X), pca.transform(Xtest)
    return X, Xtest

def CVScore(X: pd.DataFrame, y: pd.DataFrame, model): 
    scores = -cross_val_score(model, X, y, cv=5, scoring = 'neg_mean_absolute_error')
    print("%0.8f MAE with a standard deviation of %0.8f" % (scores.mean(), scores.std()))
    
def LinearReg(X: pd.DataFrame, y: pd.DataFrame):
    model = LinearRegression().fit(X, y)
    ModelPrediction(X, y, model)
    return model
    
def LogisticReg(X: pd.DataFrame, y: pd.DataFrame):
    model = LogisticRegression(max_iter=1000).fit(X, y)
    ModelPrediction(X, y, model)
    return model

def GBR(X: pd.DataFrame, y: pd.DataFrame):
    gbr = GradientBoostingRegressor(loss='absolute_error', learning_rate=0.05, n_estimators=1100, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=4
                                  , init=None, random_state=None, max_features=None
                                  , verbose=1, max_leaf_nodes=None, warm_start=False
                                  )
    model = gbr.fit(X, y)
    ModelPrediction(X, y, model)
    return model

def GBDT(X: pd.DataFrame, y: pd.DataFrame):
    gbdt = GradientBoostingClassifier(loss='log_loss', learning_rate=0.01, n_estimators=1000, subsample=1.0,
                                      criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_depth=4, min_impurity_decrease=0.0,
                                      init=None, random_state=None, max_features='sqrt', verbose=0, max_leaf_nodes=None,
                                      warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
    model = gbdt.fit(X, y)
    ModelPrediction(X, y, model)
    return model

def HGBR(X: pd.DataFrame, y: pd.DataFrame):
    hgbr = HistGradientBoostingRegressor(loss='squared_error', quantile=None, learning_rate=0.1, max_iter=1000,
                                         max_leaf_nodes=31, max_depth=4, min_samples_leaf=20, l2_regularization=0.0,
                                         max_bins=255, categorical_features=None, monotonic_cst=None, interaction_cst=None,
                                         warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1,
                                         n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
    model = hgbr.fit(X, y)
    ModelPrediction(X, y, model)
    return model

def ModelPrediction(X:pd.DataFrame, y:pd.DataFrame, model, is_testdata=False, file_name = None): 
    y_pred = model.predict(X)
    if is_testdata:
        OutputCSV(y_pred, file_name)
        return
    mae = mean_absolute_error(y, y_pred)
    print('train model mae = %0.8f' % mae)
    return y_pred

#產出CSV檔                
def OutputCSV(y_pred: pd.DataFrame, file_name):
    label = range(17170,23485)
    df = {"id":label, "Danceability": np.round(y_pred)}
    df = pd.DataFrame(df)
    Result = outputs_path + file_name + '.csv'
    df.to_csv(Result, index=False)
    print('Export Succeeded: ' + Result)
    return

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
