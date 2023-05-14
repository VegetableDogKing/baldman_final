# To-Do List
1. 加權
2. 刪離群值
3. transformation
4. encoding 改一下
5. desciption

# Report 架構
1. 三種方法: Linear Regression, Softmax, XGboost
2. 選擇feature方法: 相關係數、PCA、DAE
3. evaluation: cross validation score

# 更新動態
### 2023/5/4  
* 測試 n_estimator = 2000, max_depth = 6  
### 2023/5/5  
* 把preprocess新增到util裡面，讓code比較好看
* 把檔名修改放到前面，從上面改就可以了  
* 可以把GBDT改成 from sklearn.ensemble import GradientBoostingRegressor，這樣就不會是整數，但是參數還要在調一下
* to-do: 幫我寫一下sklearn.kfold/cross-validate來分資料(直接寫在util.py裡面就可以了)，現在還是用train 75% test 25%。應該要分成10個fold然後每個參數跑10次，可以放著讓她跑，記得把每個參數的MAE存下來之後好畫圖
