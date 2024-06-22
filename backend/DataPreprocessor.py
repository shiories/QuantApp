import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import time

class DataPreprocessor:
    
    def __init__(self, x_interval=None):

        self.x_interval = x_interval
        self.y_interval = x_interval

    def training_data(self, indicators_data, historical_data):

        # 添加 historical_data 目標值到 indicators_data
        data_set = indicators_data

        data_set.insert(loc=len(data_set.columns), column='Return', value=historical_data['Adj Close'])
        data_set['Return'] = data_set['Return'].shift(-1)  # 向下位移一行

        # 將 index 轉換為列
        data_set.reset_index(inplace=True)

        # 去掉 'Date' 列
        data_set = data_set.drop(data_set.columns[0], axis=1)

        # 獲取要轉換的列（不包括非數值列）
        #data_set = data_set.select_dtypes(include=['float64']).columns
        data_set = data_set.astype('float64')

        # 轉換並顯式設置資料類型
        data_set.iloc[:, :-1] = data_set.iloc[:, :-1].transform(lambda x: (x - x.shift(self.x_interval)) / x.shift(self.x_interval))

        # 對目標列進行 pct_change 計算，與前 y_interval 行比較
        data_set.iloc[:, -1] = data_set.iloc[:, -1].transform(lambda x: (x - x.shift(self.y_interval)) / x.shift(self.y_interval))
        data_set.dropna(inplace=True)

        # 處理無窮大值
        data_set = data_set.replace([np.inf, -np.inf], np.nan)
        # 使用中位數替換 NaN 值
        median_values = data_set.median()  # 計算中位數
        data_set = data_set.fillna(value=median_values)

        # 創建正規化器
        scaler = RobustScaler()
        # 選擇 DataFrame 中的所有列，排除最後一列
        cols_to_normalize = data_set.columns[:-1]
        # 對所選列進行正規化
        data_set[cols_to_normalize] = scaler.fit_transform(data_set[cols_to_normalize])
        # 定義閾值
        threshold = 3
        # 使用布林索引選擇未超過閾值的資料
        data_set = data_set[(np.abs(data_set) < threshold).all(axis=1)]

        # X 包含全部列去掉最後一列
        X = data_set.iloc[:, :-1]
        # y 包含最後一列
        y = data_set.iloc[:, -1]
        # 切分訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(X.copy(), y.copy(), test_size=0.2, shuffle=False, random_state=42)

        return X_train, X_test, y_train, y_test, X, y, scaler, median_values

    
    def forecast_data(self, indicators_data):

        # 添加 historical_data 目標值到 indicators_data
        data_set = indicators_data

        # 將 index 轉換為列
        data_set.reset_index(inplace=True)

        # 去掉 'Date' 列
        data_set = data_set.drop(data_set.columns[0], axis=1)

        # 獲取要轉換的列（不包括非數值列）
        data_set = data_set.astype('float64')

        # 轉換並顯式設置資料類型
        data_set = data_set.transform(lambda x: (x - x.shift(self.x_interval)) / x.shift(self.x_interval))

        # 處理無窮大值
        data_set = data_set.replace([np.inf, -np.inf], np.nan)

        # 使用中位數替換 NaN 值
        median_values = data_set.median()  # 計算中位數
        data_set = data_set.fillna(value=median_values)

        # 創建正規化器
        scaler = RobustScaler()

        # 對所選列進行正規化
        data_set = scaler.fit_transform(data_set)

        # 定義閾值
        threshold = 3

        # 使用布林索引選擇未超過閾值的資料
        data_set = data_set[(np.abs(data_set) < threshold).all(axis=1)]
        
        # 使用iloc排除第一列，並獲取第二列之後的列標題
        columns_to_use = indicators_data.columns[1:]
        data_set = pd.DataFrame(scaler.fit_transform(data_set), columns=columns_to_use)

        return data_set





