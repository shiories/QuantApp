import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from backend.TransfomeHandler import *
from backend.StockDownloader import *
from backend.IndicatorsCalculator import *
from backend.DataPreprocessor import *
from backend.ModelEvaluation import *
from backend.EnsembleEvaluation import *

class ModelPreparation:

    def __init__(self, stock_name, start_datetime, end_datetime, freq, unit, x_interval, min_quantity, attributes=None):

        # 初始化 TransfomeHandler
        self.transfome_handler = TransfomeHandler()
        self.stock_name = self.transfome_handler.handle_symbols(stock_name)
        
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.x_interval = x_interval
        self.min_quantity = min_quantity
        self.offset, self.frequency, self.interval = self.transfome_handler.handle_frequency(freq,unit)
        self.stock_downloader = StockDownloader(stock_name)
        if attributes:
            self.Indicators_calculator = IndicatorsCalculator(**attributes)
        else:
            self.Indicators_calculator = IndicatorsCalculator(overlapping=True, momentum=False, volume=False, cycle=False, price_transform=False, volatility=False, pattern=False)
        self.results = []  # 用於保存每次迴圈的結果


    def create_features(self):

        # 下載股票資料
        self.historical_data = self.stock_downloader.range(self.start_datetime, self.end_datetime, self.interval)
        #print("historical_data",self.historical_data)
        
        # 計算股票指標
        self.indicators_data = asyncio.run(self.Indicators_calculator.get_indicators(self.historical_data))
        #print("indicators_data",self.indicators_data)

    def create_model(self, Symbol, indicators_group, historical_group):

        # 處理資料集
        data_preprocessor = DataPreprocessor(self.x_interval)
        X_train, X_test, y_train, y_test, X, y, scaler, median_values = data_preprocessor.training_data(indicators_group, historical_group, )

        # 配對回歸模型
        regressor_evaluation = ModelEvaluation(X_train, X_test, y_train, y_test, X, y, model_type = "Regressor")
        test_results, selectors, regressors, selected_features_df, var_selectors = asyncio.run(regressor_evaluation.evaluate_models(Symbol))

        # 生成集成模型        
        ensemble_evaluation = EnsembleEvaluation(X_train, X_test, y_train, y_test, test_results, X, y, selectors, regressors, var_selectors, model_type = "Regressor")
        voting_results, trained_models = asyncio.run(ensemble_evaluation.evaluate_models(Symbol))
        
        # 選擇最佳模型
        best_model = self.transfome_handler.handle_model(Symbol, voting_results, self.min_quantity, trained_models, scaler, median_values)


        # 將結果保存到列表中
        self.results.append({
            'Symbol': Symbol,
            'best_model': best_model,
            'scaler': scaler,
            'median_values': median_values,
            'selected_features_df': selected_features_df,
            'test_results': test_results,
            'voting_results': voting_results
        })

        return 

    def run(self):
        self.create_features()

        # 提取股票指標資料
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_stock, stock_name) for stock_name in self.stock_name]
            for future in futures:
                future.result()
        
        self.voting_results = self.get_results(voting_results=True)
        self.test_results = self.get_results(test_results=True)
        self.selected_features_df = self.get_results(selected_features_df=True)

        
        
        return self.results, self.historical_data, self.indicators_data, self.voting_results, self.test_results, self.selected_features_df

    def process_stock(self, stock_name):
        # 提取股票指標資料
        indicators_group = self.indicators_data[self.indicators_data['Symbol'] == stock_name]
        indicators_group = indicators_group.drop(columns='Symbol')
        # 提取股票價格資料
        historical_group = self.historical_data.xs(key=stock_name, level=1, axis=1, drop_level=False)
        historical_group = historical_group.droplevel(1, axis=1)

        # 創建並評估模型
        self.create_model(stock_name, indicators_group, historical_group)

    def get_results(self, best_model=False, voting_results=False, test_results=False, selected_features_df=False):

        if best_model:
            best_model = [result for result in self.results if result['best_model'] is not None]
            return best_model
        
        if voting_results:
            voting_results = pd.concat([result['voting_results'] for result in self.results], keys=[result['Symbol'] for result in self.results], names=['Symbol'])
            return voting_results

        if test_results:
            test_results = pd.concat([result['test_results'] for result in self.results], keys=[result['Symbol'] for result in self.results], names=['Symbol'])
            return test_results

        if selected_features_df:
            selected_features_df = pd.concat([result['selected_features_df'] for result in self.results], keys=[result['Symbol'] for result in self.results], names=['Symbol'])
            return selected_features_df




    

if __name__=="__main__":
    initial_cash_dict = {'1442.TW': 46696, '2301.TW': 199522, '2364.TW': 7426, '2376.TW': 92814, '2382.TW': 233574, '3017.TW': 232386, '3701.TW': 71390, '6235.TW': 116192}
    #initial_cash_dict = {'2330.TW': 500000, '2454.TW': 100000, '2302.TW': 200000,
    #                     '2308.TW': 150000, '2329.TW': 250000, '2338.TW': 300000, '2340.TW': 400000,
    #                     '2351.TW': 120000, '2363.TW': 180000,'2369.TW': 220000}
    initial_cash_dict = {'2330.TW': 500000, '2317.TW': 100000}

    start_datetime = "2017-03-10" #預設為09:00:00
    end_datetime = "2024-03-20" #預設為13:30:00
    freq = "d" # 間隔：1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    unit = 1
    x_interval = 1
    min_quantity = 3
     # 建立ModelPreparation實例並執行
    model_prep = ModelPreparation(initial_cash_dict, start_datetime, end_datetime, freq, unit,  x_interval, min_quantity)
    results, historical_data, indicators_data, voting_results, test_results, selected_features_df = model_prep.run()

    for df in [voting_results, test_results, selected_features_df]:
        print(df)






