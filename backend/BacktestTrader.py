import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from backend.TradingSystem import *
from backend.StockDownloader import *
from backend.TradingAnalyzer import *
from backend.IndicatorsCalculator import *
from backend.TransfomeHandler import *
from backend.DataPreprocessor import *
from backend.ModelPreparation import *

class BacktestTrader:
    
    def __init__(self, initial_cash_dict, multiplier, freq, unit, x_interval, start_datetime, end_datetime, results, historical_data, min_score, attributes=None, master=None):

        # 計算 mean_dict
        mean_prices = historical_data["Close"].mean()
        trade_unit={}
        for symbol in mean_prices.index:
            trade_unit[symbol] = round(initial_cash_dict[symbol] / multiplier / mean_prices[symbol] / 1000, 3) 
            
        self.attributes = attributes
        self.master = master
        #初始化 TransfomeHandler
        self.transfome_handler = TransfomeHandler()
        # 初始化 TradingSystem
        self.trading_system = TradingSystem(initial_cash_dict,  trade_unit)
        # 初始化 StockDownloader
        self.stock_downloader = StockDownloader(initial_cash_dict,historical_data)
        # 初始化 TradingAnalyzer
        self.trading_analyzer = TradingAnalyzer(self.trading_system, initial_cash_dict, start_datetime, end_datetime)
        # 初始化 IndicatorsCalculator
        if self.attributes:
            self.indicators_calculator = IndicatorsCalculator(**self.attributes)
        else:
            self.indicators_calculator = IndicatorsCalculator(overlapping=True, momentum=False, volume=False, cycle=False, price_transform=False, volatility=False, pattern=False)
        self.results = []  # 用於保存每次迴圈的結果
        # 頻率處理
        self.offset, self.frequency, self.interval = self.transfome_handler.handle_frequency(freq, unit)
        # 時間處理
        self.start_datetime = self.transfome_handler.handle_datetime(start_datetime)
        self.end_datetime = self.transfome_handler.handle_datetime(end_datetime, is_endtime=True)
        # 生成日期範圍
        self.date_range = self.transfome_handler.handle_date_range(self.start_datetime, self.end_datetime, self.frequency)
        #接收ModelPreparation變數
        self.results = results
        self.x_interval = x_interval
        self.y_interval = x_interval
        self.min_score = min_score
        
        
    async def run(self):
        # 回測迴圈
        count = 0
        for current_time in self.date_range:
            count += 1
            self.current_time = current_time
            end_time = self.current_time + self.offset
            
            # 下載股票數據
            prices_df = self.stock_downloader.range(self.current_time, end_time, self.interval)
            if len(prices_df) > 1500:         
                prices_df = prices_df.iloc[-1500:]
                
            if prices_df.empty:
                continue
            else:
                self.historical_data = self.stock_downloader.save_data(print_data=True)
                
                if self.attributes:
                    self.indicators_calculator = IndicatorsCalculator(**self.attributes)
                else:
                    self.indicators_calculator = IndicatorsCalculator(overlapping=True, momentum=False, volume=False, cycle=False, price_transform=False, volatility=False, pattern=False)
                self.indicators_data = await self.indicators_calculator.get_indicators(self.historical_data)
                
                self.trading_system.get_data(prices_df)
                self.get_prediction()

            if count % 10 == 0 or current_time >= self.date_range[-3]:
                await self.get_table()

    def get_prediction(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_result, result) for result in self.results]
            for future in futures:
                future.result() 

    def process_result(self, result):
        symbol = result['Symbol']
        model = result['best_model']['Model']
        average_score = result['best_model']['Score']

        if average_score <= self.min_score:
            return
        # 提取股票指標資料
        indicators_group = self.indicators_data[self.indicators_data['Symbol'] == symbol]
        indicators_group = indicators_group.drop(columns='Symbol')
        
        if len(indicators_group) > 1500: 
           indicators_group = indicators_group.iloc[-1500:]
        
        # 處理資料集
        data_preprocessor = DataPreprocessor(self.x_interval)
        features_df = data_preprocessor.forecast_data(indicators_group)
        feature = features_df.iloc[[-1]]

        # 進行預測        
        predictions = float(model.predict(feature)[0])
        predictions = round(predictions * 1000 / self.x_interval, 0)

        # 下單
        if self.date_range[-5] <= self.current_time <= self.date_range[-1]:
            print(f'--------------------{self.current_time}執行平倉--------------------')
            asyncio.run( self.trading_system.liquidate_position(symbol, self.current_time))
        elif predictions != 0:
            asyncio.run( self.trading_system.execute_trade(symbol, predictions, self.current_time))

    async def get_table(self):
        # 分析表格並列印統計結果
        statistics_df = await self.trading_analyzer.analyze_statistics()
        trade_df, holdings_df, cost_df, portfolio_df = self.trading_system.get_tables_df( trade=True, holdings=True, cost=True, portfolio=True )
        print(statistics_df)
        return statistics_df, trade_df, holdings_df, cost_df, portfolio_df




if __name__=="__main__":

    initial_cash_dict = {'1442.TW': 46696, '2301.TW': 199522, '2364.TW': 7426, '2376.TW': 92814, '2382.TW': 233574, '3017.TW': 232386, '3701.TW': 71390, '6235.TW': 116192}
    initial_cash_dict = {'3231.TW': 621002, '5215.TW': 151829, '6230.TW': 227169}

    freq = 'd'   # 間隔：1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    unit = 1
    start_datetime = "2016-01-01 09:30:00" #預設為09:00:00
    end_datetime = "2022-12-31 11:30:00" #預設為13:30:00
    
    x_interval = 1
    min_quantity = 3

     # 建立ModelPreparation實例並執行
    model_prep = ModelPreparation(initial_cash_dict, start_datetime, end_datetime, freq,  unit,  x_interval, min_quantity)
    results, historical_data, indicators_data, voting_results, test_results, selected_features_df = model_prep.run()

    #print("--------------------------------------------------------------------------------------")
    #voting_results = model_prep.get_results(voting_results=True)
    #print(voting_results)

    #print("--------------------------------------------------------------------------------------")
    #test_results = model_prep.get_results(test_results=True)
    #print(test_results)

    #print("--------------------------------------------------------------------------------------")
    #selected_features_df = model_prep.get_results(selected_features_df=True)
    #print(selected_features_df)

    print("--------------------------------------------------------------------------------------")

    

    # 設定時間範圍
    start_datetime = "2023-01-01 09:00:00"
    end_datetime = "2023-03-31 12:00:00"
    
    min_score = 0.5
    # 交易單位元元
    multiplier=100
    # 創建 TradingManager 實例，並指定 freq、unit、start_datetime 和 end_datetime
    trading_backtest = BacktestTrader(initial_cash_dict, multiplier, freq, unit, x_interval, start_datetime,
                                       end_datetime, results, historical_data, min_score)

    # 運行非同步函數
    asyncio.run(trading_backtest.run())

    # 分析並列印統計結果
    statistics_df, trade_df, holdings_df, cost_df, portfolio_df = trading_backtest.get_table()

    import pandas as pd
    import re



    # 處理每個 DataFrame 並寫入 Excel 檔
    for df_name in ['statistics_df', 'trade_df', 'holdings_df', 'cost_df', 'portfolio_df']:
        df = globals().get(df_name)
        if df is not None:
                        
            # 處理日期時間列
            for col in df.columns:
                if re.search(r'\[ns, .*?\]$', str(df[col].dtype)):
                        df[col] = df[col].dt.tz_localize(None)
                        
            print(f'{df_name}:\n{df.info()}')
            df.to_excel(f'{df_name}.xlsx')


