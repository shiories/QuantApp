import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, time
from backend.TransfomeHandler import *



class StockDownloader:
    
    def __init__(self, stock_names, historical_data=None):
        # 設定 yfinance 的日誌級別為 CRITICAL，以忽略錯誤信息
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        
        #初始化 TransfomeHandler
        self.transfome_handler = TransfomeHandler()
        
        self.symbols = self.transfome_handler.handle_symbols(stock_names)
        self.market_open_time = time(9, 0)
        self.market_close_time = time(13, 30)
        if historical_data is not None:
            self.historical_data = historical_data
        else:
            self.historical_data = pd.DataFrame()


    def save_data(self, data=pd.DataFrame(), print_data=False):
        if not data.empty:
            pass
        elif print_data is True:
            return self.historical_data
        elif data.empty:
            return

        if self.historical_data.empty:
            self.historical_data = data
        else:
            # 檢查列標題是否相同
            if not data.columns.equals(self.historical_data.columns):
                raise ValueError("新下載的資料列標題不相同。")
            # 檢查資料是否重複
            if data.index.isin(self.historical_data.index).any():
                raise ValueError("新下載的資料中包含重複的索引。")
            # 連接資料
            self.historical_data = pd.concat([self.historical_data, data])


    def range(self, start_date, end_date, interval="1d", symbols=None):
        if symbols is not None:
            symbols = self.transfome_handler.handle_symbols(symbols)
        else:
            symbols = self.symbols
            
        # 處理起始日期和結束日期
        start_date = self.transfome_handler.handle_datetime(start_date)
        end_date = self.transfome_handler.handle_datetime(end_date, is_endtime =True)

        prices_df = yf.download(symbols, start=start_date, end=end_date, interval=interval, progress=False, threads=True)
        if not isinstance(prices_df.columns, pd.MultiIndex):
            multi_index = pd.MultiIndex.from_product([prices_df.columns, symbols], names=[ None, 'Symbol'])
            prices_df.columns = multi_index

        self.save_data(prices_df)
        return prices_df


    def frequency(self, start_date, end_date, freq='m', unit=1, symbols=None,market_open=None, market_close=None):
        if symbols is not None:
            symbols = self.transfome_handler.handle_symbols(symbols)
        else:
            symbols = self.symbols

        freq_mapping = {
            'm': {'frequency': f'{unit}min', 'interval': f'{unit}m', 'offset': pd.DateOffset(minutes=unit)},
            'h': {'frequency': f'{unit}h', 'interval': f'{unit}h', 'offset': pd.DateOffset(hours=unit)},
            'd': {'frequency': f'{unit}d', 'interval': f'{unit}d', 'offset': pd.DateOffset(days=unit)},
        }

        if freq not in freq_mapping:
            raise ValueError(f"無效頻率: {freq}")

        freq_info = freq_mapping[freq]
        frequency = freq_info['frequency']
        interval = freq_info['interval']
        offset = freq_info['offset']


        # 設置開盤和收盤時間，如果未提供則使用類別中的預設值
        market_open = market_open or time(9, 0)
        market_close = market_close or time(13, 30)

        # 處理起始日期和結束日期
        start_date = self.transfome_handler.handle_datetime(start_date)
        end_date = self.transfome_handler.handle_datetime(end_date, is_endtime =True)

        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        print(f"start_datetime,{start_date},end_datetime,{end_date}")
        print(f"date_range,{date_range}")
        prices_df = pd.DataFrame()

        for single_date in date_range:
            # 檢查是否在開盤區間
            if market_open <= single_date.time() < market_close:
                end_date = single_date + offset
                print(f"single_date,{single_date},end_date,{end_date}")
                prices_df = yf.download(symbols, start=single_date, end=end_date, interval=interval, progress=False, threads=True)
                print(prices_df)
                self.save_data(prices_df)
        return prices_df





# 示例用法
if __name__=="__main__":
    # 代碼輸入示範
    stock_name = {'2330.TW': 500000, '2317.TW':500000}  
    #stock_name =  ['2330.TW', '2317.TW'] 
    #stock_name =  {'2330.TW', '2317.TW'}
    #stock_name =  '2330.TW'

    # 初始化 PriceDataDownloader
    stock_downloader = StockDownloader(stock_name)

    # 設定時間範圍示範
    start_datetime = datetime(2024, 2, 21, 12, 30, tzinfo=None)
    end_datetime = datetime(2024, 2, 22, 12, 30, tzinfo=None)

    start_datetime = datetime(2024, 2, 21, tzinfo=None) #預設為09:00:00
    end_datetime = datetime(2024, 2, 22, tzinfo=None) #預設為13:30:00

    start_datetime = "2024-02-21 10:30:00"
    end_datetime = "2024-02-22 12:30:00"

    start_datetime = "2024-03-10" #預設為09:00:00
    end_datetime = "2024-03-20" #預設為13:30:00



    # 示例：下載特定時間範圍的數據
    interval = "1d" #interval間隔：1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    date = stock_downloader.range(start_datetime, end_datetime, interval)
    print(date)
    
    '''
    # 示例：不斷更新並下載特定時間範圍的數據
    freq = "m" # m、h、d
    unit = 5 # 依freq而定
    date = stock_downloader.frequency(start_datetime, end_datetime, freq, unit)
    print(date)
    '''

