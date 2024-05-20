import asyncio
import logging
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import timedelta
from backend.OnlineCrawling import *



class StockPicker:
    
    def __init__(self, stock_df, file_name=str()):
        
        self.file_name = file_name
        if file_name.endswith('.xlsx'):
            # 字串以'.xlsx'結尾
            self.data = pd.read_excel(f"{file_name}")
            print(f"{file_name}載入完成!")
        else:
            self.data =pd.read_excel(f"{file_name}.xlsx", engine="openpyxl")
            print(f"{file_name}.xlsx載入完成!")

        self.stock_df = stock_df
        # 設定 yfinance 的日誌級別為 CRITICAL，以忽略錯誤信息
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        
        self.data.drop_duplicates(inplace=True) # 刪除重複的行
        self.data.reset_index(drop=True, inplace=True)  # 重置索引
        # 檢查 'Stock code' 列中的值是否為字串類型
        if self.data['Stock code'].dtype == 'object':
            # 如果 'Stock code' 列中的值都是字串類型，則執行相應操作
            self.data['Stock code'] = self.data['Stock code'].str.split(' ').str[0]
        else:
            self.data['Stock code'] = self.data['Stock code'].astype(str)


    async def download_data(self, stock_codes, date, retry_count=0):
        try:
            if retry_count >= 5:
                print(f"已達到最大重試次數，停止下載 {date} 的股票資料。")
                return None

            stock_codes_str = [f"{code}.TW" for code in stock_codes]
            print(f"正在下載 {date} 的 {len(stock_codes_str)} 檔股票資料...")

            data = yf.download(stock_codes_str, start=date, end=date + timedelta(days=1), progress=False)

            if data.empty:
                date = date + timedelta(days=1)
                data = await self.download_data(stock_codes, date, retry_count + 1)
            elif data.columns.nlevels == 2:
                missing_stock_codes = [code for code in stock_codes_str if code not in data['Close'].columns]
                if missing_stock_codes:
                    date = date + timedelta(days=1)
                    data = await self.download_data(missing_stock_codes, date, retry_count + 1)
            return data
            
        except Exception as e:
            print(f"下載 {date} 的股票資料時發生錯誤：{e}")
            return None


    async def extract_closing_price(self, data, stock_codes, current_date):
        try:
            closing_prices = pd.DataFrame(columns=['Stock code', '財報發布日', 'Closing Price'])
            if data.columns.nlevels == 2:
                for stock_code in stock_codes:
                    closing_price = data['Close', f'{stock_code}.TW'].iloc[0]
                    new_entry = pd.DataFrame({'Stock code': [stock_code], '財報發布日': [pd.Timestamp(current_date)], 'Closing Price': [closing_price]})
                    if closing_prices.empty:
                        closing_prices = new_entry
                    else:
                        closing_prices = pd.concat([closing_prices, new_entry], ignore_index=True)
            else:
                for stock_code in stock_codes:
                    closing_price = data['Close'].iloc[0]
                    closing_prices = pd.DataFrame({'Stock code': [stock_code], '財報發布日': [pd.Timestamp(current_date)], 'Closing Price': [closing_price]})
            return closing_prices
        except Exception as e:
            print(f"提取收盤價時發生錯誤：{e}")
            return pd.DataFrame(columns=['Stock code', '財報發布日', 'Closing Price'])


    async def process_row(self, date, stock_codes):
        try:
            data = await self.download_data(stock_codes, date)
            closing_prices = await self.extract_closing_price(data, stock_codes, date)
            return closing_prices
        except Exception as e:
            print(f"處理行時發生錯誤：{e}")
            return pd.DataFrame(columns=['Stock code', '財報發布日', 'Closing Price'])


    async def process_result_row(self, row, existing_data):
        stock_codes = row["Stock code"]
        report_date = row["財報發布日"]

        closes_df = pd.DataFrame(columns=['Stock code', '財報發布日', 'Closing Price'])
        # 檢查是否存在特徵6.xlsx中的收盤價資料，並在可用時使用它們
        if not existing_data.empty:
            if 'Closing Price' in existing_data.columns:
                existing_prices = existing_data.loc[(existing_data['Stock code'].isin(stock_codes)) & (existing_data['財報發布日'] == report_date)]

                if not existing_prices.empty:
                    closes_df = existing_prices[['Stock code', '財報發布日', 'Closing Price']]
                    # 從 stock_codes 中移除已存在的股票代碼
                    stock_codes = [code for code in stock_codes if code not in closes_df.loc[closes_df['Closing Price'].notna(), 'Stock code'].tolist()]

        # 繼續下載資料
        try:
            if stock_codes:
                report_date -= timedelta(days=1)  # 加1天
                data = await self.download_data(stock_codes, report_date)
                closing_prices = await self.extract_closing_price(data, stock_codes, report_date)
                # 如果 closes_df 為空，則直接使用 closing_prices
                if closes_df.empty:
                    closes_df = closing_prices
                else:
                    if closing_prices.empty:
                        closes_df = pd.DataFrame(columns=['Stock code', '財報發布日', 'Closing Price'])
                    else:
                        merged_df = pd.merge(closes_df, closing_prices[['Stock code', 'Closing Price']], on='Stock code', how='left')
                        merged_df['Closing Price'] = merged_df['Closing Price_y'].combine_first(merged_df['Closing Price_x'])
                        merged_df.drop(['Closing Price_x', 'Closing Price_y'], axis=1, inplace=True)
                        closes_df = merged_df

        except Exception as e:
            print(f"處理 {report_date} 行時發生錯誤：{e}")
            
        return closes_df


    async def get_closes(self, max_year, min_year):
       
        merged_data = pd.merge(self.data, self.stock_df, left_on='Stock code', right_on='證券代號', how='inner')
        merged_data["財報發布日"] = pd.to_datetime(merged_data["財報發布日"])
        merged_data = merged_data[(merged_data["財報發布日"].dt.year >= min_year) & (merged_data["財報發布日"].dt.year <= max_year)]
        #merged_data["財報發布日"] = merged_data["財報發布日"] - timedelta(days=1)
        result = merged_data.groupby('財報發布日')['Stock code'].apply(list).reset_index()

        tasks = [self.process_result_row(row, self.data) for _, row in result.iterrows()]
        closes_dfs = await asyncio.gather(*tasks)

        closes_dfs = [df for df in closes_dfs if not df.empty]
        closes_df = pd.concat(closes_dfs, ignore_index=True)

        await self.integrate_closing_prices(closes_df.copy())

        if 'Closing Price' in merged_data.columns:
            merged_data.drop(columns=['Closing Price'], inplace=True)
            
        merged_data = pd.merge(merged_data, closes_df, on=['財報發布日', 'Stock code'], how='left')
        return merged_data


    async def integrate_closing_prices(self, closes_df):
        # 檢查是否存在 'Closing Price' 列，如果不存在，則添加一列
        print(f'closes_df:\n{closes_df}')
        if 'Closing Price' not in self.data.columns:
            self.data['Closing Price'] = np.nan        

        # 將closes_df拆分成大小為500的塊
        chunks = [closes_df[i:i+500] for i in range(0, closes_df.shape[0], 500)]

        for chunk in chunks:
            # 將下載到的價格與現有資料合併
            self.data = self.data.join(chunk.set_index(['Stock code', '財報發布日'])[['Closing Price']], on=['Stock code', '財報發布日'], rsuffix='_closes')
            # 填充缺失值
            self.data['Closing Price'] = self.data['Closing Price'].fillna(self.data['Closing Price_closes'])
            # 刪除多餘的列
            self.data.drop(columns=['Closing Price_closes'], inplace=True)
            self.data.drop_duplicates(inplace=True) # 刪除重複的行
            self.data.reset_index(drop=True, inplace=True)  # 重置索引
            
        if self.file_name.endswith('.xlsx'):
            self.data.to_excel(f"{self.file_name}", index=False)
            print(f"資料已更新到 {self.file_name} 中。")
        else:
            self.data.to_excel(f"{self.file_name}.xlsx", index=False)
            print(f"資料已更新到 {self.file_name}.xlsx 中。")






if __name__ == "__main__":
    
    cutoff_year = 1995
    min_company_count = 24
    online_crawling = OnlineCrawling()
    stock_df , stock_list, industry_df = online_crawling.get_stock(cutoff_year, min_company_count, excluded_industries=None)
    print(stock_df)
    print(industry_df)

    stock_picker = StockPicker(stock_df, file_name="特徵6")
    max_year = 2024
    min_year = 2015
    closes_df = asyncio.run(stock_picker.get_closes(max_year, min_year))
    print("每日收盤價:")
    print(closes_df)





