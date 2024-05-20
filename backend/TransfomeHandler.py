import pytz  
import pandas as pd
from datetime import datetime, time



class TransfomeHandler:

    def __init__(self):
        pass
    
    
    def handle_frequency(self, freq, unit=1):

        # 頻率處理
        freq_mapping = {
            'm': {'frequency': f'{unit}min', 'interval': f'{unit}m', 'offset': pd.DateOffset(minutes=unit)},
            'h': {'frequency': f'{unit}h', 'interval': f'{unit}h', 'offset': pd.DateOffset(hours=unit)},
            'd': {'frequency': f'{unit}d', 'interval': f'{unit}d', 'offset': pd.DateOffset(days=unit)},
        }

        if freq not in freq_mapping:
            raise ValueError(f"無效頻率: {freq}")

        freq_info = freq_mapping[freq]
        self.offset = freq_info['offset']  # 將 offset 設定為類別成員變數
        self.frequency = freq_info['frequency']
        self.interval = freq_info['interval']
        
        return self.offset, self.frequency, self.interval


    def handle_datetime(self, input_datetime, default_time=time(9, 0), is_endtime=False, timezone='Asia/Taipei'):
        # 時間處理
        if isinstance(input_datetime, datetime):
            # 如果是 datetime，檢查時間是否在 09:00:00 到 13:30:00 之間
            if time(9, 0) <= input_datetime.time() <= time(13, 30):
                return input_datetime.astimezone(pytz.timezone(timezone))  # 转换时区
            else:
                return input_datetime.replace(hour=default_time.hour, minute=default_time.minute, second=default_time.second).astimezone(pytz.timezone(timezone))  # 加上 09:00:00 或 13:30:00 并设置时区
        elif isinstance(input_datetime, str):
            # 如果是字串，嘗試轉換為 datetime
            try:
                datetime_obj = datetime.strptime(input_datetime, '%Y-%m-%d %H:%M:%S')
                # 檢查時間是否在 09:00:00 到 13:30:00 之間
                if time(9, 0) <= datetime_obj.time() <= time(13, 30):
                    return datetime_obj.astimezone(pytz.timezone(timezone))  # 转换时区
                else:
                    return datetime_obj.replace(hour=default_time.hour, minute=default_time.minute, second=default_time.second).astimezone(pytz.timezone(timezone))  # 加上 09:00:00 或 13:30:00 并设置时区
            except ValueError:
                # 如果轉換失敗，嘗試只包含日期的格式，並預設時間為 09:00:00 或 13:30:00
                try:
                    datetime_obj = datetime.strptime(input_datetime, '%Y-%m-%d')
                    return datetime_obj.replace(hour=default_time.hour, minute=default_time.minute, second=default_time.second).astimezone(pytz.timezone(timezone)) if not is_endtime else datetime_obj.replace(hour=13, minute=30, second=0).astimezone(pytz.timezone(timezone))  # 设置时区信息
                except ValueError:
                    raise ValueError("無效的日期時間字串格式。必須是 '%Y-%m-%d %H:%M:%S' 或 '%Y-%m-%d'。")
        else:
            raise ValueError("輸入類型無效。必須是日期時間物件或字串。")
         
    
    def handle_date_range(self, start_datetime, end_datetime, frequency, timezone='Asia/Taipei'):        
        # 使用频率生成日期范围，同时规范化到午夜
        if frequency == '1d':
            date_range = pd.date_range(start=start_datetime, end=end_datetime, freq=frequency, tz=timezone, normalize=True)
        else:
            date_range = pd.date_range(start=start_datetime, end=end_datetime, freq=frequency, tz=timezone)
            # 过滤时间，只保留 9:00 到 13:30 的时间
            date_range = date_range[(date_range.time >= time(9, 0)) & (date_range.time <= time(13, 30))]

        return date_range


    def handle_symbols(self, input_symbols):
        symbols = []
        if isinstance(input_symbols, (list, set)):
            # 如果是列表或集合，假設包含有效的股票代碼
            symbols.extend(input_symbols)
        elif isinstance(input_symbols, dict):
            # 如果是字典，提取字典的鍵作為股票代碼列表
            symbols.extend(list(input_symbols.keys()))
        elif isinstance(input_symbols, str):
            # 如果是字串，假設已經是有效的股票代碼，直接返回一個包含該代碼的列表
            symbols.append(input_symbols)
        else:
            raise ValueError("輸入無效。 必須是字串、列表、集合或字典。")
        
        # 返回 symbols
        return symbols
               
    
    def handle_model(self, Symbol, voting_results, min_quantity, trained_models, scaler, median_values, printing=True):
        print(f'voting_results: \n{voting_results}')
        best_row = voting_results[voting_results['篩選模型數量'] >= min_quantity].sort_values(by='Average score', ascending=False).iloc[0]
        print(f'best_row: \n{best_row}')
        best_model_count = int(best_row['篩選模型數量'])
        best_average_score = best_row['Average score']
        model_name = f"{Symbol}_model_{best_model_count}"

        # 从训练模型字典中提取对应的模型
        model = trained_models.get(model_name)

        if model is None:
            print(f"找不到模型：{model_name}")
            return None
        
        print(f"\n{Symbol}的最佳投票器模型共 {best_model_count} 種組合：{model_name}，分數: {best_average_score}")
        
        if printing == True:
            print(f"{model}")

        # 修改的代码以存储所需的信息
        best_model = {
            'Symbol': Symbol,
            'ModelName': model_name,
            'Model': model,
            'Scaler': scaler,
            'MedianValues': median_values,
            'Score': best_average_score
        }

        return best_model