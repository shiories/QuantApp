import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor


class GroupPreprocessor:
    
    def __init__(self, scaler=None, median_values=pd.Series()):
        self.scaler = scaler
        self.median_values = median_values


    def training_data(self, data_set, y=None, predict=False):
        if y is not None:
            data_set["y"] = y
            
        columns_to_drop = [col for col in data_set.columns if 'Unnamed' in col]
        data_set = data_set.drop(columns=columns_to_drop)
        data_set = data_set.dropna(subset=[data_set.columns[-1]]).copy()

        data_set = self.data_clean(data_set)
        data_set = self.group_scaler( data_set, "Stock code", "財報發布日")

        if self.scaler:
            scaler = self.scaler
        else:
            scaler = MinMaxScaler()
            
        cols_to_normalize = data_set.columns[:-1]
        data_set[cols_to_normalize] = scaler.fit_transform(data_set[cols_to_normalize])
            
        data_set = data_set.replace([np.inf, -np.inf], np.nan)
        
        if not self.median_values.empty:
            median_values = self.median_values
        else:
            median_values = data_set.median() 
            
        data_set = data_set.fillna(value=median_values)
        
        data_set.sort_values(by=["財報發布日", "Stock code"], inplace=True)
        
        model_df = data_set.sort_values(by='財報發布日')
        mean = (model_df.dropna().iloc[:,-1].mean())*1.2
        if mean < 0:
            mean = 0
            
        if predict:
            split_index = int(len(model_df) * 0.95)
            model_df = data_set.iloc[:split_index]
            predict_df = data_set.iloc[split_index:]
            X_predict = predict_df.iloc[:, :-1]
            Y_answer = predict_df.iloc[:, -1]
            Y_predict_df = pd.DataFrame(Y_answer)
            Y_predict_df.columns = ['Return']
        else:
            X_predict = pd.DataFrame()
            Y_predict_df = pd.DataFrame()
        
        x = model_df.iloc[:, :-1]
        y = model_df.iloc[:, -1]
        y = (y > mean).astype(int)

        x_train, x_test, y_train, y_test = train_test_split(x.copy(), y.copy(), test_size=0.2, shuffle=False, random_state=42)
                
        return x_train, x_test, y_train, y_test, x, y, scaler, median_values, X_predict, Y_predict_df


    def data_clean(self, df=pd.DataFrame(), col_=0.2, row_=0.2, col_row_col=False, clean_na=True):
        pd.set_option('future.no_silent_downcasting', True)
        print(f'處理前:  {df.shape[1]}列, {len(df)}行 ')
        df = df.replace("--", np.nan).replace({',': ''}, regex=True).infer_objects(copy=False)
        df = df.replace(0, np.nan).infer_objects(copy=False)
        
        for col in df.columns[2:]:
            if df[col].dtype == 'O':
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass
        if clean_na:
            if col_row_col:
                df = df.dropna(thresh=len(df.columns) * (1 - row_))
            df = df.dropna(axis=1, thresh=len(df) * (1 - col_))
            df = df.dropna(thresh=len(df.columns) * (1 - row_))
        df = df.fillna(0)
        print(f'處理後:  {df.shape[1]}列, {len(df)}行 ')
        return df


    def group_scaler(self, data_set, col_1, col_2=None):
        data_set[str(col_1)] = data_set[str(col_1)].astype(str)
        grouped = data_set.groupby(str(col_1))
        processed_data = []
        with ThreadPoolExecutor() as executor:
            for _, group in grouped:
                processed_data.append(executor.submit(self.process_group, group, col_1, col_2))
        processed_data = [future.result() for future in processed_data]
        data_set = pd.concat(processed_data)
        
        return data_set


    def process_group(self, group, col_1, col_2=None):
        group = group.set_index([str(col_1), col_2])
        numeric_columns = group.select_dtypes(include=['int64', 'float64']).columns
        group = group[numeric_columns]  
        group.drop_duplicates(inplace=True)
        
        scaler = MinMaxScaler()
        cols_to_normalize = group.columns[:-1]
        group[cols_to_normalize] = scaler.fit_transform(group[cols_to_normalize])
        zero_ratio = (group == 0).sum() / len(group)
        columns_to_drop = zero_ratio[zero_ratio > 0.5].index
        group.drop(columns_to_drop, axis=1, inplace=True)
        
        group = group.pct_change(fill_method=None)
        group["Return"] = group["Closing Price"].shift(-1)
        group.drop(["Closing Price"], axis=1, inplace=True)
        group = group.iloc[1:-1]

        group = group.replace([np.inf, -np.inf], np.nan)
        median_values = group.median() 
        group = group.fillna(value=median_values)
        
        return group


    def merge_data(self, df_temp=pd.DataFrame(), df=pd.DataFrame(), lest_col=str(), index_drop=False):
        if df.empty:
            df = df_temp.copy()
        else:
            if index_drop:
                df.reset_index(drop=False, inplace=True)
            existing_columns = list(df.columns)
            new_columns = list(df_temp.columns)
            columns_to_add = [col for col in new_columns if col not in existing_columns]

            if not columns_to_add:
                df = pd.concat([df, df_temp], ignore_index=True, sort=False)
            else:
                for col in columns_to_add:
                    df[col] = pd.Series(dtype=df_temp[col].dtype)
                df = pd.concat([df, df_temp], ignore_index=True, sort=False)
                
        if lest_col:
            df[lest_col] = df.pop(lest_col)

        return df
