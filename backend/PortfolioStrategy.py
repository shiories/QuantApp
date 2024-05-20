import asyncio
import pandas as pd
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from backend.StockPicker import *
from backend.OnlineCrawling import *
from backend.GroupPreprocessor import *
from backend.EnsembleEvaluation import *
from backend.ModelEvaluation import *
from backend.TransfomeHandler import *
from backend.PortfolioAllocation import *
from backend.StatsEvaluator import *

class PortfolioStrategy:
    
    def __init__(self, min_year: int = 2015, max_year: int = 2024, min_company_count: int = 15, file_name: str = None, min_quantity: int = 1, total_investment: int = 1000000):       
        self.max_year = max_year
        self.min_year = min_year
        self.min_company_count = min_company_count
        self.file_name = file_name
        self.min_quantity = min_quantity
        self.total_investment =total_investment

    def run(self, total_investment: int =1000000, portfolio: str ='max_sharpe', min_industry_mean: float =0.05, multiplier:int =7):
        closes_df = self.get_closes_df()
        test_results, selected_features_df, voting_results, stats_df, rise_stock_list, latest_date = self.train_model(closes_df, min_industry_mean, multiplier)
        if not rise_stock_list:
            print(f'由於條件過於嚴格，沒有任何股票被選出，請重新進行參數調整。')
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, None
        Initial_principal = self.get_portfolio(total_investment, portfolio, rise_stock_list, latest_date)
        return test_results, selected_features_df, voting_results, stats_df, Initial_principal, latest_date

    def get_closes_df(self):
        
        #篩選合格產別
        online_crawling = OnlineCrawling()
        stock_df , stock_list, industry_df = online_crawling.get_stock(self.min_year, self.min_company_count, excluded_industries=None)
        print(stock_df)
        print(industry_df)
        
        #將財務比率表附上股價
        stock_picker = StockPicker(stock_df, self.file_name)
        closes_df = asyncio.run(stock_picker.get_closes(self.max_year, self.min_year))
        
        return closes_df

    def get_model(self, group_df, industry):
        #處理模型資料
        group_preprocessor = GroupPreprocessor()
        x_train, x_test, y_train, y_test, x, y, scaler, median_values, X, Y_answer = group_preprocessor.training_data(group_df, predict=True)
        #篩選方法與模型配對
        classifiers_evaluation = ModelEvaluation(x_train, x_test, y_train, y_test, x, y, model_type = "Classifier")
        results_df, selectors, classifiers, features_df, var_selectors = asyncio.run(classifiers_evaluation.evaluate_models(industry))
        print(f'組合結果:\n{results_df}')
        #擬合集成模型
        ensemble_evaluation = EnsembleEvaluation(x_train, x_test, y_train, y_test, results_df, x, y, selectors, classifiers, var_selectors, model_type = "Classifier")
        voting_df, trained_models = asyncio.run(ensemble_evaluation.evaluate_models(industry))
        print(f'投票結果:\n{voting_df}')

        #提取最佳模型
        transfome_handler =TransfomeHandler()
        best_model = transfome_handler.handle_model(industry, voting_df, self.min_quantity, trained_models, scaler, median_values, printing=False)
        model = best_model['Model']
        return X, Y_answer, voting_df, results_df, features_df, model

    def get_predictions(self, X, Y_answer, model, min_industry_mean, multiplier):

        print(f'預測模型範圍:{X.index.get_level_values("財報發布日").min()} ~ {X.index.get_level_values("財報發布日").max()}')
        #預測結果
        Y_answer['predictions'] = model.predict(X)

        # 創建 DataEvaluator 實例
        stats_evaluator = StatsEvaluator(Y_answer, "predictions", min_industry_mean, multiplier)
        # 評估資料集並獲取結果和比較統計
        best_label, best_score, stats = stats_evaluator.evaluate_datasets()
        if best_label != "":
            select_stocks = Y_answer.loc[Y_answer['predictions'] == best_label]
            latest_date = select_stocks.index.get_level_values("財報發布日").max()
            three_months_ago = latest_date - timedelta(days=60)
            rise_list = select_stocks.query('@three_months_ago <= `財報發布日` <= @latest_date').index.get_level_values("Stock code").drop_duplicates().tolist()
            print(f'預測股票範圍:{three_months_ago} ~ {latest_date}')
            print(f'預測股票:\n{rise_list}')
        else:
            rise_list=[]
            latest_date = Y_answer.index.get_level_values("財報發布日").max()

        return stats, rise_list, latest_date

    def train_model(self, closes_df, min_industry_mean, multiplier):
        # 將 closes_df 按照 "產業別" 列分組
        grouped_closes_df = closes_df.groupby('產業別')
        
        selected_features_df, test_results, voting_results, stats_df = [pd.DataFrame() for _ in range(4)]

        rise_stock_list = []
        
        group_preprocessor = GroupPreprocessor()

        def process_industry(industry, group_df):

            print(f'開始處理 {industry} 產業')
            if len(group_df) > 1500:
                X, Y_answer, voting_df, results_df, features_df, model = self.get_model(group_df, industry)
                stats, rise_list, latest_date = self.get_predictions(X, Y_answer, model, min_industry_mean, multiplier)
                
                # 將 test_results 和 selected_features_df 分別添加到 merged_results 中
                for df in [results_df, features_df, voting_df, stats]:
                    df['產業別'] = industry
    
                return results_df, features_df, voting_df, stats, rise_list, latest_date
            else:
                print(f'{industry}的資料數量為{len(group_df)}, 少於1000不進行訓練。')
                

            
        # 使用 ThreadPoolExecutor 並行處理每個行業
        with ThreadPoolExecutor() as executor:
            # 將處理每個行業的任務提交給 ThreadPoolExecutor
            futures = [executor.submit(process_industry, industry, group_df) for industry, group_df in grouped_closes_df]
            
            # 等待所有任務完成
            for future in futures:
                # 獲取任務的返回結果
                result = future.result()
                
                # 解包結果
                if result: 
                    results_df, features_df, voting_df, stats, rise_list, latest_date = result
                    # 將結果合併到主函數的變數中
                    test_results = group_preprocessor.merge_data(test_results, results_df, lest_col='Average score', index_drop=True)
                    selected_features_df = group_preprocessor.merge_data(selected_features_df, features_df, lest_col='Average score', index_drop=True)
                    voting_results = group_preprocessor.merge_data(voting_results, voting_df, lest_col='Average score', index_drop=True)
                    stats_df = group_preprocessor.merge_data(stats_df, stats, index_drop=True)
                    rise_stock_list.extend(rise_list)
                    latest_date = latest_date

        dfs = {"test_results": test_results, "selected_features_df": selected_features_df, "voting_results": voting_results, "stats_df": stats_df}
        
        for name, df in dfs.items():
            df.reset_index(drop=True, inplace=True)
            df.set_index('產業別', inplace=True)
            df.index.name = '產業別'
            print(f"{name}:\n{df}")

        print(f"rise_stock_list:\n{rise_stock_list}")

        return test_results, selected_features_df, voting_results, stats_df, rise_stock_list, latest_date

    def get_portfolio(self, total_investment, portfolio, rise_stock_list, latest_date):
        # 投資組合
        portfolio_instance = PortfolioAllocation(total_investment, rise_stock_list, latest_date, year=5)

        if portfolio == "max_sharpe":
            _Initial_principal = portfolio_instance.max_sharpe_portfolio()
        elif portfolio == "min_variance":
            _Initial_principal = portfolio_instance.min_variance_portfolio()
        elif portfolio == "mvo_portfolio":
            _Initial_principal = portfolio_instance.mvo_portfolio()
        elif portfolio == "equal_weights":
            _Initial_principal = portfolio_instance.equal_weights_portfolio()

        print(f"{portfolio}組合本金: \n {_Initial_principal}")

        return _Initial_principal



if __name__=="__main__":
    
    file_name = "特徵6"
    min_year = 2016 #開始年分
    max_year = 2023 #結束年分
    min_company_count = 50 #產業別最少公司數
    min_quantity = 3  #最小模型數量
    min_industry_mean = 0.05 #各產業別要求最低報酬
    multiplier = 7 # 評分倍數(對標準差容忍度)
    total_investment = 1000000 #總本金
    portfolio = "mvo_portfolio"  #投資組合方法 max_sharpe、min_variance、mvo_portfolio、equal_weights

    portfolio_strategy = PortfolioStrategy(min_year, max_year, min_company_count, file_name, min_quantity)
    test_results, selected_features_df, voting_results, stats_df, Initial_principal, latest_date = portfolio_strategy.run(total_investment, portfolio, min_industry_mean, multiplier)

    for df_name in ["test_results", "selected_features_df", "voting_results", "stats_df"]:
        df = globals().get(df_name)
        if df is not None:
            print(f'{df_name}:')
            print(df)


