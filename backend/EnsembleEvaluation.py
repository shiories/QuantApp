
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, 
                             mean_squared_error, mean_absolute_error, max_error, r2_score, explained_variance_score, median_absolute_error)
from sklearn.preprocessing import FunctionTransformer
import time
import numpy as np
import warnings
import asyncio
import pandas as pd

class EnsembleEvaluation:
    
    def __init__(self, X_train, X_test, y_train, y_test, test_results, X, y, selectors, models, var_selectors, model_type : str="Regressor" or "Classifier"):
        self.model_type=model_type
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.test_results = test_results
        self.selectors = selectors
        self.models = models
        self.X = X
        self.y = y
        self.var_selectors = var_selectors
        self.trained_models = {}
        self.voting_results = pd.DataFrame()
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def average_scoring(self):
        if self.model_type == "Regressor":
            # 將0到正無窮的分數類型轉換為0到1的倒數，並將超過3的值設置為0
            inverse_columns = ['均方誤差', '平均絕對誤差', '最大誤差', '中位數絕對誤差']
            voting_results_inverse = self.voting_results.copy()
            voting_results_inverse[inverse_columns] = 1 - voting_results_inverse[inverse_columns]
            voting_results_inverse[inverse_columns] = np.where(voting_results_inverse[inverse_columns] < 0, 0, voting_results_inverse[inverse_columns])
            # 計算平均值
            numeric_columns = voting_results_inverse.drop(columns=['篩選模型數量']).select_dtypes(include=np.number)
            average_score = numeric_columns.mean(axis=1)
            # 將平均標準化分數加入結果表格
            self.voting_results['Average score'] = average_score
            # 根據平均標準化分數降冪排列
            self.voting_results = self.voting_results.sort_values(by='Average score', ascending=False)
            
        elif self.model_type == "Classifier":
            self.voting_results['Average score'] = (self.voting_results['F1-score'] + self.voting_results['ROC-AUC'] + self.voting_results['Kappa']) / 3
            self.voting_results = self.voting_results.sort_values(by=['Average score', 'F1-score', 'ROC-AUC', 'Kappa'],
                                                        ascending=[False, False, False, False])

    async def evaluate_models(self, Symbol):
        self.Symbol = Symbol
        tasks = [self.evaluate_model(i) for i in range(1, 9)] # int(0.07 * len(self.test_results)) + 1
        results = await asyncio.gather(*tasks)
        
        for result in results:
            i, result_df, model = result
            self.voting_results = pd.concat([self.voting_results, result_df], ignore_index=True)
            self.trained_models[f"{self.Symbol}_model_{i}"] = model
            
        self.average_scoring()
        

        print(f'---------------------------------------------{Symbol}投票模型結束---------------------------------------------')

        return self.voting_results, self.trained_models   

    def fit_var_selector(self, var_selector_name, var_selector_option):
        if var_selector_option is not None:
            X_train_copy = self.X_train.copy()
            y_train_copy = self.y_train.copy()
            if self.var_selectors[var_selector_name] is None:
                var_selector = var_selector_option.fit(X_train_copy, y_train_copy)
                self.var_selectors[var_selector_name] = var_selector
            else:
                var_selector = self.var_selectors[var_selector_name]
        else:
            var_selector = None

        return var_selector

    async def evaluate_model(self, i):
        
        start_time = time.time()
        
        X_train_copy = self.X_train.copy()
        X_test_copy = self.X_test.copy()
        y_train_copy = self.y_train.copy()
        y_test_copy = self.y_test.copy()
        
        selected_models = self.test_results[['篩選方法', '模型種類']].head(i)
        ensemble_models = []
        voting_selector = []

        for index, row in selected_models.iterrows():
            var_selector_option = self.selectors[row['篩選方法']]
            models_option = self.models[row['模型種類']]
            ensemble_models.append({'篩選方法': row['篩選方法'], '模型種類': row['模型種類'], '模型實例': models_option, '篩選實例': var_selector_option})

            var_selector_name = f"{row['篩選方法']}_{row['模型種類']}"
            var_selector = self.fit_var_selector(row['篩選方法'], var_selector_option) if var_selector_option is not None else FunctionTransformer(lambda x: x)
            voting_selector.append((var_selector_name, var_selector))

        feature_union = FeatureUnion(voting_selector)
        
        if self.model_type == "Regressor":
            voting_model = VotingRegressor(estimators=[(f"{model['篩選方法']}_{model['模型種類']}", model['模型實例']) for model in ensemble_models])
        elif self.model_type == "Classifier":
            voting_model = VotingClassifier(estimators=[(f"{model['篩選方法']}_{model['模型種類']}", model['模型實例']) for model in ensemble_models])

        model = make_pipeline(feature_union, voting_model)
        model.fit(X_train_copy, y_train_copy)
        
        result_df = self.model_scoring( i, model, X_test_copy, y_test_copy)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{self.Symbol}集成模型{i}耗時: {elapsed_time:.2f} 秒")
        
        return i, result_df, model

    def model_scoring(self, i, model, X_test_copy, y_test_copy):
        y_pred = model.predict(X_test_copy)
        if self.model_type == "Regressor":
            mse = mean_squared_error(y_test_copy, y_pred)
            mae = mean_absolute_error(y_test_copy, y_pred)
            max_err = max_error(y_test_copy, y_pred)
            r_squared = r2_score(y_test_copy, y_pred)
            explained_var = explained_variance_score(y_test_copy, y_pred)
            median_abs_err = median_absolute_error(y_test_copy, y_pred)
            
            result = {
                '篩選模型數量': i,
                '均方誤差': mse,
                '平均絕對誤差': mae,
                '最大誤差': max_err,
                'R平方': r_squared,
                '解釋變異比例': explained_var,
                '中位數絕對誤差': median_abs_err,
            }
            
        elif self.model_type == "Classifier":
            accuracy = accuracy_score(y_test_copy, y_pred)
            recall = recall_score(y_test_copy, y_pred)
            precision = precision_score(y_test_copy, y_pred, zero_division=1)
            f1 = f1_score(y_test_copy, y_pred)
            roc_auc = roc_auc_score(y_test_copy, y_pred)
            kappa = cohen_kappa_score(y_test_copy, y_pred)
            kappa_normalized = (kappa + 1) / 2

            result = {
                '篩選模型數量': i,
                '準確率': accuracy,
                '精確率': precision,
                '召回率': recall,
                'ROC-AUC': roc_auc,
                'F1-score': f1,
                'Kappa': kappa_normalized
            }
        
        return pd.DataFrame([result])

