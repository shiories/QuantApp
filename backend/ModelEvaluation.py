import asyncio
import pandas as pd
import numpy as np
import time
from sklearn.metrics import (mean_squared_error, mean_absolute_error, max_error, r2_score, explained_variance_score, median_absolute_error)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, f_regression, GenericUnivariateSelect, RFE,
                                       mutual_info_regression, chi2, mutual_info_classif, SelectFromModel)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier)
from sklearn.linear_model import (LinearRegression, Lasso, ElasticNet, LogisticRegression)
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score)

from backend.SelectTopFeatures import *


class ModelEvaluation:
    
    def __init__(self, X_train, X_test, y_train, y_test, X, y, model_type : str="Regressor" or "Classifier"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X = X
        self.y = y
        self.var_selectors = {}
        self.model_type = model_type
        self.selectors = self.selectors_type()
        self.models = self.models_type()


    def selectors_type(self):
        
        total_features = self.X_train.shape[1]
        selected_n = int(total_features * 0.2)
        threshold_ratio = 0.8
        threshold = threshold_ratio * np.var(self.X_train, axis=0).mean()
        
        if self.model_type == "Regressor":
            self.selectors = {
                'None': None,
                'VarianceThreshold': VarianceThreshold(threshold=threshold),
                'SelectKBest': SelectKBest(f_regression, k=selected_n),
                'GenericUnivariateSelect': GenericUnivariateSelect(score_func=f_regression, mode='k_best', param=selected_n),
                'F_regression': SelectKBest(f_regression, k=selected_n),
                'SelectFromModel': SelectFromModel(estimator=RandomForestRegressor(n_estimators=50, random_state=42), max_features=selected_n),
                'RFE': RFE(estimator=LinearRegression(), n_features_to_select=selected_n),
            }
            
        elif self.model_type == "Classifier":
            self.selectors = {
                'None': None,
                'VarianceThreshold': VarianceThreshold(threshold=threshold),
                'SelectKBest': SelectKBest(f_regression, k=selected_n),
                'GenericUnivariateSelect': GenericUnivariateSelect(score_func=f_regression, mode='k_best', param=selected_n),
                'RFE': RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=selected_n),
                'MutualInfoRegression': SelectKBest(mutual_info_regression, k=selected_n),
                'Chi2': SelectKBest(chi2, k=selected_n),
                'MutualInfoClassif': SelectKBest(mutual_info_classif, k=selected_n),
                'SelectTop_P': SelectTopFeatures(positive_K=10),
                'SelectTop_N': SelectTopFeatures(negative_K=10),
                'SelectTop_PN': SelectTopFeatures(positive_K=10, negative_K=10)
            }
        
        return self.selectors


    def models_type(self):
                
        if self.model_type == "Regressor":
            self.models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(alpha=0.005, max_iter=1000, tol=0.01, selection='cyclic', random_state=42),
                'ElasticNet': ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=1000, tol=0.01, selection='cyclic', random_state=42),
                'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
                'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
                'AdaBoostRegressor': AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42),
                'BaggingRegressor': BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=50, random_state=42),
                'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(kernel='rbf'),
                'NuSVR': NuSVR(kernel='rbf'),
                'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5),
                'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=2, random_state=42),
            }
            
        elif self.model_type == "Classifier":
            self.models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
                'LogisticRegression': LogisticRegression(solver='liblinear', max_iter=5000, penalty='l1', C=1, random_state=42),
                'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=5000, activation="relu", solver='adam', random_state=42),
                'GaussianNB': GaussianNB(),
                'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
                'AdaBoost': AdaBoostClassifier(algorithm='SAMME', n_estimators=50, random_state=42),
                'ExtraTrees': ExtraTreesClassifier(n_estimators=50, max_depth=5, random_state=42),
                'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=50, random_state=42),
                'CalibratedClassifier': CalibratedClassifierCV(estimator=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1), cv=3)
            }
        
        return self.models


    async def evaluate_models(self, Symbol):
        start_time = time.time()
        self.test_results = pd.DataFrame()
        self.selected_features_df = pd.DataFrame()

        for var_selector_name, var_selector_option in self.selectors.items():
            if var_selector_name in self.var_selectors:
                var_selector = self.var_selectors[var_selector_name]
            else:
                var_selector, features_df = await self.fit_var_selector(var_selector_name, var_selector_option)
                self.var_selectors[var_selector_name] = var_selector
                self.selected_features_df = pd.concat([self.selected_features_df, features_df], ignore_index=True)

            tasks = [self.run_models(var_selector_name, var_selector, models_name, models_option, Symbol)
                     for models_name, models_option in self.models.items()]

            results = await asyncio.gather(*tasks)
            for df_result in results:
                self.test_results = pd.concat([self.test_results, df_result], ignore_index=True)

        self.average_scoring()

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f'---------------------{Symbol}模型組合結束，總耗時: {minutes} 分{seconds} 秒------ ---------------')

        return self.test_results, self.selectors, self.models, self.selected_features_df, self.var_selectors


    async def run_models(self, var_selector_name, var_selector, models_name, models, Symbol):
        start_time = time.time()

        X_train_copy = self.X_train.copy()
        X_test_copy = self.X_test.copy()
        y_train_copy = self.y_train.copy()
        y_test_copy = self.y_test.copy()

        model = make_pipeline(
            FunctionTransformer(lambda x: x) if var_selector is None else var_selector,
            models
        )

        model.steps[-1][1].feature_names = X_train_copy.columns

        model.fit(X_train_copy, y_train_copy)

        result_df = self.model_scoring( model, X_test_copy, y_test_copy, var_selector_name, models_name)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{Symbol}模型耗時: {elapsed_time:.2f} 秒, 組合: {var_selector_name} - {models_name}")

        return result_df


    async def fit_var_selector(self, var_selector_name, var_selector_option):
        if var_selector_option is not None:
            X_train_copy = self.X_train.copy()
            y_train_copy = self.y_train.copy()
            var_selector = var_selector_option.fit(X_train_copy, y_train_copy)
        else:
            var_selector = None

        selected_features = np.full(self.X.shape[1], 'O', dtype=object)
        if var_selector is not None:
            selected_features[var_selector.get_support(indices=True)] = 'X'

        features_df = pd.DataFrame(columns=['篩選方法'] + [f'{feature}' for feature in self.X_train])
        features_df = pd.DataFrame([([var_selector_name] + list(selected_features))], columns=features_df.columns)

        return var_selector, features_df


    def model_scoring(self, model, X_test_copy, y_test_copy, var_selector_name, models_name):
        y_pred = model.predict(X_test_copy)
        if self.model_type == "Regressor":
            mse = mean_squared_error(y_test_copy, y_pred)
            mae = mean_absolute_error(y_test_copy, y_pred)
            max_err = max_error(y_test_copy, y_pred)
            r_squared = r2_score(y_test_copy, y_pred)
            explained_var = explained_variance_score(y_test_copy, y_pred)
            median_abs_err = median_absolute_error(y_test_copy, y_pred)
                    
            result = {
                '篩選方法': var_selector_name,
                '模型種類': models_name,
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
                '篩選方法': var_selector_name,
                '模型種類': models_name,
                '準確率': accuracy,
                '精確率': precision,
                '召回率': recall,
                'ROC-AUC': roc_auc,
                'F1-score': f1,
                'Kappa': kappa_normalized
            }
        
        return pd.DataFrame([result])


    def average_scoring(self):
        
        if self.model_type == "Regressor":
            # 將0到正無窮的分數類型轉換為0到1的倒數，並將超過3的值設置為0
            inverse_columns = ['均方誤差', '平均絕對誤差', '最大誤差', '中位數絕對誤差']
            test_results_inverse = self.test_results.copy()
            test_results_inverse[inverse_columns] = 1 - test_results_inverse[inverse_columns]
            test_results_inverse[inverse_columns] = np.where(test_results_inverse[inverse_columns] < 0, 0, test_results_inverse[inverse_columns])

            # 計算平均值
            numeric_columns = test_results_inverse.select_dtypes(include=np.number)
            average_score = numeric_columns.mean(axis=1)

            # 將平均標準化分數加入結果表格
            self.test_results['Average score'] = average_score
            # 根據平均標準化分數降序排列
            self.test_results = self.test_results.sort_values(by='Average score', ascending=False)
            
        elif self.model_type == "Classifier":
            self.test_results['Average score'] = (self.test_results['F1-score'] + self.test_results['ROC-AUC'] + self.test_results['Kappa']) / 3
            self.test_results = self.test_results.sort_values(by=['Average score', 'F1-score', 'ROC-AUC', 'Kappa'],
                                                        ascending=[False, False, False, False])

        self.selected_features_df["特徵數"] = (self.selected_features_df == "O").sum(axis=1)
        average_scores = self.test_results.groupby("篩選方法")["Average score"].mean()
        self.selected_features_df["Average score"] = self.selected_features_df["篩選方法"].map(average_scores)
        self.selected_features_df = self.selected_features_df.sort_values(by=['Average score'], ascending=[False])


    
