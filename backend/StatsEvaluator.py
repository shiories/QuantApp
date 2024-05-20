import numpy as np
import pandas as pd
from scipy.stats import zscore

class StatsEvaluator:
    
    def __init__(self, data, label_column, min_industry_mean:float =0, multiplier:int =7):
        self.data = data
        self.label_column = label_column
        self.stats_comparison = None
        self.min_industry_mean = min_industry_mean
        self.multiplier = multiplier
    def evaluate_datasets(self):
        grouped_data = self.data.groupby(self.label_column)
        best_datasets = {}

        for label, group in grouped_data:
            dataset = group.drop(columns=[self.label_column])
            mean = dataset.mean().mean()
            std = dataset.std().mean()

            # 調整 mean_z_score
            if mean > self.min_industry_mean:
                score = mean / std
                best_datasets[label] = score
                print(f"標籤: {label}, 得分: {round(score, 4)} (mean={round(mean,4)} std={round(std,4)})")
            else:
                best_datasets[label] = 0
                print(f"標籤: {label}, 得分: 平均值低於最低標準{self.min_industry_mean}，評為 0 分)")


        best_label = max(best_datasets, key=best_datasets.get)            
        best_score = best_datasets[best_label]
        
        if best_score > self.min_industry_mean * self.multiplier:
            print(f"最優類別為 {best_label} ，得分: {round(best_score, 4)}")
        else:
            best_label=""
            best_score=0
            print(f"所有類別分數均低於最低標準{self.min_industry_mean}的{self.multiplier}倍，沒有找到結果")

        stats_comparison = self._generate_comparison_stats()
        print(f"統計比較:\n{stats_comparison}")
        
        return best_label, best_score, stats_comparison



    def _generate_comparison_stats(self):
        # 按照標籤分組資料
        grouped_data = self.data.groupby(self.label_column)
        
        # 創建一個空的DataFrame來存儲統計比較
        stats_comparison = pd.DataFrame()
        
        # 獲取原始資料的統計描述（排除標籤列）
        overall_stats = self.data.drop(columns=[self.label_column]).describe()
        overall_stats_df = pd.DataFrame(overall_stats)
        overall_stats_df.columns = ["100%"]
        stats_comparison = pd.concat([overall_stats_df, stats_comparison], axis=1)
        
        # 逐一處理每個分類的資料
        for label, group in grouped_data:
            # 獲取該分類下的統計描述
            stats = group.drop(columns=[self.label_column]).describe()
            stats_df = stats.copy()
            stats_df.columns=[str(label)]

            # 添加至統計比較DataFrame，使用該標籤作為索引
            stats_comparison = pd.concat([stats_comparison, stats_df], axis=1)


            # 計算標籤統計值相對於原始資料統計值的百分比
            stats = stats / overall_stats
            stats.columns=[str(label) + '%']

            stats_comparison = pd.concat([stats_comparison, stats], axis=1)

        return stats_comparison



if __name__=="__main__":
    # 示例數據
    np.random.seed()
    data = pd.DataFrame({
        "Return1": np.random.normal(loc=0.05, scale=0.05, size=1000),
        "Label": np.random.choice([0, 1, 2, 3, 4], size=1000)
    })


    min_industry_mean = 0.05
    # 創建 DataEvaluator 實例
    stats_evaluator = StatsEvaluator(data, "Label", min_industry_mean)

    # 評估資料集並獲取結果和比較統計
    best_label, best_score, stats_comparison = stats_evaluator.evaluate_datasets()

