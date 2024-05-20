import numpy as np
import pandas as pd

class SelectTopFeatures:
    
    def __init__(self, positive_K=0, negative_K=0):
        self.positive_K = positive_K
        self.negative_K = negative_K
        self.selected_features = None


    def fit(self, X, y):
        if self.positive_K == 0 and self.negative_K == 0:
            raise ValueError("Positive_K 和 Negative_K 應該至少其中一個不為0")
        
        correlations = np.abs(X.corrwith(y))
        sorted_indices = correlations.sort_values(ascending=False).index

        if self.positive_K != 0 and self.negative_K != 0:
            selected_indices = sorted_indices[:self.positive_K].append(sorted_indices[-self.negative_K:])
            selected_features = selected_indices
        elif self.positive_K != 0:
            selected_features = sorted_indices[:self.positive_K]
        elif self.negative_K != 0:
            selected_features = sorted_indices[-self.negative_K:]
            
        self.selected_features = X.columns.isin(selected_features.tolist())

        return self


    def transform(self, X):
        self.transformed_X = X.iloc[:, self.selected_features] if isinstance(X, pd.DataFrame) else X[:, self.selected_features]
        return self.transformed_X

    
    def get_support(self, indices=True):
        if indices:
            return np.where(~self.selected_features)[0]
        else:
            return ~self.selected_features

    