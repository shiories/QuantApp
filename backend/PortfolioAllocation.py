import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import yfinance as yf
from datetime import datetime, timedelta

class PortfolioAllocation:
    
    def __init__(self, total_investment, rise_stock_list, latest_date, year):
        self.rise_stock_list = rise_stock_list
        self.total_investment = total_investment
        self.get_data(latest_date, year)

    def get_data(self, latest_date, year):
        # 将股票代码加上 '.TW' 并下载数据
        start_date = latest_date - timedelta(days=365*year)  # 往前几年
        end_date = latest_date
        # 下载数据
        data = yf.download(" ".join([stock + '.TW' for stock in self.rise_stock_list]), start=start_date, end=end_date, progress=False)
        # 只保留第一层索引和 'Adj Close' 列
        self.data = data['Adj Close']
        print(self.data)

    def _calculate_portfolio_variance(self, weights, cov_matrix):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_variance

    def _get_optimal_weights(self, mu, S, method='max_sharpe'):
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))  # 添加非负约束
        if method == 'max_sharpe':
            weights = ef.max_sharpe()
        elif method == 'min_variance':
            weights = ef.min_volatility()
        elif method == 'mvo':
            weights = ef.max_sharpe()
        elif method == 'equal_weights':
            num_assets = len(self.data.columns)
            weights = np.array([1/num_assets] * num_assets)
            return {self.data.columns[i]: weights[i] for i in range(num_assets) if weights[i] > 0}
        else:
            raise ValueError("方法无效。 请从max_sharpe、min_variance、mvo、equal_weights 中选择。")
        return {asset: weight for asset, weight in weights.items() if weight > 0}

    def _clean_weights(self, weights):
        # 四舍五入到1%的精度并剔除小于1%的权重
        rounded_weights = {asset: round(weight, 2) for asset, weight in weights.items() if weight >= 0.01}
        # 重新平衡权重，使其总和为1
        total_weight = sum(rounded_weights.values())
        balanced_weights = {asset: weight / total_weight for asset, weight in rounded_weights.items()}
        # 按照总投资金额分配资金
        investment_allocation = {asset: round(weight * self.total_investment) for asset, weight in balanced_weights.items()}
        return investment_allocation

    def max_sharpe_portfolio(self):
        mu = expected_returns.mean_historical_return(self.data)
        S = risk_models.sample_cov(self.data)
        weights = self._get_optimal_weights(mu, S, method='max_sharpe')
        cleaned_weights = self._clean_weights(weights)
        return cleaned_weights

    def min_variance_portfolio(self):
        mu = expected_returns.mean_historical_return(self.data)
        S = risk_models.sample_cov(self.data)
        weights = self._get_optimal_weights(mu, S, method='min_variance')
        cleaned_weights = self._clean_weights(weights)
        return cleaned_weights

    def mvo_portfolio(self):
        mu = expected_returns.mean_historical_return(self.data)
        S = risk_models.sample_cov(self.data)
        weights = self._get_optimal_weights(mu, S, method='mvo')
        cleaned_weights = self._clean_weights(weights)
        return cleaned_weights

    def equal_weights_portfolio(self):
        num_assets = len(self.data.columns)
        weights_equal = np.array([1/num_assets] * num_assets)
        weights_dict = {asset: weight for asset, weight in zip(self.data.columns, weights_equal)}
        return self._clean_weights(weights_dict)




if __name__=="__main__":
    
    total_investment = 1000000 #總本金
    rise_stock_list = ['3622', '4942', '5234', '2489', '3024', '3031', '3059', '3356', '4935',
                       '6405', '2374', '2406', '3051', '3481', '3576', '6168', '9933', '9905', '1516',
                       '8033', '9941', '1416', '1437', '3040', '5871', '9927', '2496', '9917', '9942',
                       '9944', '9945', '2348', '2514', '2904', '9919', '3030', '8021', '2488', '6201',
                       '2482', '2423', '2464', '6196', '2360', '2459', '2474', '2477', '3518', '3617',
                       '5225', '6192', '6215', '6283', '2354', '2359', '2404', '2461', '3043', '3305',
                       '3665', '6139', '2312', '2317', '2373', '2433', '3018', '8201', '2441', '3006',
                       '2451', '3014', '2408', '2449', '2351', '5471', '2401', '2548', '2506', '6226',
                       '1805', '2516', '2528', '6177', '1456', '2536', '2539', '3056', '5521', '1316',
                       '1808', '2504', '2505', '2511', '2520', '2530', '2545', '2546', '2547', '3052',
                       '3703', '5531', '5533', '1438', '1439', '2509', '2515', '2524', '2527', '2534',
                       '2537', '2538', '2540', '2923', '9906', '9946', '2442', '2501', '2535', '2542',
                       '2543', '5522', '5534', '1477', '1476', '1413', '1434', '1451', '1459', '1445',
                       '1454', '1417', '1440', '1414', '1464', '1473', '1465', '1466', '1468', '1409',]
    
    portfolio = "mvo_portfolio"
    latest_date = datetime(2023, 11, 1)
    portfolio_instance = PortfolioAllocation(total_investment, rise_stock_list, latest_date, year=5)

    if portfolio == "max_sharpe":
            Initial_principal = portfolio_instance.max_sharpe_portfolio()
    elif portfolio == "min_variance":
            Initial_principal = portfolio_instance.min_variance_portfolio()
    elif portfolio == "mvo_portfolio":
            Initial_principal = portfolio_instance.mvo_portfolio()
    elif portfolio == "equal_weights":
            Initial_principal = portfolio_instance.equal_weights_portfolio()
            
    print(f'{portfolio}組合本金: \n {Initial_principal}')
    

