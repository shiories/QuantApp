import numpy as np
from scipy.optimize import minimize
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
        # 將股票代碼加上 '.TW' 並下載資料
        start_date = latest_date - timedelta(days=365*year)  # 往前五年
        end_date = latest_date
        # 下載資料
        data = yf.download(" ".join([stock + '.TW' for stock in self.rise_stock_list]), start=start_date, end=end_date, progress=False )
        # 只保留第一級索引和 'Close' 列
        self.data = data['Close']
        print(self.data)


    def _calculate_portfolio_variance(self, weights, cov_matrix):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_variance


    def _get_optimal_weights(self, mu, S, method='max_sharpe'):
        if method == 'max_sharpe':
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
        elif method == 'min_variance':
            ef = EfficientFrontier(mu, S)
            weights = ef.min_volatility()
        elif method == 'mvo':
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
        elif method == 'equal_weights':
            num_assets = len(self.data.columns)
            weights = np.array([1/num_assets] * num_assets)
        else:
            raise ValueError("方法無效。 請從max_sharpe、min_variance、mvo、equal_weights 中選擇。")
        return weights


    def _clean_weights(self, weights):
        return {asset: round(weight * self.total_investment) for asset, weight in weights.items() if weight > 0}


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
    rise_stock_list = ['3622', '4942', '5234', '2489', '3024', '3031', '3059', '3356', '4935', '6226',
                       '6405', '2374', '2406', '3051', '3481', '3576', '6168', '9933', '9905', '1516',
                       '8033', '9941', '1416', '1437', '3040', '5871', '9927', '2496', '9917', '9942',
                       '9944', '9945', '2348', '2514', '2904', '9919', '3030', '8021', '2488', '6201',
                       '2482', '2423', '2464', '6196', '2360', '2459', '2474', '2477', '3518', '3617',
                       '5225', '6192', '6215', '6283', '2354', '2359', '2404', '2461', '3043', '3305',
                       '3665', '6139', '2312', '2317', '2373', '2433', '3018', '8201', '2441', '3006',
                       '2451', '3014', '2408', '2449', '2351', '5471', '2401', '2548', '2506', '2597',
                       '1805', '2516', '2528', '6177', '1456', '2536', '2539', '3056', '5521', '1316',
                       '1808', '2504', '2505', '2511', '2520', '2530', '2545', '2546', '2547', '3052',
                       '3703', '5531', '5533', '1438', '1439', '2509', '2515', '2524', '2527', '2534',
                       '2537', '2538', '2540', '2923', '9906', '9946', '2442', '2501', '2535', '2542',
                       '2543', '5522', '5534', '1477', '1476', '1413', '1434', '1451', '1459', '1445',
                       '1454', '1417', '1440', '1414', '1464', '1473', '1465', '1466', '1468', '1409',
                       '1410', '1418', '1446', '1449', '1455', '1457', '1460', '1467', '1470', '1474',
                       '1475', '1402', '1419', '1444', '1447', '2439', '2455', '3380', '3311', '3596',
                       '2450', '3047', '3704', '4904', '2498', '6142', '4906', '2345', '2412', '5388',
                       '8011', '2321', '2485', '3025', '3027', '3669', '6152', '2314', '2332', '2424',
                       '3045', '3694', '8101', '2809', '2832', '2812', '2816', '2855', '2820', '2849',
                       '2852', '2836', '2838', '2850', '2851', '2884', '2801', '2834', '2845', '2867',
                       '6005', '2883', '2885', '2889', '2881', '2882', '2887', '2890', '2891', '5880',
                       '2880', '2886', '2888', '2892', '2009', '5538', '3023', '2308', '6224', '3501',
                       '3003', '3229', '3058', '1560', '8374', '1535', '1515', '1513', '1526', '1538',
                       '1539', '1517', '1519', '1529', '1540', '1589', '2049', '3167', '4532', '1506',
                       '1527', '1530', '1532', '1590', '1504', '1528', '1583', '2371', '4526', '2301',
                       '2395', '2397', '2353', '2405', '3046', '3515', '6206', '2376', '3022', '2305',
                       '3013', '3060', '8210', '2352', '2465', '2495', '3231', '4916', '2331', '2356',
                       '2362', '2365', '2377', '2382', '2387', '3002', '3057', '6117', '6166', '6230',
                       '6277', '2357', '2380', '2417', '2425', '3494', '3706', '6128', '6235', '8114',
                       '9912', '2324', '2364', '2399', '3005', '3701', '4938', '5215']
    
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

