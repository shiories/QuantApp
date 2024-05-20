import pandas as pd
import numpy as np
import asyncio
from backend.OnlineCrawling import *
from backend.TransfomeHandler import *

class TradingAnalyzer:

    def __init__(self, trading_system, initial_cash_dict, start_datetime, end_datetime):
        #初始化 TransfomeHandler
        self.transfome_handler = TransfomeHandler()
       # 時間處理
        start_datetime = self.transfome_handler.handle_datetime(start_datetime)
        end_datetime = self.transfome_handler.handle_datetime(end_datetime, is_endtime=True)
        self.trading_days = (end_datetime - start_datetime).days + 1

        self.trading_system = trading_system
        self.initial_cash_dict = initial_cash_dict
        self.total_initial_cash = sum(self.initial_cash_dict.values())
        self.get_rate = OnlineCrawling()
        self.risk_free_rate = self.get_rate.get_rate()
        if self.risk_free_rate is not None:
            self.risk_free_rate /= 100
        else:
            print("Error: 無風險利率為 None，請檢查網站資料是否正常。")

        
    async def update_tables(self):
        # 提取相關資訊
        self.trade_df, self.holdings_df, self.cost_df, self.portfolio_df = self.trading_system.get_tables_df(
            trade=True, holdings=True, cost=True, portfolio=True
        )

        # 進行打印或其他操作
        print("\n交易明細:\n", self.trade_df)
        print("\n持倉明細:\n", self.holdings_df)
        print("\n庫存均價:\n", self.cost_df)
        print("\n資金水位:\n", self.portfolio_df)
        
        
    async def calculate_win_ratio(self, symbol_trades, symbol, principal):
        return (symbol_trades['returns'] > 0).sum() / len(symbol_trades)


    async def calculate_returns_rate(self, symbol_trades, symbol, principal):
        returns = symbol_trades['returns'].sum()
        
        return returns / principal


    async def calculate_max_drawdown(self, symbol_trades, symbol, principal):
        cumulative_returns = symbol_trades['returns'].cumsum()
        max_drawdown = (cumulative_returns - cumulative_returns.expanding().max()) / principal
        max_drawdown = max_drawdown.min()

        return max_drawdown


    async def calculate_cumulative_returns(self, symbol_trades, symbol, principal):
        cumulative_returns = symbol_trades['returns'].cumsum().iloc[-1]
        return (cumulative_returns + principal) / principal


    async def calculate_sortino_ratio(self, symbol_trades, symbol, principal):
        returns = symbol_trades['returns'] / principal
        downside_returns = np.where(returns < 0, returns, 0)
        if len(downside_returns) == 0 or np.isnan(downside_returns).all():
            return np.nan
        downside_deviation = np.std(downside_returns, ddof=1)
        if downside_deviation == 0:
            return np.nan
        sortino_ratio = (returns.mean() / downside_deviation)
        return sortino_ratio


    async def calculate_calmar_ratio(self, symbol_trades, annualized_return, symbol, principal):
        max_drawdown = await self.calculate_max_drawdown(symbol_trades, symbol, principal)
        if np.isnan(max_drawdown) or max_drawdown == 0:
            return np.nan
        calmar_ratio = annualized_return / max_drawdown
        return calmar_ratio


    async def calculate_annualized_return(self, symbol_trades, returns_rate, symbol, principal):
        annualized_return = (1 + returns_rate) ** (252 / self.trading_days) - 1
        return annualized_return


    async def calculate_sharpe_ratio(self, symbol_trades, symbol, principal):
        df = symbol_trades['returns'] /  principal
        excess_returns = df.mean() - (self.risk_free_rate * (self.trading_days / 365) )
        volatility = df.std()
        sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
        return sharpe_ratio
    
    
    async def analyze_statistics(self):
        await self.update_tables()
        if  self.trade_df.empty:
            return pd.DataFrame()
        
        statistics = {
            "統計時間": [],
            "股票代碼": [],
            "交易勝率": [],
            "報酬率": [],
            "累計報酬": [],
            "最大回撤": [],
            "夏普比率": [],
            #"Sortino比率": [],
            #"Calmar比率": [],
            "年化報酬率": [],
            "價格走勢": []
        }

        # 遍歷每個股票
        self.trade_df = self.trade_df[["symbol","date","returns", "price"]].dropna()
        price_column = self.trade_df.pop("price")  
        self.trade_df["price"] = price_column
        
        all_df = self.trade_df.copy()
        all_df = all_df.groupby("date")["returns"].sum().reset_index()        # 將相同日期的 "returns" 列做加總
        all_df.insert(0, "symbol", "portfolio")
        
        self.trade_df = pd.concat([self.trade_df, all_df], ignore_index=True)

        
        for symbol in self.trade_df['symbol'].unique():
            symbol_trades = self.trade_df[self.trade_df['symbol'] == symbol]
            if symbol == "portfolio":
                principal = self.total_initial_cash
            else:
                principal = self.initial_cash_dict.get(symbol)
                
            if len(symbol_trades) < 3:
                time=win_ratio = returns_rate = cumulative_returns = max_drawdown = sharp_ratio = sortino_ratio = annualized_return = calmar_ratio = price_trend = np.nan
            else:
                time = symbol_trades['date'].iloc[-1].strftime('%Y-%m-%d')
                win_ratio = await self.calculate_win_ratio(symbol_trades, symbol, principal)
                returns_rate = await self.calculate_returns_rate(symbol_trades, symbol, principal)
                cumulative_returns = await self.calculate_cumulative_returns(symbol_trades, symbol, principal)
                max_drawdown = await self.calculate_max_drawdown(symbol_trades, symbol, principal)
                sharp_ratio = await self.calculate_sharpe_ratio(symbol_trades, symbol, principal)
                #sortino_ratio = await self.calculate_sortino_ratio(symbol_trades, symbol, principal)
                annualized_return = await self.calculate_annualized_return(symbol_trades, returns_rate, symbol, principal)
                #calmar_ratio = await self.calculate_calmar_ratio(symbol_trades, annualized_return, symbol, principal)
                if symbol == "portfolio":
                    price_trend = None
                else:
                    price_trend = (symbol_trades['price'].iloc[-1] - symbol_trades['price'].iloc[0]) / symbol_trades['price'].iloc[0]

            # 將統計結果添加到字典中
            statistics["統計時間"].append(time)
            statistics["股票代碼"].append(symbol)
            statistics["交易勝率"].append(win_ratio)
            statistics["報酬率"].append(returns_rate)
            statistics["累計報酬"].append(cumulative_returns)
            statistics["最大回撤"].append(max_drawdown)
            statistics["夏普比率"].append(sharp_ratio)
            #statistics["Sortino比率"].append(sortino_ratio)
            #statistics["Calmar比率"].append(calmar_ratio)
            statistics["年化報酬率"].append(annualized_return)
            statistics["價格走勢"].append(price_trend)
                        
        # 將統計結果轉換為DataFrame
        statistics_df = pd.DataFrame(statistics)

        return statistics_df
    

