import pytz
import pandas as pd
import asyncio


class TradingSystem:

    def __init__(self, initial_cash_dict, trade_unit, slippage=0):
        print(f'交易單位設定: {trade_unit}')
        self.trade_unit = trade_unit
        self.cash_dict = initial_cash_dict.copy()  # 初始本金字典：symbol -> initial_cash
        self.holdings = {}  # 股票持有情況：symbol -> quantity
        self.cost_basis = {}  # 持有成本：symbol -> cost_basis
        self.trade_history = []  # 交易歷史
        self.holdings_history = []  # 持倉歷史
        self.cost_basis_history = []  # 成本基礎歷史
        self.prices_df = pd.DataFrame()  # 價格表格
        self.trading_fee_rate = 0.001425  # 證券手續費
        self.trading_tax_rate = 0.003  # 證券交易稅
        self.slippage = slippage  #滑點參數


    def get_data(self, prices_df):
        # 確保價格表格不是空的
        if not prices_df.empty:
            # 將價格表格的時區設置為 Asia/Taipei
            self.prices_df = prices_df.tz_localize('Asia/Taipei')
        else:
            print("價格表格為空，無法更新")


    def get_tables_df(self, trade=False, holdings=False, cost=False, portfolio=False):
        data_frames = []

        if trade:
            data_frames.append(pd.DataFrame(self.trade_history))

        if holdings:
            holdings_data = []
            for date, holdings_dict in self.holdings_history:
                row = {"Date": date}
                row.update(holdings_dict)
                holdings_data.append(row)
            holdings_df = pd.DataFrame(holdings_data).fillna(0)  # 將 NaN 值填充為 0
            data_frames.append(holdings_df)

        if cost:
            cost_basis_data = []
            for entry in self.cost_basis_history:
                date, cost_dict = entry
                row = {"Date": date}
                row.update(cost_dict)
                cost_basis_data.append(row)
            cost_basis_df = pd.DataFrame(cost_basis_data).fillna(0)  # 將 NaN 值填充為 0
            data_frames.append(cost_basis_df)


    
        if portfolio is not None:
            # 添加這行以打印 portfolio_df
            # 更新 portfolio_df
            portfolio_df = self.get_portfolio_df()
            assert portfolio_df is not None, "portfolio_df 不應該為 None"
            portfolio_df.insert(0, "Date", self.prices_df.index[-1])  # 在第一列插入日期
            portfolio_df = portfolio_df.round(0)
            data_frames.append(portfolio_df)
                
        return tuple(data_frames)


    async def execute_trade(self, symbol, quantity, date):
        # 計算實際交易數量，乘以 1000
        actual_quantity = int(quantity * 1000 * self.trade_unit[symbol])
        # 初始報酬為 NaN
        returns = None
        # 將 date 物件本地化為 'Asia/Taipei'
        date = date.replace(tzinfo=pytz.timezone('Asia/Taipei'))

        # 檢查當前日期是否在價格表格的索引中
        if date not in self.prices_df.index:
            print(f"{symbol} 警告: 沒有{date}的價格資料")
            return None

        # 從價格表格中取得當天特定股票的價格
        if symbol not in self.prices_df.columns.get_level_values('Ticker'):
            print(f"{symbol} 警告: 不在價格表格中")
            return None

         # 從價格資料表格中取得當天特定股票的收盤價格
        price = pd.to_numeric(self.prices_df.loc[date, ('Close', symbol)], errors='coerce')
        cost = price * actual_quantity

        # 計算滑點交易成本
        slippage_cost = self.slippage * price * abs(actual_quantity)
        # 計算交易手續費
        trading_fee = price * actual_quantity * self.trading_fee_rate
        # 計算交易稅
        trading_tax = price * actual_quantity * self.trading_tax_rate
        
        if actual_quantity > 0: # 股票買入(cost為正值)
            if self.cash_dict[symbol] >= cost:
                #計算總成本
                cost -= trading_fee + slippage_cost
                 # 調整現金帳戶
                self.cash_dict[symbol] -= cost
                # 更新持倉
                if symbol in self.holdings:
                    self.holdings[symbol] += actual_quantity
                    # 使用新的成本來更新庫存均價
                    self.cost_basis[symbol] = (
                        self.holdings[symbol] * self.cost_basis[symbol] + cost
                    ) / (self.holdings[symbol] + actual_quantity)
                else:
                    self.holdings[symbol] = actual_quantity
                    self.cost_basis[symbol] = cost / actual_quantity

                # 列印下單訊息
                print(f"{symbol} 下單明細：在{date} 買入{actual_quantity} 單位，單價: {round(price,2)}，總成本: {round(cost,0)}，現金餘額: {round(self.cash_dict[symbol],0)}")

            else:
                print(f"{symbol} 現金不足: 無法在 {date} 上購買 {actual_quantity} 單位")

        elif actual_quantity < 0: # 股票賣出(cost為負值)
            if symbol in self.holdings and self.holdings[symbol] >= abs(actual_quantity):

                #計算總成本
                cost -= trading_fee + slippage_cost + trading_tax
                # 調整現金帳戶
                self.cash_dict[symbol] -= cost
                self.holdings[symbol] += actual_quantity
                if self.holdings[symbol] == 0:
                    del self.holdings[symbol]
                    del self.cost_basis[symbol]

                # 取得最新的庫存均價
                latest_cost_basis = self.cost_basis_history[-1][1].get(symbol, 0)
                # 計算賣出操作的收益
                returns = (price - latest_cost_basis) * -actual_quantity + trading_fee + slippage_cost + trading_tax

                # 列印下單訊息
                print(f"{symbol} 下單明細：在{date} 賣出{abs(actual_quantity)} 單位，單價: {round(price,2)}，總價: {round(cost,0)}，現金餘額: {round(self.cash_dict[symbol], 0)}")

            else:
                print(f"{symbol} 持倉不足: 無法在 {date} 上出售 {abs(actual_quantity)} 單位")

        # 保存持倉和成本基準的歷史記錄
        self.holdings_history.append((date, dict(self.holdings)))
        self.cost_basis_history.append((date, dict(self.cost_basis)))

        trade_detail = {
            "symbol": symbol,
            "date": date,
            "action": "buy" if actual_quantity > 0 else "sell",
            "price": price,
            "quantity": actual_quantity,
            "cash": self.cash_dict[symbol],
            "returns": returns
        }

        self.trade_history.append(trade_detail)
        return trade_detail


    def get_portfolio_df(self):
        # 確保價格表格不是空的
        if self.prices_df.empty:
            return None

        # 確保價格表格的索引不為空
        if self.prices_df.index.empty:
            return None

        # 從價格表格中獲取當前價格
        current_prices = self.prices_df.loc[self.prices_df.index[-1], "Close"]

        if isinstance(current_prices, pd.Series):
            current_prices = current_prices.to_dict()
        elif not isinstance(current_prices, dict):
            current_prices = {}

        # 創建 DataFrame 的數據字典
        portfolio_data = {"Symbol": [], "Stock Value": [], "Cash Value": []}

        # 創建一個集合，包含 initial_cash_dict 和 self.holdings 中所有的股票
        all_symbols = set(self.cash_dict.keys()).union(set(self.holdings.keys()))

        # 迴圈計算每支股票的價值
        for symbol in all_symbols:
            if symbol in current_prices:
                # 計算每支股票的市值和現金
                stock_value = current_prices[symbol] * self.holdings.get(symbol, 0)
                cash_value = self.cash_dict.get(symbol, 0)
                # 將結果添加到 DataFrame
                portfolio_data["Symbol"].append(symbol)
                portfolio_data["Stock Value"].append(stock_value)
                portfolio_data["Cash Value"].append(cash_value)

        # 將結果轉換為 DataFrame
        portfolio_df = pd.DataFrame(portfolio_data)

        # 添加新的列到 DataFrame，表示每行總和
        portfolio_df["Total"] = portfolio_df["Stock Value"] + portfolio_df["Cash Value"]
        # 添加新的行到 DataFrame，表示總計
        portfolio_df.loc[len(portfolio_df)] = ["Portfolio", portfolio_df["Stock Value"].sum(), 
                                               portfolio_df["Cash Value"].sum(), portfolio_df["Total"].sum()]
        
        return portfolio_df


    async def liquidate_position(self, symbol, date):
        # 将 date 对象本地化为 'Asia/Taipei'
        date = date.replace(tzinfo=pytz.timezone('Asia/Taipei'))
        
        # 檢查當前日期是否在價格表格的索引中
        if date not in self.prices_df.index:
            print(f"{symbol} 警告: 沒有 {date} 的價格數據，無法平倉")

            return None

        # 平倉指定標的的持倉
        if symbol in self.holdings:
            # 計算平倉全部
            quantity = round(-self.holdings[symbol] / (1000 * self.trade_unit[symbol]),0)
            # 新增以下 print 語句以顯示相關數據
            print(f"{symbol} 目前庫存: {self.holdings[symbol]} 單位 , 執行平倉:  在{date}平倉 {abs(quantity)} 單位")

            await self.execute_trade(symbol, quantity, date)

            # 新增以下 print 語句以顯示相關數據
            print(f"平倉後庫存情況：{self.holdings}")

        else:
            print(f"{symbol} 無需平倉:  {symbol}的庫存為零")


    async def adjust_position(self, symbol, date, proportion):
        # 将 date 对象本地化为 'Asia/Taipei'
        date = date.replace(tzinfo=pytz.timezone('Asia/Taipei'))
        
        # 檢查當前日期是否在價格表格的索引中
        if date not in self.prices_df.index:
            print(f"{symbol} 警告: 沒有 {date} 的價格數據，無法調整")
            return None

        # 調整指定標的的持倉
        if symbol in self.holdings:
            if proportion is not None and (-1 <= proportion <= 1):
                # 計算調整數量
                quantity = int(self.holdings[symbol] * proportion) / (1000 * self.trade_unit[symbol])

                # 新增以下 print 語句以顯示相關數據
                if proportion > 0:
                    print(f"{symbol} 加碼: 在 {date} 買入 {abs(quantity)} 單位")
                elif proportion < 0:
                    print(f"{symbol} 減碼: 在 {date} 賣出 {abs(quantity)} 單位")
                print(f"調整前庫存情況：{self.holdings}")

                await self.execute_trade(symbol, quantity, date)

                # 新增以下 print 語句以顯示相關數據
                print(f"調整後庫存情況：{self.holdings}")
            else:
                print("比例應在 -1 到 1 之間")
        else:
            print(f"{symbol} 無法調整:  {symbol}的庫存為零")


    def print_current_state(self):
        # 打印最新的持倉歷史
        if self.holdings_history:
            holdings_history_df = pd.DataFrame(self.holdings_history).tail(1)
            print("最新持倉明細：\n", holdings_history_df)

        # 打印最新的成本基礎歷史
        if self.cost_basis_history:
            cost_basis_history_df = pd.DataFrame(self.cost_basis_history).tail(1)
            print("最新庫存均價：\n", cost_basis_history_df)

        # 打印最新的投資組合價值
        portfolio_value = self.get_portfolio_df()
        print("最新投資組合價值：", portfolio_value)
        

