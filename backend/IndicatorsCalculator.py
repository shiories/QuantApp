import pandas as pd
import talib
import yfinance as yf
from backend.StockDownloader import *
import asyncio


class IndicatorsCalculator:
    
    def __init__(self, overlapping=False, momentum=False, volume=False, cycle=False, 
                 price_transform=False, volatility=False, pattern=False):
        self.stock_data = pd.DataFrame()
        self.result_df = pd.DataFrame()
        self.overlapping = overlapping
        self.momentum = momentum
        self.volume = volume
        self.cycle = cycle
        self.price_transform = price_transform
        self.volatility = volatility
        self.pattern = pattern


    async def calculate_indicators(self,symbol_data):

    
        # 從股票資料中提取相關的欄位
        prices = symbol_data
            # 刪除Close列
        prices = prices.drop(columns=['Close'])
        # 將Adj Close重命名為Close
        prices = prices.rename(columns={'Adj Close': 'Close'})
        
        #print("calculate_indicators-prices:",prices)
        
        # 使用 talib 計算常見的技術指標
        result_df = pd.DataFrame()
        
        # 重疊研究
        if self.overlapping:
            overlapping_studies = pd.DataFrame({
                'SMA': talib.SMA(prices['Close']),
                'EMA': talib.EMA(prices['Close']),
                'RSI': talib.RSI(prices['Close']),
                'MACD': talib.MACD(prices['Close'])[0],
                'B_B_Upper': talib.BBANDS(prices['Close'])[0],
                'B_B_Middle': talib.BBANDS(prices['Close'])[1],
                'B_B_Lower': talib.BBANDS(prices['Close'])[2],
                'HT_TRENDLINE': talib.HT_TRENDLINE(prices['Close']),
                'DEMA': talib.DEMA(prices['Close']),
                'KAMA': talib.KAMA(prices['Close']),
                'MA': talib.MA(prices['Close']),
                'MAMA': talib.MAMA(prices['Close'])[0],
                'MAVP': talib.MAVP(prices['Open'], prices['Close'].values),
                'MIDPOINT': talib.MIDPOINT(prices['Close']),
                'MIDPRICE': talib.MIDPRICE(prices['High'], prices['Low']),
                'SAR': talib.SAR(prices['High'], prices['Low']),
                'SAREXT': talib.SAREXT(prices['High'], prices['Low']),
                'T3': talib.T3(prices['Close']),
                'TEMA': talib.TEMA(prices['Close']),
                'TRIMA': talib.TRIMA(prices['Close']),
                'WMA': talib.WMA(prices['Close']),
            })
            result_df = pd.concat([result_df, overlapping_studies], axis=1)

        # 動量指標
        if self.momentum:
            momentum_indicators = pd.DataFrame({
                'ADX': talib.ADX(prices['High'], prices['Low'], prices['Close']),
                'ADXR': talib.ADXR(prices['High'], prices['Low'], prices['Close']),
                'APO': talib.APO(prices['Close']),
                'AROON_UP': talib.AROON(prices['High'], prices['Low'])[0],
                'AROON_DOWN': talib.AROON(prices['High'], prices['Low'])[1],
                'AROONOSC': talib.AROONOSC(prices['High'], prices['Low']),
                'BOP': talib.BOP(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CCI': talib.CCI(prices['High'], prices['Low'], prices['Close']),
                'CMO': talib.CMO(prices['Close']),
                'DX': talib.DX(prices['High'], prices['Low'], prices['Close']),
                'MACDFIX': talib.MACDFIX(prices['Close'])[0],
                'MFI': talib.MFI(prices['High'], prices['Low'], prices['Close'], prices['Volume']),
                'MINUS_DI': talib.MINUS_DI(prices['High'], prices['Low'], prices['Close']),
                'MINUS_DM': talib.MINUS_DM(prices['High'], prices['Low']),
                'MOM': talib.MOM(prices['Close']),
                'PLUS_DI': talib.PLUS_DI(prices['High'], prices['Low'], prices['Close']),
                'PLUS_DM': talib.PLUS_DM(prices['High'], prices['Low']),
                'PPO': talib.PPO(prices['Close']),
                'ROC': talib.ROC(prices['Close']),
                'ROCP': talib.ROCP(prices['Close']),
                'ROCR': talib.ROCR(prices['Close']),
                'ROCR100': talib.ROCR100(prices['Close']),
                'STOCH_K': talib.STOCH(prices['High'], prices['Low'], prices['Close'])[0],
                'STOCH_D': talib.STOCH(prices['High'], prices['Low'], prices['Close'])[1],
                'STOCHF_K': talib.STOCHF(prices['High'], prices['Low'], prices['Close'])[0],
                'STOCHF_D': talib.STOCHF(prices['High'], prices['Low'], prices['Close'])[1],
                'STOCHRSI_K': talib.STOCHRSI(prices['Close'])[0],
                'STOCHRSI_D': talib.STOCHRSI(prices['Close'])[1],
                'TRIX': talib.TRIX(prices['Close']),
                'ULTOSC': talib.ULTOSC(prices['High'], prices['Low'], prices['Close']),
                'WILLR': talib.WILLR(prices['High'], prices['Low'], prices['Close']),
            })
            result_df = pd.concat([result_df, momentum_indicators], axis=1)

        # 成交量指示器
        if self.volume:
            volume_indicators = pd.DataFrame({
                'AD': talib.AD(prices['High'], prices['Low'], prices['Close'], prices['Volume']),
                'ADOSC': talib.ADOSC(prices['High'], prices['Low'], prices['Close'], prices['Volume']),
                'OBV': talib.OBV(prices['Close'], prices['Volume']),
            })
            result_df = pd.concat([result_df, volume_indicators], axis=1)

        # 週期指示器
        if self.cycle:
            cycle_indicators = pd.DataFrame({
                'HT_DCPERIOD': talib.HT_DCPERIOD(prices['Close']),
                'HT_DCPHASE': talib.HT_DCPHASE(prices['Close']),
                'HT_PHASOR_INPHASE': talib.HT_PHASOR(prices['Close'])[0],
                'HT_PHASOR_QUADRATURE': talib.HT_PHASOR(prices['Close'])[1],
                'HT_SINE_SINE': talib.HT_SINE(prices['Close'])[0],
                'HT_SINE_LEADSINE': talib.HT_SINE(prices['Close'])[1],
                'HT_TRENDMODE': talib.HT_TRENDMODE(prices['Close']),
            })
            result_df = pd.concat([result_df, cycle_indicators], axis=1)

        # 價格轉換
        if self.price_transform:
            price_transform = pd.DataFrame({
                'AVGPRICE': talib.AVGPRICE(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'MEDPRICE': talib.MEDPRICE(prices['High'], prices['Low']),
                'TYPPRICE': talib.TYPPRICE(prices['High'], prices['Low'], prices['Close']),
                'WCLPRICE': talib.WCLPRICE(prices['High'], prices['Low'], prices['Close']),
            })
            result_df = pd.concat([result_df, price_transform], axis=1)

        # 波動率指標
        if self.volatility:
            volatility_indicators = pd.DataFrame({
                'ATR': talib.ATR(prices['High'], prices['Low'], prices['Close']),
                'NATR': talib.NATR(prices['High'], prices['Low'], prices['Close']),
                'TRANGE': talib.TRANGE(prices['High'], prices['Low'], prices['Close']),
            })
            result_df = pd.concat([result_df, volatility_indicators], axis=1)
            
        # 模式識別
        if self.pattern:

            pattern_recognition = pd.DataFrame({
                'CDL2CROWS': talib.CDL2CROWS(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDL3BLACKCROWS': talib.CDL3BLACKCROWS(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDL3INSIDE': talib.CDL3INSIDE(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDL3LINESTRIKE': talib.CDL3LINESTRIKE(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDL3OUTSIDE': talib.CDL3OUTSIDE(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLABANDONEDBABY': talib.CDLABANDONEDBABY(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLBELTHOLD': talib.CDLBELTHOLD(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLBREAKAWAY': talib.CDLBREAKAWAY(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLCONCEALBABYSWALL': talib.CDLCONCEALBABYSWALL(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLDOJI': talib.CDLDOJI(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLDOJISTAR': talib.CDLDOJISTAR(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLENGULFING': talib.CDLENGULFING(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLEVENINGSTAR': talib.CDLEVENINGSTAR(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLGAPSIDESIDEWHITE': talib.CDLGAPSIDESIDEWHITE(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLHAMMER': talib.CDLHAMMER(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLHANGINGMAN': talib.CDLHANGINGMAN(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLHARAMI': talib.CDLHARAMI(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLHARAMICROSS': talib.CDLHARAMICROSS(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLHIGHWAVE': talib.CDLHIGHWAVE(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLHIKKAKE': talib.CDLHIKKAKE(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLHOMINGPIGEON': talib.CDLHOMINGPIGEON(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLIDENTICAL3CROWS': talib.CDLIDENTICAL3CROWS(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLINNECK': talib.CDLINNECK(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLKICKING': talib.CDLKICKING(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLKICKINGBYLENGTH': talib.CDLKICKINGBYLENGTH(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLLONGLINE': talib.CDLLONGLINE(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLMARUBOZU': talib.CDLMARUBOZU(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLMATCHINGLOW': talib.CDLMATCHINGLOW(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLMATHOLD': talib.CDLMATHOLD(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLMORNINGSTAR': talib.CDLMORNINGSTAR(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLONNECK': talib.CDLONNECK(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLPIERCING': talib.CDLPIERCING(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLSHORTLINE': talib.CDLSHORTLINE(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLSPINNINGTOP': talib.CDLSPINNINGTOP(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLSTALLEDPATTERN': talib.CDLSTALLEDPATTERN(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLSTICKSANDWICH': talib.CDLSTICKSANDWICH(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLTAKURI': talib.CDLTAKURI(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLTASUKIGAP': talib.CDLTASUKIGAP(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLTHRUSTING': talib.CDLTHRUSTING(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLTRISTAR': talib.CDLTRISTAR(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS(prices['Open'], prices['High'], prices['Low'], prices['Close']),
                'CDLXSIDEGAP3METHODS': talib.CDLXSIDEGAP3METHODS(prices['Open'], prices['High'], prices['Low'], prices['Close']),
            })
            result_df = pd.concat([result_df, pattern_recognition], axis=1)

        result_df = result_df.dropna()
        return result_df


    async def get_indicators(self, stock_data):
        if stock_data.empty:
            self.stock_data = stock_data
        else:
            self.stock_data = pd.concat([self.stock_data, stock_data], axis=0)

        if self.stock_data.columns.nlevels < 2:
            self.result_df = self.calculate_indicators(self.stock_data)
        else:
            unique_symbols = self.stock_data.columns.get_level_values(1).unique()
            all_results = []
            
            async for results in self.run_group_async(unique_symbols):
                all_results.append(results)

            self.result_df = pd.concat(all_results, axis=0)

        return self.result_df


    async def run_group_async(self, unique_symbols):
        tasks = []
        for symbol in unique_symbols:
            task = asyncio.create_task(self.calculate_indicators_async(symbol))
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            yield await task


    async def calculate_indicators_async(self, symbol):
        symbol_data = self.stock_data.xs(key=symbol, level=1, axis=1, drop_level=False)
        symbol_data = symbol_data.droplevel(1, axis=1)  # 刪除原始的第二層列標題
        results = await self.calculate_indicators(symbol_data)
        results['Symbol'] = symbol  # 添加 'Symbol' 列以識別股票
        return results




if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime, time
    # 使用自己的下載套件
    stock_name = {'2330.TW': 500000, '2317.TW': 100000}  
    stock_downloader = StockDownloader(stock_name)
    start_datetime = datetime(2024, 3, 1, 12, 30, tzinfo=None)
    end_datetime = datetime(2024, 3, 10, 12, 30, tzinfo=None)
    interval = "15m"  # interval間隔：1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    stock_data = stock_downloader.range(start_datetime, end_datetime, interval)
    # 設定計算指標類型
    calculator = IndicatorsCalculator(overlapping=True, momentum=True, volume=True, cycle=True, price_transform=True, volatility=True, pattern=False)
    
    # 開始自動下載數據
    indicators_table = calculator.get_indicators(stock_data)
    print(indicators_table)

