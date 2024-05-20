import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

class OnlineCrawling:
    
    def __init__(self):
        self.rate_url ="https://www.cnyes.com/futures/html5chart/TW10YY.html"
        self.stock_url ="https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=1"

        
    def get_rate(self):
        url = self.rate_url
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        # 發送HTTP請求，並在請求中添加頭資訊
        response = requests.get(url, headers=headers)

        # 定義 risk_free_rate 變數
        risk_free_rate = None

        try:
            # 檢查請求是否成功
            if response.status_code == 200:
                # 使用BeautifulSoup解析HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # 找到表格所在的 div
                tab_div = soup.find('div', class_='tab')

                # 檢查是否找到了 tab_div 元素
                if tab_div:
                    # 使用 pandas 將 HTML 表格轉換為 DataFrame
                    df = pd.read_html(StringIO(str(tab_div)))[0]  # 使用 StringIO 物件
                    # 提取 "收盤價" 列的值
                    risk_free_rate = df.at[0, '收盤價']

                    # 列印 "收盤價" 的值
                    #print(f"無風險利率: {risk_free_rate}")
                else:
                    print("Error: 未找到包含表格的 div 元素.")
            else:
                print(f"Failed to retrieve data. Status code: {response.status_code}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
        finally:
            # 在使用完 response 物件後進行適當的清理，例如關閉連接
            response.close()

        if risk_free_rate is not None:
            risk_free_rate = float(risk_free_rate)  # 將無風險利率轉換為浮點數類型

        return risk_free_rate
    

    def get_stock(self, cutoff_year=None, min_company_count=None, excluded_industries=None):
        # 下載資料
        url = self.stock_url
        res = requests.get(url)
        html_text = res.text
        html_file = StringIO(html_text)
        stock_df = pd.read_html(html_file)[0]
        
        # 設定column名稱
        stock_df.columns = stock_df.iloc[0]
        stock_df = stock_df.iloc[1:]         
        stock_df = stock_df.dropna(thresh=3, axis=0).dropna(thresh=3, axis=1)
        selected_columns = ['有價證券代號', '有價證券名稱', '市場別', '有價證券別', '產業別', '公開發行/上市(櫃)/發行日']
        stock_df = stock_df.loc[:, selected_columns]
        stock_df = stock_df.drop(columns=['有價證券別'])
        stock_df = stock_df.rename(columns={'有價證券代號': '證券代號','有價證券名稱': '公司名稱', '公開發行/上市(櫃)/發行日': '公開發行日'})
        stock_df = stock_df.reset_index(drop=True)

        # 根據參數進行篩選
        if cutoff_year:
            stock_df['公開發行日'] = pd.to_datetime(stock_df['公開發行日'])
            stock_df = stock_df[stock_df['公開發行日'].dt.year <= cutoff_year]

        if excluded_industries:
            stock_df = stock_df[~stock_df['產業別'].isin(excluded_industries)]

        # 過濾公司數量少於最小數量的產業別
        if min_company_count:
            industry_counts = stock_df['產業別'].value_counts()
            industries_to_keep = industry_counts[industry_counts >= min_company_count].index.tolist()
            stock_df = stock_df[stock_df['產業別'].isin(industries_to_keep)]

        # 產生代碼列表
        stock_list = stock_df['證券代號'].tolist()
        stock_list = [code + '.TW' for code in stock_list]

        # 產生產業別數量表
        industry_df = stock_df.groupby('產業別').size().reset_index(name='公司數')
        industry_df = industry_df.sort_values(by='公司數', ascending=False).reset_index(drop=True)

        return stock_df, stock_list, industry_df







if __name__ == "__main__" :
    online_crawling = OnlineCrawling()
    risk_free_rate = online_crawling.get_rate()
    print(f"收盤價: {risk_free_rate}")  

    stock_df , stock_list, industry_df = online_crawling.get_stock(cutoff_year=2000, 
                                                                   excluded_industries=['電子零組件業'], min_company_count=10)
    print(f"股票列表: {stock_df}")  
    print(f"產業列表: {industry_df}")  


