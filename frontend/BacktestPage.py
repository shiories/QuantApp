import datetime
import asyncio
import sys
import flet as ft
from backend.BacktestTrader import *
from frontend.DataPage import *
from frontend.AppTile import *
from frontend.OutputRedirector import *
  

class BacktestPage:

    def __init__(self, app, page):
        self.app = app
        self.page = page
        self.download_buttons = []  # 用於存儲下載按鈕實例
        self.test_results = None
        self.selected_features_df = None
        self.voting_results = None
        self.Initial_principal = "{'2330.TW': 500000, '2317.TW': 100000}"
        self.latest_date="2010-01-01"
        self.attributes = None

    def create_app(self, df, df_name):
        self.app_tile = AppTile( name=df_name, view=DataPage(self.app, df=df).build(), app=self.app)
        self.app_tile.open_app(df)  

    def get_value(self):
        self.Initial_principal, self.results, self.historical_data, self.indicators_data, self.attributes, self.x_interval = self.page.model_page.transmit_value()
        self.Initial_principal_textfield.value= self.Initial_principal
        self.app.update()

    def update_start_date(self):        # 更新日期
        self.start_date_textfield.value = self.start_date_picker.value.strftime("%Y-%m-%d")
        self.app.update()

    def update_end_date(self):        # 更新日期
        self.end_date_textfield.value = self.end_date_picker.value.strftime("%Y-%m-%d")
        self.app.update()

    def build(self):

        # 創建一個文字方塊控制項，用於顯示程式輸出
        self.output_textbox = ft.Text(value="", overflow=ft.TextOverflow.VISIBLE)
        
        # 開始日期選擇器
        self.start_date_picker = ft.DatePicker(help_text="開始日期",opacity=0.9,value=datetime(2023, 1, 1),on_change= lambda _: self.update_start_date(),
                                    first_date=datetime(2000, 1, 1), last_date=datetime(2025, 12, 31) )
        self.app.overlay.append(self.start_date_picker)
        self.start_date_button = ft.ElevatedButton( "選擇日期", icon=ft.icons.CALENDAR_MONTH, on_click=lambda _: self.start_date_picker.pick_date() )
        self.start_date_textfield = ft.TextField(hint_text=  self.start_date_picker.value.strftime("%Y-%m-%d"), max_length=10, label="開始日期", expand=True)

        # 結束日期選擇器
        self.end_date_picker = ft.DatePicker(help_text="結束日期",opacity=0.9,value=datetime(2023, 12, 31),on_change= lambda _: self.update_end_date(),
                                    first_date=datetime(2000, 1, 1), last_date=datetime(2025, 12, 31) )
        self.app.overlay.append(self.end_date_picker)
        self.end_date_button = ft.ElevatedButton( "選擇日期", icon=ft.icons.CALENDAR_MONTH, on_click=lambda _: self.end_date_picker.pick_date() )
        self.end_date_textfield = ft.TextField(hint_text=  self.end_date_picker.value.strftime("%Y-%m-%d"), max_length=10, label="結束日期", expand=True)

        # 投資組合
        self.Initial_principal_textfield = ft.TextField(value=f"{self.Initial_principal}", hint_text="{'2330.TW': 500000, '2317.TW': 100000}", label="投資組合", expand=True)

        # 移動滑軌
        self.min_score_slider = ft.Slider(value=60, min=1, max=100, label="   最低模型分數 : {value}%   ", divisions=99, expand=True)
        self.multiplier_slider = ft.Slider(value=100, min=10, max=1000, label="   可交易次數 : {value} 次  ", divisions=99, expand=True)

        #執行按鈕
        self.execute_button = ft.ElevatedButton(text="     執  行     ", on_click=lambda _: self.execute_strategy(), expand=True)

        # 創建左側佈局
        left_up = ft.Column(width=500,adaptive=True,
            controls=[ 
                ft.Container(height=40), 
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.Initial_principal_textfield
                    ]
                ),
                ft.Container(height=15), 
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.start_date_textfield,
                        ft.Container(width=10), 
                        self.start_date_button
                    ]
                ),
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.end_date_textfield,
                        ft.Container(width=10), 
                        self.end_date_button,
                    ]
                ),
                ft.Container(height=15), 
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.min_score_slider,
                    ]
                ),
                ft.Container(height=15), 
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.multiplier_slider,
                    ]
                ),
                ft.Container(height=15), 
            ],
        )

        # 創建右上方佈局
        right_up = ft.Column(adaptive=True,
                height=450,
                width=900,
                scroll=ft.ScrollMode.AUTO,
                auto_scroll=True,
            controls=[
                ft.Container(height=40), 
                self.output_textbox,
                ft.Container(height=10), 
                  # 使用ft.Row將下載按鈕水準排列
            ]
        )

        # 創建左下方佈局
        left_down = ft.Row(width=520,adaptive=True,
                    controls=[self.execute_button]
                    ) 

        # 在build()方法中為下載按鈕創建ft.Row控制項
        self.download_buttons = [
            ft.ElevatedButton(text="統計資料", on_click=lambda _: self.create_app(self.statistics_df, "統計資料"), visible=False, expand=True),
            ft.ElevatedButton(text="交易明細", on_click=lambda _: self.create_app(self.trade_df, "交易明細"), visible=False, expand=True),
            ft.ElevatedButton(text="持倉數量", on_click=lambda _: self.create_app(self.holdings_df, "持倉數量"), visible=False, expand=True),
            ft.ElevatedButton(text="庫存均價", on_click=lambda _: self.create_app(self.cost_df, "庫存均價"), visible=False, expand=True),
            ft.ElevatedButton(text="資金總計", on_click=lambda _: self.create_app(self.portfolio_df, "資金總計"), visible=False, expand=True),

            ]
        

        right_down =ft.Row(
            width=900, adaptive=True, height=40,
                           controls= self.download_buttons, )
        

        # 返回佈局，左右兩側佈局並排
        return ft.Column(controls=[
                ft.Row( width=1500, adaptive=True, height=500,
                       controls=[left_up, ft.Container(width=20), right_up]),
                ft.Container(width=20), 
                ft.Row( width=1500, adaptive=True, height=50,
                       controls=[left_down, ft.Container(width=20),right_down]),
                ]
        )

    def execute_strategy(self):
        sys.stdout = OutputRedirector(self.output_textbox)
        sys.stderr = OutputRedirector(self.output_textbox)

        for button in self.download_buttons:
            button.visible = False
        self.get_value()
        self.app.splash = ft.ProgressBar()
        self.execute_button.text = "     請稍後...    " 
        self.app.update()       

        # 在這裡執行投資組合策略的代碼
        start_date = self.start_date_textfield.value
        end_date = self.end_date_textfield.value
        multiplier = self.multiplier_slider.value
        min_score = self.min_score_slider.value / 100

        if isinstance(self.Initial_principal_textfield.value, str):
            Initial_principal = eval(self.Initial_principal_textfield.value)
        else:
            Initial_principal = self.Initial_principal_textfield.value
                        
        freq = 'd'   # 間隔：1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        unit = 1
        # 運行投資組合策略
        trading_backtest = BacktestTrader(Initial_principal, multiplier, freq, unit, self.x_interval, start_date,
                                        end_date, self.results, self.historical_data, min_score, self.attributes)
    
        asyncio.run(trading_backtest.run())
        self.statistics_df, self.trade_df, self.holdings_df, self.cost_df, self.portfolio_df = asyncio.run(trading_backtest.get_table())

        #time.sleep(5)
        # 設置下載按鈕可見性

        for button in self.download_buttons:
            button.visible = True

        self.app.splash = None
        self.execute_button.text = "     執  行     " 
        self.app.update()       





if __name__=="__main__":

    def main(page: ft.Page):
        # 主函數，設置頁面標題與初始化應用程式
        page.title = "量化分析應用"
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.scroll = ft.ScrollMode.ADAPTIVE
        page.window_width = 1500
        page.window_height = 650
        # 創建應用程式控制項並將其添加到頁面
        backtest_page = BacktestPage(page,None)
        page.add(backtest_page.build())  # 將build()方法添加到頁面

    # 啟動應用程式
    ft.app(main)

