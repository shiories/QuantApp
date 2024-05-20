import flet as ft
import datetime
import sys
from backend.ModelPreparation import *
from frontend.DataPage import *
from frontend.AppTile import *
from frontend.OutputRedirector import *

  

class ModelPage:
    
    def __init__(self, app, page):
        self.app = app
        self.page = page
        self.download_buttons = []  # 用於存儲下載按鈕實例
        self.test_results = None
        self.selected_features_df = None
        self.voting_results = None
        self.historical_data = None
        self.indicators_data = None
        self.Initial_principal = "{'2330.TW': 500000, '2317.TW': 100000}"
        self.latest_date="2010-01-01"
        self.attributes = None


    def create_app(self, df, df_name):
        self.app_tile = AppTile( name=df_name, view=DataPage(self.app, df=df).build(), app=self.app)
        self.app_tile.open_app(df)  


    def get_value(self):
        self.Initial_principal, self.latest_date = self.page.portfolio_page.transmit_value()
        self.Initial_principal_textfield.value= self.Initial_principal
        self.app.update()


    def update_start_date(self):        # 更新日期
        self.start_date_textfield.value = self.start_date_picker.value.strftime("%Y-%m-%d")
        self.app.update()


    def update_end_date(self):        # 更新日期
        self.end_date_textfield.value = self.end_date_picker.value.strftime("%Y-%m-%d")
        self.app.update()


    def handle_change(self):
        selected_values = self.indicators_button.selected
        self.attributes = {
            "overlapping": "1" in selected_values,
            "momentum": "2" in selected_values,
            "volume": "3" in selected_values,
            "cycle": "4" in selected_values,
            "price_transform": "5" in selected_values,
            "volatility": "6" in selected_values
        }


    def build(self):
        # 創建一個文字方塊控制項，用於顯示程式輸出
        self.output_textbox = ft.Text(value="", overflow=ft.TextOverflow.VISIBLE)
        
        # 開始日期選擇器
        self.start_date_picker = ft.DatePicker(help_text="開始日期",opacity=0.9,value=datetime(2016, 1, 1),on_change= lambda _: self.update_start_date(),
                                    first_date=datetime(2000, 1, 1), last_date=datetime(2025, 12, 31) )
        self.app.overlay.append(self.start_date_picker)
        self.start_date_button = ft.ElevatedButton( "選擇日期", icon=ft.icons.CALENDAR_MONTH, on_click=lambda _: self.start_date_picker.pick_date() )
        self.start_date_textfield = ft.TextField(hint_text=  self.start_date_picker.value.strftime("%Y-%m-%d"), max_length=10, label="開始日期", expand=True)

        # 結束日期選擇器
        self.end_date_picker = ft.DatePicker(help_text="結束日期",opacity=0.9,value=datetime(2022, 12, 31),on_change= lambda _: self.update_end_date(),
                                    first_date=datetime(2000, 1, 1), last_date=datetime(2025, 12, 31) )
        self.app.overlay.append(self.end_date_picker)
        self.end_date_button = ft.ElevatedButton( "選擇日期", icon=ft.icons.CALENDAR_MONTH, on_click=lambda _: self.end_date_picker.pick_date() )
        self.end_date_textfield = ft.TextField(hint_text=  self.end_date_picker.value.strftime("%Y-%m-%d"), max_length=10, label="結束日期", expand=True)


        # 投資組合
        self.Initial_principal_textfield = ft.TextField(value=f"{self.Initial_principal}", hint_text="{'2330.TW': 500000, '2317.TW': 100000}", label="投資組合", expand=True)

        # 移動滑軌
        self.x_interval_slider = ft.Slider(value=1, min=1, max=20, label="   移動平均窗口 : {value} 天  ", divisions=19, expand=True)
        self.min_quantity_slider = ft.Slider(value=3, min=1, max=8, label="   最小集成模型數 : {value}   ", divisions=7, expand=True)

        #執行按鈕
        self.execute_button = ft.ElevatedButton(text="     執  行     ", on_click=lambda _: self.execute_strategy(), expand=True)

        self.indicators_button = ft.SegmentedButton( allow_multiple_selection=True,width=500,
                selected_icon=ft.Icon(ft.icons.CANDLESTICK_CHART),key="指標類型",
                selected={"1"},
                on_change= lambda _: self.handle_change(),
                segments=[
                    ft.Segment(value="1", label=ft.Text("重疊"), tooltip="21種指標"),
                    ft.Segment(value="2", label=ft.Text("動量"), tooltip="31種指標"),
                    ft.Segment(value="3", label=ft.Text("成交"), tooltip="3種指標"),
                    ft.Segment(value="4", label=ft.Text("週期"), tooltip="7種指標"),
                    ft.Segment(value="5", label=ft.Text("價格"), tooltip="4種指標"),
                    ft.Segment(value="6", label=ft.Text("波動"), tooltip="3種指標"),
                ]
            )

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
                        self.indicators_button,
                    ]
                ),
                ft.Container(height=15), 
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.x_interval_slider,
                        self.min_quantity_slider,
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
            ft.ElevatedButton(text="訓練-模型組合", on_click=lambda _: self.create_app(self.test_results, "訓練-模型組合"), visible=False, expand=True),
            ft.ElevatedButton(text="訓練-特徵篩選", on_click=lambda _: self.create_app(self.selected_features_df, "訓練-特徵篩選"), visible=False, expand=True),
            ft.ElevatedButton(text="訓練-投票模型", on_click=lambda _: self.create_app(self.voting_results, "訓練-投票模型"), visible=False, expand=True),

            ]

        ft.SegmentedButton

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
            
        self.app.splash = ft.ProgressBar()
        self.execute_button.text = "     請稍後...    " 
        self.app.update()       

        # 在這裡執行投資組合策略的代碼
        start_date = self.start_date_textfield.value
        end_date = self.end_date_textfield.value
        self.x_interval = int(self.x_interval_slider.value)
        min_quantity = self.min_quantity_slider.value
        
        self.handle_change()
        
        if isinstance(self.Initial_principal_textfield.value, str):
            self.Initial_principal = eval(self.Initial_principal_textfield.value)
        else:
            self.Initial_principal = self.Initial_principal_textfield.value
                
        freq = 'd'   # 間隔：1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        unit = 1

        # 運行投資組合策略
        model_prep = ModelPreparation(self.Initial_principal, start_date, end_date, freq,  unit,  self.x_interval, min_quantity, self.attributes)
        self.results, self.historical_data, self.indicators_data, self.voting_results, self.test_results, self.selected_features_df = model_prep.run()
        #time.sleep(5)
        # 設置下載按鈕可見性
        
        for button in self.download_buttons:
            button.visible = True

        self.app.splash = None
        self.execute_button.text = "     執  行     " 
        self.page.backtest_page.get_value()
        self.app.update()       


    def transmit_value(self):
        return self.Initial_principal, self.results, self.historical_data, self.indicators_data, self.attributes, self.x_interval




if __name__=="__main__":

    def main(page: ft.Page):
        # 主函數，設置頁面標題與初始化應用程式
        page.title = "量化分析應用"
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.scroll = ft.ScrollMode.ADAPTIVE
        page.window_width = 1500
        page.window_height = 650
        # 創建應用程式控制項並將其添加到頁面
        model_page = ModelPage(page,None)
        page.add(model_page.build())  # 將build()方法添加到頁面

    # 啟動應用程式
    ft.app(main)
