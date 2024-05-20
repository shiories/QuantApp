import flet as ft
import sys
from frontend.DataPage import *
from frontend.AppTile import *
from frontend.OutputRedirector import *
from frontend.FileExecution import *
from backend.PortfolioStrategy import *




class PortfolioPage:
    
    def __init__(self, app, page):
        self.app = app
        self.page =page
        self.download_buttons = []  # 用於存儲下載按鈕實例
        self.Initial_principal = ""
        self.latest_date="2010-01-01"
        self.test_results = None
        self.selected_features_df = None
        self.voting_results = None
        self.stats_df = None
        self.file_picker = FileExecution(self.app, self.page)
        self.file_name = None

        
    def create_app(self, df, df_name):
        self.app_tile = AppTile( name=df_name, view=DataPage(self.app, df=df).build(), app=self.app)
        self.app_tile.open_app(df)  
              
        
    def files_result(self):
        self.file_name = self.file_picker.get_file_name()
        self.app.update()
        return self.file_name

        
    def build(self):
        # 創建一個文字方塊控制項，用於顯示程式輸出
        self.output_textbox = ft.Text(value="", overflow=ft.TextOverflow.VISIBLE)
        
        # 建立檔選擇器和文字控制項
        self.file_picker_button = self.file_picker.build_file_picker()
        
        
        # 創建文字方塊控制項、下拉式功能表、滑塊和按鈕
        self.file_name_textfield = ft.TextField(hint_text=".xlsx", label="excel檔案名稱", expand=True)
        self.min_year_textfield = ft.TextField(hint_text=2015, max_length=4, label="最小年度範圍", expand=True)
        self.max_year_textfield = ft.TextField(hint_text=2023, max_length=4, label="最大年度範圍", expand=True)
        self.min_industry_mean_textfield = ft.TextField(hint_text=0.05, label="最小產業平均報酬", expand=True)
        self.total_investment_textfield = ft.TextField(hint_text=1000000, label="投資組合總本金", expand=True)
        self.portfolio_dropdown = ft.Dropdown(label="選擇投資組合模型", hint_content="最大夏普比率模型", expand=True,
            options=[
                ft.dropdown.Option("最大夏普比率模型"),
                ft.dropdown.Option("最小方差模型"),
                ft.dropdown.Option("馬可維茲模型"),
                ft.dropdown.Option("等權重模型"),
            ],
        )
        self.min_company_count_slider = ft.Slider(value=25, min=0, max=50, label="   產業別最少公司數 : {value}   ", divisions=50, expand=True)
        self.min_quantity_slider = ft.Slider(value=3, min=1, max=8, label="   最小集成模型數 : {value}   ", divisions=7, expand=True)
        self.multiplier_slider = ft.Slider(value=7, min=1, max=10, label="   最少平均值乘數(越大越嚴格) : {value}   ", divisions=9, expand=True)
        self.execute_button = ft.ElevatedButton(text="     執  行     ", on_click=lambda _: self.execute_strategy(), expand=True)


        # 創建左側佈局
        left_up = ft.Column(adaptive=True, expand=4,
            controls=[ 
                ft.Row(adaptive = True,
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.file_picker_button,
                        ft.Container(width=10), 
                        self.min_quantity_slider
                    ]
                ),
                ft.Container(height=15), 
                ft.Row(adaptive = True,
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.min_year_textfield,
                        ft.Container(width=10), 
                        self.max_year_textfield,
                    ]
                ),
                ft.Row(adaptive = True,
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.min_industry_mean_textfield,
                        ft.Container(width=10), 
                        self.multiplier_slider,
                    ]
                ),
                ft.Container(height=15), 
                ft.Row(adaptive = True,
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(width=10), 
                        self.portfolio_dropdown,
                        ft.Container(width=10), 
                        self.total_investment_textfield,
                    ]
                ),
                ft.Container(height=15), 
                ft.Row(adaptive = True,
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        self.min_company_count_slider,
                    ]
                ),
            ],
        )

        # 創建右上方佈局
        right_up = ft.Column(expand=7, height=450,
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
        left_down = ft.Row(adaptive=True, expand=4,
                    controls=[self.execute_button]
                    ) 

        # 在build()方法中為下載按鈕創建ft.Row控制項
        self.download_buttons = [
            ft.ElevatedButton(text="選股-模型組合", on_click=lambda _: self.create_app(self.test_results, "選股-模型組合"), visible=False, expand=True),
            ft.ElevatedButton(text="選股-特徵篩選", on_click=lambda _: self.create_app(self.selected_features_df, "選股-特徵篩選"), visible=False, expand=True),
            ft.ElevatedButton(text="選股-投票模型", on_click=lambda _: self.create_app(self.voting_results, "選股-投票模型"), visible=False, expand=True),
            ft.ElevatedButton(text="預測統計", on_click=lambda _: self.create_app(self.stats_df, "預測統計"), visible=False, expand=True)
            ]

        

        right_down =ft.Row(width=900, adaptive=True, height=40,
                           controls= self.download_buttons, )
        

        # 返回佈局，左右兩側佈局並排
        return ft.Column(controls=[
                ft.Container(height=10), 
                ft.Row( width=1500, adaptive=True,spacing=30, alignment=ft.MainAxisAlignment.SPACE_AROUND,
                       controls=[ft.Container(width=5), left_up, ft.Container(width=20), right_up, ft.Container(width=5)]),
                ft.Container(height=40),
                ft.Row( width=1500, adaptive=True,spacing=30, alignment=ft.MainAxisAlignment.SPACE_AROUND,
                       controls=[ft.Container(width=5), left_down, ft.Container(width=20), right_down, ft.Container(width=5)]),
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
        file_name = self.files_result()
        print(f'檔案檢查:{file_name}')
        min_company_count = int(self.min_company_count_slider.value)
        portfolio = self.portfolio_dropdown.value
        min_year = int(self.min_year_textfield.value)
        max_year = int(self.max_year_textfield.value)
        min_quantity = int(self.min_quantity_slider.value)
        min_industry_mean = float(self.min_industry_mean_textfield.value)
        multiplier = int(self.multiplier_slider.value)
        total_investment = int(self.total_investment_textfield.value)
        
        # 創建中英文映射字典
        portfolio_mapping = {
            "最大夏普比率模型": "max_sharpe",
            "最小方差模型": "min_variance",
            "馬可維茲模型": "mvo_portfolio",
            "等權重模型": "equal_weights"
        }

        # 獲取中文字串並映射為對應的英文字串
        portfolio_value = portfolio_mapping.get(portfolio, portfolio)

        # 運行投資組合策略
        portfolio_strategy = PortfolioStrategy(min_year, max_year, min_company_count, file_name, min_quantity)
        self.test_results, self.selected_features_df, self.voting_results, self.stats_df, self.Initial_principal, self.latest_date = portfolio_strategy.run(total_investment, portfolio_value, min_industry_mean, multiplier)
        #time.sleep(5)
        # 設置下載按鈕可見性
        for button in self.download_buttons:
            button.visible = True

        self.app.splash = None
        self.execute_button.text = "     執  行     " 
        self.app.update()       
        self.page.model_page.get_value()


    def transmit_value(self):
        return self.Initial_principal, self.latest_date




if __name__=="__main__":

    def main(page: ft.Page):
        # 主函數，設置頁面標題與初始化應用程式
        page.title = "量化分析應用"
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.scroll = ft.ScrollMode.ADAPTIVE
        page.window_width = 1500
        page.window_height = 650
        # 創建應用程式控制項並將其添加到頁面
        portfolio_page = PortfolioPage(page,None)
        page.add(portfolio_page.build())  # 將build()方法添加到頁面

    # 啟動應用程式
    ft.app(main)
