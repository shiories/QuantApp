import flet as ft
from frontend.MainPage import *
from frontend.PortfolioPage import *
from frontend.ModelPage import *
from frontend.BacktestPage import *



class QuantApp:
    
    def __init__(self, app):
        self.app = app
        self.portfolio_page = PortfolioPage(self.app, self)
        self.model_page = ModelPage(self.app, self)
        self.backtest_page = BacktestPage(self.app, self)
        self.main_page = MainPage(self.app, self)
        # 儲存頁面實例的字典
        self.pages = {
            "主頁": self.main_page.build(),
            "投資組合": self.portfolio_page.build(),
            "模型訓練": self.model_page.build(),
            "交易回測": self.backtest_page.build(),
        }
        self.app.adaptive = True
        self.current_page = None
        self.visible_pages = [self.pages["主頁"], self.pages["投資組合"], self.pages["模型訓練"], self.pages["交易回測"]]
        self.current_index = 0
        self.visible_pages[self.current_index].visible = True
        # 呼叫 main 函數
        self.main()
        

    def build(self):
        # 初始化標籤導航
        self.filter = ft.Tabs(
            scrollable=False,
            selected_index="",
            tabs=[
                ft.Tab(text="主頁"),
                ft.Tab(text="投資組合"),
                ft.Tab(text="模型訓練"),
                ft.Tab(text="交易回測")
            ],
            on_change=lambda _: self.tabs_changed(),
        )

        # 初始頁面佈局
        self.layout = ft.Column(adaptive=True, controls=[self.filter] + self.visible_pages)

        return self.layout


    def main(self):
        # 設定頁面標題與初始化應用程式
        self.app.title = "量化分析應用"
        self.app.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.app.scroll = ft.ScrollMode.ADAPTIVE
        self.app.window_width = 1500
        self.app.window_height = 700
        self.app.adaptive = True
        self.app.expand = True
        self.app.theme_mode = ft.ThemeMode.DARK
        # 建立應用程式控制項並將其新增至頁面
        self.app.add(self.build())
        self.navigate("主頁")


    def tabs_changed(self):
        # 根據選擇的標籤切換頁面
        status = self.filter.tabs[self.filter.selected_index].text
        self.navigate(status)


    def set_selected_tab(self, tab_name):
        # 遍歷標籤列表，尋找符合的標籤，並將其索引設為選定索引
        for i, tab in enumerate(self.filter.tabs):
            if tab.text == tab_name:
                self.filter.selected_index = i    
        self.tabs_changed()


    def navigate(self, page_name):
        # 導航至指定頁面
        if page_name in self.pages:
            # 關閉其他頁面
            for page in self.visible_pages:
                if page != self.pages[page_name]:
                    page.visible = False
            # 顯示新頁面
            self.current_index = list(self.pages.keys()).index(page_name)
            self.visible_pages[self.current_index].visible = True
            self.app.update()
        else:
            print(f"Error: Page '{page_name}' not found.")


# 啟動應用程式
ft.app(QuantApp)
