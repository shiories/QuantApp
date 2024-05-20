import flet as ft

class MainPage:
    
    def __init__(self, app, page):
        self.app = app
        self.page = page


    def _create_popup_button(self):
        # 创建弹出窗口按钮
        popup_dialog = ft.AlertDialog(bgcolor=ft.colors.with_opacity(1, '#0c151b'),
            title=ft.Text(value="指導教授：楊耿杰老師", style=ft.TextThemeStyle.HEADLINE_SMALL),
            content=ft.Text(style=ft.TextThemeStyle.TITLE_LARGE,
                value=f"學    生：\n       C108161148張名華\n       C110161133陳衍赫\n       C110161136林嘉晟\n       C110161138林江山\n       C110161139鄧佳慶")
        )
        popup_button = ft.IconButton( icon=ft.icons.FIRST_PAGE,icon_size=50,
            on_click=lambda _: self.show_popup_dialog(popup_dialog)
        )
        
        return ft.Row(
            alignment=ft.MainAxisAlignment.START,
            controls=[ popup_button ])
        

    def show_popup_dialog(self, popup_dialog):
        # 显示弹出窗口
        self.app.dialog = popup_dialog
        popup_dialog.open = True
        self.app.update()



    def build(self):
        self.icon = self._create_popup_button()
        return ft.Column(
            controls=[
                ft.Container(height=15), 
                ft.Row(controls=[ self.icon , ft.Text(value="高雄科技大學 金融資訊系 機器學習結合基本面與技術面-選股與交易的應用", style=ft.TextThemeStyle.DISPLAY_SMALL)],
                    alignment=ft.MainAxisAlignment.CENTER,adaptive = True,
                ),
                ft.Container(height=15), 
                ft.Row(adaptive = True,
                    controls=[
                        ft.Container(width=70),
                        self._create_page_intro("投資組合", f"      透過自備的特徵表格來篩選股票，表格需有'Stock code'列，以及'財報發布日'列。\n\n      首先會自動下載股票資訊將表格篩選出符合條件的股票，並自動附加收盤價儲存回提供的檔案，接下來進行各產業別的模型配對與擬合最佳投票模型，並依照條件篩選出符合的股票代碼，並組成投資組合。",
                                                "投資組合"),
                        ft.Container(width=50),
                        self._create_page_intro("交易模型訓練", f"      將投資組合的個別股票進行訓練、模型組合、擬合投票模型、並選出最優模型。\n\n      可經由第一分頁取得投資組合，也可依照格式自行設計。",
                                                "模型訓練"),
                        ft.Container(width=50),
                        self._create_page_intro("交易回測", f"      利用得到的各公司最佳模型來進行回測交易，並進行統計分析。\n\n      此程式必須先經過第二分頁的模型訓練。",
                                                "交易回測"),
                        ft.Container(width=80),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    spacing=20,
                ),
                ft.Container(height=40), 
                ft.Row(width=1500,adaptive = True,
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    controls=[ft.Container(width=360),
                        ft.Text("此軟體僅供學術研究用途，不應用於實際交易，請自行承擔責任。", style=ft.TextThemeStyle.TITLE_LARGE, expand=True),
                        ft.Container(width=360),                    
                    ]
                ),
            ],
        )

    def _create_page_intro(self, title, description, page_name):
        return ft.Container(
            border=ft.border.all(2, ft.colors.BLUE_100),
            border_radius= ft.border_radius.all(30),
            padding=20,
            height=350,
            content=
                ft.Column(
                    expand=True,
                    alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.TextButton(on_click= lambda _: self.page.set_selected_tab(page_name),
                            content=ft.Column(alignment=ft.MainAxisAlignment.CENTER, 
                                controls=[ft.Text(width=300, value=title, style=ft.TextThemeStyle.HEADLINE_MEDIUM, color=ft.colors.BLUE_100),
                                          ])),
                        ft.Text(width=330, value=description, style=ft.TextThemeStyle.BODY_LARGE)
                    ]
                )
        )




