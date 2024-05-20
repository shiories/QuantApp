import flet as ft
import time
from frontend.FileExecution import *


class AppTile:
    def __init__(self, name, view, app):
        self.view = view
        self.name = name
        self.app = app
        self.icon = ft.IconButton(icon=ft.icons.KEYBOARD_BACKSPACE_ROUNDED,on_click=lambda _: self.close_app())
        self.file_picker = FileExecution(self.app, None)


    def open_app(self, df):
        # 建立文件選擇器和文字控制項
        self.file_picker_button = self.file_picker.build_file_saver(df)
        
        self.app.views.append(
            ft.View(scroll=ft.ScrollMode.ADAPTIVE,adaptive=True,
                controls=[ ft.AppBar( title=ft.Text(f"{self.name}") , leading=self.icon,
                                     actions=[self.create_download_button(df),ft.Container(width=50), self.file_picker_button,ft.Container(width=50)] ), self.view ]
                )
            )
        
        self.app.open = True
        self.app.update()


    def create_download_button(self, df):
        self.download_button = ft.ElevatedButton( visible=True, expand=True,
                                        text="            下 載                ", icon=ft.icons.FILE_DOWNLOAD, on_click=lambda _: self.download_data(df))
        return self.download_button


    def close_app(self):
        self.app.open = False
        self.app.views.pop()
        self.app.update()


    def download_data(self, df):       
        file_path = f"{self.name}.xlsx"
        #檢查日期時間列是否有時區信息
        
        import re        
        for col in df.columns:
            if re.search(r'\[ns, .*?\]$', str(df[col].dtype)):
                    df[col] = df[col].dt.tz_localize(None)
                
        df.to_excel(file_path, index=True)
        print(f"已保存到 {file_path}")
        self.download_button.text=f"已保存到 {file_path}"
        self.app.update()
        time.sleep(5)
        self.download_button.text="                下 載                "
        self.app.update()



        