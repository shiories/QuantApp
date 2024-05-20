import flet as ft
import os
import pandas as pd


class FileExecution:
    
    def __init__(self, app, page):
        self.app = app
        self.page = page


    def get_file_name(self):
        return self.file_path
    
    
    def pick_files_result(self, e: ft.FilePickerResultEvent):
        for file_info in e.files:
            file_name = os.path.basename(file_info.path)  # 获取文件名
            self.file_path = file_info.path   # 获取文件路径
            self.file_picker_button.text = f'已載入 {file_name}'
            self.app.update()
        return self.file_path
    
        
    def build_file_picker(self):
        # 建立文件選擇器和文字控制項
        self.file_picker = ft.FilePicker(on_result=self.pick_files_result)
        self.file_picker_button = ft.ElevatedButton(text="           選擇檔案               ", icon=ft.icons.UPLOAD_FILE, on_click=lambda _: self.file_picker.pick_files(allowed_extensions=["xlsx"]))

        # 将文件选择器添加到页面上
        self.app.overlay.append(self.file_picker)

        return self.file_picker_button


    def save_file_result(self, e: ft.FilePickerResultEvent):
        file_info = e.path
        print(file_info)
        if file_info is not None:
            file_name = os.path.basename(file_info)
            
            import re        
            for col in self.df.columns:
                if re.search(r'\[ns, .*?\]$', str(self.df[col].dtype)):
                        self.df[col] = self.df[col].dt.tz_localize(None)
 
            self.df.to_excel(f'{ file_info}.xlsx')
            self.file_saver_button.text = f'已保存到 {file_name} .xlsx'
            self.app.update()

        
    def build_file_saver(self, df):
        self.df = df
        # 建立文件選擇器和文字控制項
        self.file_saver = ft.FilePicker(on_result=self.save_file_result)
        self.file_saver_button = ft.ElevatedButton(text="           另存新檔               ", icon=ft.icons.FILE_DOWNLOAD, on_click=lambda _: self.file_saver.save_file(allowed_extensions=["xlsx"]))
        # 將文件選擇器添加到頁面上
        self.app.overlay.append(self.file_saver)

        return self.file_saver_button



if __name__ == "__main__":

    def main(page: ft.Page):
        # 主函数，设置页面标题与初始化应用程序
        page.title = "量化分析應用"
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.scroll = ft.ScrollMode.ADAPTIVE
        page.window_width = 1500
        page.window_height = 650

        # 创建应用程序控件并将其添加到页面
        file_execution = FileExecution(page, None)
        page.add(file_execution.build_file_saver(pd.DataFrame()))  # 将build()方法添加到页面

    # 启动应用程序
    ft.app(main)
