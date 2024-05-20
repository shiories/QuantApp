import flet as ft
import pandas as pd

class DataPage:
    
    def __init__(self, page, df=None, df_name=None):
        #self.group_preprocessor = GroupPreprocessor()
        self.page = page
        self.df = df
        if df_name:
            self.df = pd.read_excel(f"{df_name}.xlsx")
        elif df is not None:
            self.df = df
        else:
            raise ValueError("You must provide either a DataFrame or a DataFrame name.")
        #self.df = self.group_preprocessor.data_clean(df=self.df,clean_na=False)
        self.df = self.df.round(4)
        
        self.df.reset_index(inplace=True)

        if "index" in self.df.columns:
            self.df = self.df.drop(columns=["index"])
        if "level_0" in self.df.columns:
            self.df = self.df.drop(columns=["level_0"])
        if "level_1" in self.df.columns:
            self.df = self.df.drop(columns=["level_1"])
        
        
    def headers(self):
        return [ft.DataColumn(ft.Text(header)) for header in self.df.columns]


    def rows(self):
        rows = []
        for index, row in self.df.iterrows():
            cells = [ft.DataCell(ft.Text(str(row[header]))) for header in self.df.columns]
            rows.append(ft.DataRow(cells=cells))
        return rows


    def build(self):
        datatable = ft.DataTable(expand=True, columns=self.headers(), rows=self.rows())
        
        return ft.ResponsiveRow([ft.Row(adaptive=True, alignment=ft.CrossAxisAlignment.END,
                           controls= [datatable], )
        ])



if __name__=="__main__":

    def main(page=ft.Page):
        # 創建 DataPage 的實例並將其添加到頁面中
        df_page_1 = DataPage(page, df_name="voting_results")
        #df_page_2 = DataPage(page, df=pd.read_excel("your_dataframe_name.xlsx"))
        
        #page.add(df_page_2.build())
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.scroll = ft.ScrollMode.ADAPTIVE
        page.window_width = 1500
        page.window_height = 650
        page.adaptive = True
        page.add(df_page_1.build())

    # 啟動應用程序
    ft.app(target=main)
