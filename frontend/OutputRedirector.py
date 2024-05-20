

class OutputRedirector:
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, text):
        # 將文本寫入輸出文字方塊
        self.textbox.value += text    # 追加文本並換行
        self.textbox.update()  # 更新文字方塊
