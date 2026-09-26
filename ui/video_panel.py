import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class VideoPanel(ttk.LabelFrame):
    """视频显示面板"""

    def __init__(self, master, **kwargs):
        super().__init__(master, text="视频显示", **kwargs)
        self.canvas = tk.Canvas(self, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.current_image = None
        self.current_photo = None

    def update_frame(self, image):
        """更新显示帧"""
        if image is None:
            return

        # OpenCV图像已经是BGR格式，包含绘制好的检测结果
        # 直接转换为RGB用于显示
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 调整大小
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w > 10 and canvas_h > 10:
            ratio = min(canvas_w / pil_img.width, canvas_h / pil_img.height)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

        # 更新显示
        self.current_image = pil_img
        self.current_photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.current_photo,
            anchor=tk.CENTER
        )
