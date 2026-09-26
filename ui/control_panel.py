import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path


class ControlPanel(ttk.LabelFrame):
    """控制面板"""

    def __init__(self, master, start_cb, stop_cb, **kwargs):
        super().__init__(master, text="控制面板", **kwargs)
        self.start_callback = start_cb
        self.stop_callback = stop_cb
        self.video_path = None
        self._setup_ui()

    def _setup_ui(self):
        # 视频源选择
        self.source_var = tk.StringVar(value="camera")
        ttk.Radiobutton(self, text="480p", variable=self.source_var,
                        value="camera", command=self._update_ui_state).pack(anchor=tk.W)
        ttk.Radiobutton(self, text="1080p", variable=self.source_var,
                        value="file", command=self._update_ui_state).pack(anchor=tk.W)

        # 文件选择
        self.file_button = ttk.Button(self, text="选择视频文件",
                                      command=self._select_file, state=tk.DISABLED)
        self.file_button.pack(fill=tk.X, padx=5, pady=2)
        self.file_label = ttk.Label(self, text="未选择文件")
        self.file_label.pack(fill=tk.X, padx=5)

        # 控制按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.start_btn = ttk.Button(btn_frame, text="开始检测", command=self._on_start)
        self.start_btn.pack(side=tk.LEFT, expand=True)
        self.stop_btn = ttk.Button(btn_frame, text="停止检测", command=self._on_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, expand=True)

        # 性能信息
        self.fps_label = ttk.Label(self, text="FPS: -")
        self.fps_label.pack(anchor=tk.W)

        # 检测信息显示
        self.detection_frame = ttk.LabelFrame(self, text="检测信息")
        self.detection_frame.pack(fill=tk.X, padx=5, pady=5)

        self.detection_label = ttk.Label(self.detection_frame, text="检测对象: 0")
        self.detection_label.pack(anchor=tk.W)

        self.detection_details = tk.Text(self.detection_frame, height=10, width=30)
        self.detection_details.pack(fill=tk.BOTH, expand=True)

        # 行为检测信息显示
        self.behavior_frame = ttk.LabelFrame(self, text="行为状态")
        self.behavior_frame.pack(fill=tk.X, padx=5, pady=5)

        self.behavior_status = ttk.Label(self.behavior_frame, text="状态: 检测中")
        self.behavior_status.pack(anchor=tk.W)

        self.behavior_details = tk.Text(self.behavior_frame, height=5, width=30)
        self.behavior_details.pack(fill=tk.BOTH, expand=True)

    def _update_ui_state(self):
        """更新UI状态"""
        if self.source_var.get() == "file":
            self.file_button.config(state=tk.NORMAL)
            self.start_btn.config(state=tk.NORMAL if self.video_path else tk.DISABLED)
        else:
            self.file_button.config(state=tk.DISABLED)
            self.start_btn.config(state=tk.NORMAL)

    def _select_file(self):
        """选择视频文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            self.file_label.config(text=Path(file_path).name)
            self._update_ui_state()

    def _on_start(self):
        """开始检测回调"""
        source_type = self.source_var.get()  # "camera" 或 "file"
        source = 0 if source_type == "camera" else self.video_path

        # 同时传递源类型和源
        self.start_callback(source_type, source)

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def _on_stop(self):
        """停止检测回调"""
        self.stop_callback()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def update_fps(self, fps):
        """更新FPS显示"""
        self.fps_label.config(text=f"FPS: {fps:.1f}")

    def update_detections(self, detections):
        """更新检测信息显示"""
        self.detection_label.config(text=f"检测对象: {len(detections)}")

        # 清空并更新详细信息
        self.detection_details.delete(1.0, tk.END)
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = map(int, det['xyxy'])
            self.detection_details.insert(tk.END,
                                          f"{i}. {det['label']}\n位置: ({x1},{y1})-({x2},{y2})\n\n")

    def update_behavior_info(self, results):
        """更新行为检测信息"""
        state_analysis = results.get("state_analysis", {})

        # 状态颜色映射
        state_colors = {
            "正常驾驶": "green",
            "注意力分散警告": "orange",
            "疲劳驾驶警告": "red",
            "手机使用警告": "purple"
        }

        status_text = state_analysis.get("description", "状态: 检测中")
        status_color = state_colors.get(status_text, "black")

        self.behavior_status.config(
            text=status_text,
            foreground=status_color
        )

        self.behavior_details.delete(1.0, tk.END)

        # 添加详细指标
        metrics = state_analysis.get("metrics", {})
        self.behavior_details.insert(tk.END, "详细指标:\n")
        self.behavior_details.insert(tk.END, f"闭眼时长: {metrics.get('eye_close_duration', 0):.1f}s\n")
        self.behavior_details.insert(tk.END, f"哈欠次数: {metrics.get('yawn_count', 0)}\n")
        self.behavior_details.insert(tk.END, f"环顾时长: {metrics.get('look_around_duration', 0):.1f}s\n")
        self.behavior_details.insert(tk.END, f"手机使用: {metrics.get('phone_use_duration', 0):.1f}s\n")

        # 添加建议
        self.behavior_details.insert(tk.END, "\n建议:\n")
        if "疲劳" in status_text:
            self.behavior_details.insert(tk.END, "建议停车休息\n")
        elif "手机" in status_text:
            self.behavior_details.insert(tk.END, "请勿使用手机\n")
        elif "分散" in status_text:
            self.behavior_details.insert(tk.END, "请集中注意力\n")
        else:
            self.behavior_details.insert(tk.END, "保持良好驾驶状态\n")
