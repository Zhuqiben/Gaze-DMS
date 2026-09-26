import tkinter as tk
from tkinter import ttk
from .control_panel import ControlPanel
from .video_panel import VideoPanel


class MainWindow:
    """主窗口"""

    def __init__(self, root, start_cb, stop_cb, update_ui_cb):
        self.root = root
        self.root.title("驾驶员监控系统")
        self.root.geometry("1200x800")
        self.alarm_active = False
        self.alarm_sound = None  # 可以加载报警音效

        # 创建UI组件
        self._setup_ui(start_cb, stop_cb)
        self.update_ui_callback = update_ui_cb
        self._start_main_loop()

    def _setup_ui(self, start_cb, stop_cb):
        """初始化用户界面"""
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # 视频显示区域
        self.video_panel = VideoPanel(main_pane, width=900)
        main_pane.add(self.video_panel, weight=3)

        # 控制面板
        self.control_panel = ControlPanel(
            main_pane,
            start_cb=start_cb,
            stop_cb=stop_cb,
            width=300
        )
        main_pane.add(self.control_panel, weight=1)

    def _start_main_loop(self):
        """启动主循环"""
        self.update_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_ui(self):
        """更新UI状态"""
        if self.update_ui_callback:
            self.update_ui_callback()
        self.root.after(30, self.update_ui)

    def on_close(self):
        """关闭窗口"""
        if hasattr(self, 'stop_callback'):
            self.stop_callback()
        self.root.destroy()

    def update_video_frame(self, image):
        """更新视频帧"""
        self.video_panel.update_frame(image)

    def update_fps(self, fps):
        """更新FPS显示"""
        self.control_panel.update_fps(fps)

    def update_detection_info(self, detections):
        """更新检测信息"""
        self.control_panel.update_detections(detections)

    def update_behavior_info(self, behavior_results):
        """更新行为检测信息"""
        self.control_panel.update_behavior_info(behavior_results)

        # 检查是否需要触发报警
        state_analysis = behavior_results.get("state_analysis", {})
        if state_analysis.get("state", 0) in [2, 3]:  # DANGER或PHONE_USE状态
            self.trigger_alarm(state_analysis["description"])
        else:
            self.stop_alarm()

    def trigger_alarm(self, reason):
        """触发报警"""
        if not self.alarm_active:
            self.alarm_active = True
            # 播放声音报警
            if self.alarm_sound:
                self.alarm_sound.play()

            # 显示弹窗警告
            self.show_alert_popup(reason)

    def stop_alarm(self):
        """停止报警"""
        if self.alarm_active:
            self.alarm_active = False
            if self.alarm_sound:
                self.alarm_sound.stop()

    def show_alert_popup(self, reason):
        """显示警告弹窗"""
        popup = tk.Toplevel(self.root)
        popup.title("驾驶警告")

        if "疲劳" in reason:
            color = "red"
        elif "手机" in reason:
            color = "purple"
        else:
            color = "orange"

        label = tk.Label(popup, text=f"警告: {reason}", font=("Arial", 16), fg=color)
        label.pack(padx=20, pady=20)

        # 5秒后自动关闭
        popup.after(5000, popup.destroy)
