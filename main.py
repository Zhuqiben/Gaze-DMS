import time
from config import AppConfig
from my_detector import Detector
from video_processor import VideoProcessor
from ui.main_window import MainWindow
import tkinter as tk

class App:
    """主应用程序"""
    def __init__(self):
        self.config = AppConfig()
        self.detector = Detector(self.config)
        self.processor = VideoProcessor(self.detector)
        self.fps_history = []

        self.root = tk.Tk()
        self.main_window = MainWindow(
            self.root,
            start_cb=self.start_detection,
            stop_cb=self.stop_detection,
            update_ui_cb=self.update_ui
        )

    def start_detection(self, source_type, source):
        """开始检测"""
        self.processor.start(source_type, source)

    def stop_detection(self):
        """停止检测"""
        self.processor.stop()

    def update_ui(self):
        """更新UI状态"""
        start_time = time.time()
        frame, detections, behavior_results = self.processor.get_result()
        if frame is not None:
            self.main_window.update_video_frame(frame)
            self.main_window.update_detection_info(detections)
            self.main_window.update_behavior_info(behavior_results)

            fps = 1 / max(time.time() - start_time, 0.001)
            self.fps_history = (self.fps_history + [fps])[-5:]
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            self.main_window.update_fps(avg_fps)

    def run(self):
        """运行"""
        self.root.mainloop()

if __name__ == "__main__":
    app = App()
    app.run()
