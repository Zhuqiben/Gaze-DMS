import cv2
import time
import queue
import threading

class VideoProcessor:
    """视频处理流水线"""

    def __init__(self, detector):
        self.detector = detector
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.capture_thread = None
        self.processing_thread = None
        self.running = False
        self.source_type = 'camera'

    def start(self, source_type, source):
        """启动视频处理"""
        self.stop()
        self.running = True
        self.source_type = source_type
        self.detector.set_source_type(source_type)

        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            args=(source,),
            daemon=True
        )
        self.processing_thread = threading.Thread(
            target=self._process_frames,
            daemon=True
        )
        self.capture_thread.start()
        self.processing_thread.start()

    def stop(self):
        """停止视频处理"""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()

    def _capture_frames(self, source):
        """捕获视频帧"""
        cap = cv2.VideoCapture(source)

        # 根据模式设置参数
        if self.source_type == 'file':
            # 文件模式保持原始分辨率
            pass
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 摄像头模式设为标清

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if self.source_type == 'file':  # 文件模式循环播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # 文件模式不进行resize，保持原始分辨率
            if self.frame_queue.qsize() < 2:
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)
        cap.release()

    def _process_frames(self):
        """处理视频帧"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                try:
                    result = self.detector.detect_and_draw(frame)
                    if self.result_queue.qsize() < 2:
                        self.result_queue.put(result)
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    continue
            else:
                time.sleep(0.01)

    def get_result(self):
        """获取最新结果"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None, [], {}
