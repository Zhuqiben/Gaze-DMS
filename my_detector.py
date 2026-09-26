import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from gaze_estimation.gaze_estimator import GazeEstimator
from gaze_estimation.gaze_estimator.common import Visualizer
from behavior_detector import BehaviorDetector


class Detector:
    """检测器封装"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and config.device != 'cpu' else 'cpu')
        self.half = self.device.type != 'cpu'
        self.model = None
        self.stride = None
        self.names = None
        self.colors = None
        self.source_type = 'camera'

        # 初始化视线估计器
        self.gaze_estimator = GazeEstimator(config.gaze_config)
        self.gaze_visualizer = Visualizer(self.gaze_estimator.camera)

        # 初始化行为检测器
        self.behavior_detector = BehaviorDetector(config)

        self._init_model()

    def _init_model(self):
        """初始化模型"""
        self.model = attempt_load(self.config.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self._warmup()

    def _warmup(self):
        """模型预热"""
        if self.device.type != 'cpu':
            img = torch.zeros((1, 3, self.config.img_size, self.config.img_size), device=self.device)
            self.model(img.half() if self.half else img.float())

    def set_source_type(self, source_type):
        """设置视频源类型"""
        self.source_type = source_type
        self.behavior_detector.set_source_type(source_type)  # 同步设置行为检测器的模式

    def _preprocess_image(self, image):
        """图像预处理通用方法"""
        if self.source_type == 'file':
            try:
                image = image[:, self.config.crop_x:self.config.crop_x + self.config.crop_width]
            except:
                pass

        img = letterbox(image, self.config.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        return img.unsqueeze(0) if img.ndimension() == 3 else img

    def detect(self, image):
        """执行检测并返回结果"""
        if image is None:
            return None, []

        im0 = image.copy()
        img = self._preprocess_image(image)
        pred = self.model(img, augment=self.config.augment)[0]
        pred = non_max_suppression(
            pred, self.config.conf_thres, self.config.iou_thres,
            classes=self.config.classes, agnostic=self.config.agnostic_nms
        )

        detections = []
        for det in pred:
            if len(det):
                if self.source_type == 'file':
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image[:,
                                                                         self.config.crop_x:self.config.crop_x + self.config.crop_width].shape).round()
                    det[:, [0, 2]] += self.config.crop_x
                else:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    detections.append({
                        'xyxy': [int(x) for x in xyxy],
                        'conf': conf.item(),
                        'cls': cls.item(),
                        'label': f'{self.names[int(cls)]} {conf:.2f}',
                        'color': self.colors[int(cls)]
                    })

        return im0, detections

    def detect_and_draw(self, image):
        """执行检测并在图像上绘制结果"""
        if image is None:
            return None, [], {}

        # 保存原始图像用于行为检测
        behavior_image = image.copy()
        if self.source_type == 'file':
            try:
                behavior_image = behavior_image[:, self.config.crop_x:self.config.crop_x + self.config.crop_width]
            except Exception as e:
                print(f"ROI裁剪失败: {str(e)}")

        # 执行目标检测
        image, detections = self.detect(image)
        if image is None:
            return None, [], {}

        # 绘制目标检测结果
        for det in detections:
            self._draw_detection(image, det)

        # 执行视线估计
        processed_frame = self._preprocess_for_gaze(image)
        undistorted = cv2.undistort(
            processed_frame,
            self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients
        )

        self.gaze_visualizer.set_image(processed_frame.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)

        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_gaze_elements(face)
            gaze_viz = self.gaze_visualizer.image

            if self.source_type == 'file':
                gaze_viz = self._restore_gaze_visualization(gaze_viz, image)

            if gaze_viz is not None and gaze_viz.shape == image.shape:
                image = cv2.addWeighted(image, 0.7, gaze_viz, 0.3, 0)

        # 执行行为检测（传递检测结果）
        behavior_results = self.behavior_detector.detect_behavior(behavior_image, detections)
        self._draw_behavior_results(image, behavior_results)

        return image, detections, behavior_results

    def _draw_behavior_results(self, image, results):
        """绘制行为检测结果"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("simhei.ttf", 20)
        except:
            font = ImageFont.load_default()

        # 绘制状态
        state_analysis = results.get("state_analysis", {})
        status_text = state_analysis.get("description", "状态: 检测中")
        color = (0, 255, 0) if "正常" in status_text else (255, 0, 0)

        draw.text((10, 10), status_text, font=font, fill=color)

        # 绘制详细指标
        metrics = state_analysis.get("metrics", {})
        y_pos = 40
        for name, value in metrics.items():
            label = {
                "eye_close_duration": "闭眼时长",
                "yawn_count": "哈欠次数",
                "look_around_duration": "环顾时长",
                "phone_use_duration": "手机使用"
            }.get(name, name)

            draw.text((10, y_pos), f"{label}: {value:.1f}s" if isinstance(value, float) else f"{label}: {value}",
                      font=font, fill=(255, 255, 255))
            y_pos += 25

        image[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _preprocess_for_gaze(self, image):
        """为视线估计预处理图像"""
        if self.source_type == 'file':
            return cv2.resize(image[:, self.config.crop_x:self.config.crop_x + self.config.crop_width], (640, 480))
        return cv2.resize(image, (640, 480))

    def _restore_gaze_visualization(self, viz_image, original_frame):
        """将视线估计结果还原到原始图像坐标"""
        try:
            restored = original_frame.copy()
            original_height = original_frame.shape[0]
            resized = cv2.resize(viz_image, (self.config.crop_width, original_height))

            # 检查裁剪区域是否有效
            if (self.config.crop_x + self.config.crop_width) > original_frame.shape[1]:
                print(
                    f"裁剪区域超出图像宽度: {self.config.crop_x + self.config.crop_width} > {original_frame.shape[1]}")
                return None

            restored[:, self.config.crop_x:self.config.crop_x + self.config.crop_width] = resized
            return restored
        except Exception as e:
            print(f"恢复视线可视化错误: {str(e)}")
            return None

    def _draw_gaze_elements(self, face):
        """绘制视线估计元素"""
        if self.config.gaze_config.demo.show_bbox:
            self.gaze_visualizer.draw_bbox(face.bbox)
        if self.config.gaze_config.demo.show_landmarks:
            self.gaze_visualizer.draw_points(face.landmarks, color=(0, 255, 255), size=1)
        if self.config.gaze_config.demo.show_head_pose:
            self.gaze_visualizer.draw_model_axes(face,
                                                 self.config.gaze_config.demo.head_pose_axis_length, lw=2)

        length = self.config.gaze_config.demo.gaze_visualization_length
        left_eye_center = face.model3d[36:42].mean(axis=0)
        right_eye_center = face.model3d[42:48].mean(axis=0)
        self.gaze_visualizer.draw_3d_line(
            left_eye_center, left_eye_center + length * face.gaze_vector)
        self.gaze_visualizer.draw_3d_line(
            right_eye_center, right_eye_center + length * face.gaze_vector)

    def _draw_detection(self, image, detection):
        """绘制单个检测结果"""
        x1, y1, x2, y2 = detection['xyxy']
        color = detection['color']
        label = detection['label']

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 10),
                      (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
