import cv2
import numpy as np
import time
from collections import deque
from tracker import Tracker
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio


class DriverState:
    NORMAL = 0
    FATIGUE = 1
    DISTRACTION = 2
    PHONE_USE = 3


class FatigueDistractionAnalyzer:
    """疲劳和分神分析器（使用加权融合算法）"""

    def __init__(self, config):
        self.config = config
        self.state_history = deque(maxlen=30 * 60)
        self.current_state = DriverState.NORMAL
        self.last_state_change_time = time.time()

        # 从配置加载权重和阈值
        self.weights = config.fusion_config['weights']
        self.distraction_threshold = config.fusion_config['thresholds']['distraction']
        self.fatigue_threshold = config.fusion_config['thresholds']['fatigue']

        # 状态持续时间统计
        self.eye_close_duration = 0
        self.yawn_count = 0
        self.last_yawn_time = 0
        self.look_around_duration = 0
        self.phone_use_duration = 0

    def update(self, behavior_results, detections):
        """加权融合算法更新状态"""
        # 获取各模块检测结果
        phone_detected = any(d['label'].startswith('phone') for d in detections)
        gaze_deviation = behavior_results['abnormal_types']['looking_around']['active']
        abnormal_behavior = (
            behavior_results['abnormal_types']['eyes_closed']['active'] or
            behavior_results['abnormal_types']['mouth_open']['active']
        )

        # 计算综合得分
        score = (
            self.weights['phone'] * float(phone_detected) +
            self.weights['gaze'] * float(gaze_deviation) +
            self.weights['behavior'] * float(abnormal_behavior)
        )

        # 状态判断
        new_state = DriverState.NORMAL
        if phone_detected and score >= self.distraction_threshold:
            new_state = DriverState.PHONE_USE
        elif score >= self.distraction_threshold:
            new_state = DriverState.DISTRACTION
        elif abnormal_behavior and score >= self.fatigue_threshold:
            new_state = DriverState.FATIGUE

        # 更新状态历史
        self.state_history.append(new_state)
        self.current_state = new_state

        # 更新持续时间统计
        self._update_duration_stats(behavior_results, phone_detected)

        return self._get_state_description()

    def _update_duration_stats(self, behavior_results, phone_detected):
        """更新各状态持续时间"""
        # 眼睛状态
        if behavior_results['abnormal_types']['eyes_closed']['active']:
            self.eye_close_duration += 1 / 5  # 假设5FPS
        else:
            self.eye_close_duration = 0

        # 嘴巴状态
        if behavior_results['abnormal_types']['mouth_open']['active']:
            if time.time() - self.last_yawn_time > 4:
                self.yawn_count += 1
                self.last_yawn_time = time.time()

        # 环顾状态
        if behavior_results['abnormal_types']['looking_around']['active']:
            self.look_around_duration += 1 / 5
        else:
            self.look_around_duration = 0

        # 手机使用
        if phone_detected:
            self.phone_use_duration += 1 / 5
        else:
            self.phone_use_duration = 0

    def _get_state_description(self):
        """生成状态描述"""
        state_descriptions = {
            DriverState.NORMAL: "正常驾驶",
            DriverState.FATIGUE: "疲劳驾驶警告",
            DriverState.DISTRACTION: "注意力分散警告",
            DriverState.PHONE_USE: "手机使用警告"
        }

        return {
            "state": self.current_state,
            "description": state_descriptions.get(self.current_state, "未知状态"),
            "metrics": {
                "eye_close_duration": self.eye_close_duration,
                "yawn_count": self.yawn_count,
                "look_around_duration": self.look_around_duration,
                "phone_use_duration": self.phone_use_duration,
                "score": self._calculate_current_score()
            }
        }

    def _calculate_current_score(self):
        """计算当前综合得分"""
        return min(1.0, (
            0.4 * min(1.0, self.phone_use_duration / 2.0) +
            0.3 * min(1.0, self.look_around_duration / 3.0) +
            0.3 * min(1.0, self.eye_close_duration / 2.0)
        ))


class BehaviorDetector:
    """行为检测器封装"""

    def __init__(self, config):
        self.config = config
        self.tracker = Tracker(width=640, height=480, model_type=3, detection_threshold=0.6)
        self.source_type = 'camera'
        self.analyzer = FatigueDistractionAnalyzer(config)

        # 特征点索引
        self.lStart, self.lEnd = 42, 48
        self.rStart, self.rEnd = 36, 42
        self.mStart, self.mEnd = 49, 66

        # 标准头部姿态
        self.standard_pose = {
            'camera': [-175, 10, 90],
            'file': [180, 20, 90]
        }

    def set_source_type(self, source_type):
        """设置视频源类型"""
        self.source_type = source_type

    def detect_behavior(self, frame, detections=[]):
        """执行行为检测"""
        # 图像尺寸标准化
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))

        # 初始化结果
        results = {
            "status": "normal",
            "abnormal_types": {
                "eyes_closed": {"active": False, "duration": 0.0},
                "mouth_open": {"active": False, "duration": 0.0},
                "looking_around": {"active": False, "duration": 0.0}
            },
            "current_pose": self.standard_pose[self.source_type],
            "state_analysis": {}
        }

        try:
            # 人脸检测
            faces = self.tracker.predict(frame)
            if len(faces) > 0:
                face = max(faces, key=lambda f: f.bbox[3])

                # 眼睛状态检测
                left_eye = face.lms[self.lStart:self.lEnd]
                right_eye = face.lms[self.rStart:self.rEnd]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
                if ear < self.config.EYE_AR_THRESH:
                    results["abnormal_types"]["eyes_closed"]["active"] = True

                # 嘴巴状态检测
                mar = mouth_aspect_ratio(face.lms)
                if mar > self.config.MOUTH_AR_THRESH:
                    results["abnormal_types"]["mouth_open"]["active"] = True

                # 头部姿态检测
                if self._is_looking_around(face.euler, self.standard_pose[self.source_type]):
                    results["abnormal_types"]["looking_around"]["active"] = True

            # 综合状态分析
            results["state_analysis"] = self.analyzer.update(results, detections)

            # 更新综合状态
            if results["state_analysis"]["state"] != DriverState.NORMAL:
                results["status"] = "abnormal"

        except Exception as e:
            print(f"Behavior detection error: {str(e)}")

        return results

    def _angle_diff(self, a, b):
        """计算角度差"""
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    def _is_looking_around(self, euler_angles, standard_pose):
        """判断是否环顾四周"""
        return (
            self._angle_diff(standard_pose[0], euler_angles[0]) >= 45 or
            self._angle_diff(standard_pose[1], euler_angles[1]) >= 45 or
            self._angle_diff(standard_pose[2], euler_angles[2]) >= 45
        )
