class AppConfig:
    """应用程序配置"""

    def __init__(self):
        # 目标检测配置
        self.weights = 'outputs/exp5/weights/best.pt'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = 'cpu'
        self.classes = None
        self.augment = False
        self.agnostic_nms = False
        self.crop_x = 600
        self.crop_width = 1320

        # 行为检测配置
        self.EYE_AR_THRESH = 0.18  # 眼睛纵横比阈值
        self.MOUTH_AR_THRESH = 0.6  # 嘴巴纵横比阈值

        # 新增加权融合配置
        self.fusion_config = {
            'weights': {
                'phone': 0.4,
                'gaze': 0.3,
                'behavior': 0.3
            },
            'thresholds': {
                'distraction': 0.6,
                'fatigue': 0.3
            }
        }

        # 疲劳检测配置
        self.fatigue_config = {
            'eye_close_threshold': 2.0,
            'yawn_frequency_threshold': 3,
            'perclos_threshold': 0.2
        }

        # 分神检测配置
        self.distraction_config = {
            'look_around_threshold': 3.0,
            'phone_use_threshold': 2.0
        }

        # 报警配置
        self.alarm_config = {
            'sound_enabled': True,
            'sound_path': 'alarm.wav',
            'popup_enabled': True
        }

        # 视线估计配置
        self.gaze_config = self._get_gaze_config()

    def _get_gaze_config(self):
        """获取视线估计配置"""
        from yacs.config import CfgNode as CN
        config = CN()

        # 基础配置
        config.mode = 'MPIIFaceGaze'
        config.device = self.device

        # 模型配置
        config.model = CN()
        config.model.name = 'resnet_simple'
        config.model.backbone = CN()
        config.model.backbone.name = 'resnet_simple'
        config.model.backbone.pretrained = 'resnet18'
        config.model.backbone.resnet_block = 'basic'
        config.model.backbone.resnet_layers = [2, 2, 2]

        # 视线估计器配置
        config.gaze_estimator = CN()
        config.gaze_estimator.checkpoint = 'data/models/mpiifacegaze/resnet_simple_14/checkpoint_0015.pth'
        config.gaze_estimator.camera_params = 'data/calib/sample_params.yaml'
        config.gaze_estimator.normalized_camera_params = 'data/calib/normalized_camera_params_face.yaml'
        config.gaze_estimator.normalized_camera_distance = 1.0

        # 人脸检测器配置
        config.face_detector = CN()
        config.face_detector.mode = 'dlib'
        config.face_detector.dlib = CN()
        config.face_detector.dlib.model = 'data/dlib/shape_predictor_68_face_landmarks.dat'

        # 图像变换配置
        config.transform = CN()
        config.transform.mpiifacegaze_face_size = 224
        config.transform.mpiifacegaze_gray = False

        # 演示配置
        config.demo = CN()
        config.demo.use_camera = True
        config.demo.display_on_screen = False
        config.demo.wait_time = 1
        config.demo.video_path = 'test1.mp4'
        config.demo.output_dir = 'videos_test'
        config.demo.output_file_extension = 'mp4'
        config.demo.head_pose_axis_length = 0.05
        config.demo.gaze_visualization_length = 0.05
        config.demo.show_bbox = True
        config.demo.show_head_pose = True
        config.demo.show_landmarks = True
        config.demo.show_normalized_image = False
        config.demo.show_template_model = False

        return config
