# ================ demo.py ================
import cv2
import torch
import os
import numpy as np
from model import EMOTIONS, VGG19Emotion
import torchvision.transforms as transforms

class EnhancedEmotionDetector:
    def __init__(self, model_path, device='cuda'):
        # 模型加载增强
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        self.device = device
        self.model = VGG19Emotion().to(device)

        # 安全加载模型权重
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
        self.model.eval()

        # 预处理配置：保持和训练时一致
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 多级联检测器：加载人脸和眼部检测分类器
        self.face_cascade = cv2.CascadeClassifier('data/haarcascade_files/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('data/haarcascade_files/haarcascade_eye.xml')

    def _enhance_eyes(self, face_img):
        """
        使用 YUV 色彩空间对眼部区域进行亮度增强，避免彩色失真
        步骤：
          1. 将输入的 BGR 图像转换为 YUV（亮度 + 色度）空间。
          2. 使用灰度图检测眼部区域。
          3. 对眼部区域对应的 Y 通道（亮度）进行直方图均衡化。
          4. 将处理后的 YUV 图像转换回 BGR 格式返回。
        """
        # 转换到 YUV 空间
        yuv = cv2.cvtColor(face_img, cv2.COLOR_BGR2YUV)
        # 为了检测眼部区域，转换为灰度图
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
        for (ex, ey, ew, eh) in eyes:
            # 仅对 Y 通道（亮度通道）进行直方图均衡化
            y_channel = yuv[ey:ey+eh, ex:ex+ew, 0]
            y_channel_eq = cv2.equalizeHist(y_channel)
            yuv[ey:ey+eh, ex:ex+ew, 0] = y_channel_eq
        # 转换回 BGR 色彩空间
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return enhanced

    def detect(self, frame):
        # 检测人脸区域
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # 对眼部区域进行亮度增强（避免色差）
            face_img = self._enhance_eyes(face_img)

            # 对增强后的区域进行预处理
            processed_img = cv2.resize(face_img, (224, 224))
            tensor_img = self.transform(processed_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_img)
                _, pred = torch.max(outputs, 1)
                emotion = EMOTIONS[pred.item()]

            # 绘制矩形框和类别标签
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

if __name__ == '__main__':
    print("OpenCV版本:", cv2.__version__)

    try:
        detector = EnhancedEmotionDetector('models/vgg19_emotion.pth')
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detector.detect(frame)
            cv2.imshow('Emotion Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
