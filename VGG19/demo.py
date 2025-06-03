# ================ 修复版 demo.py ================
import cv2
import torch
import os
import numpy as np
from model import EMOTIONS, VGG19Emotion
import torchvision.transforms as transforms


class EmotionDetector:
    def __init__(self, model_path, device='cuda'):
        # 添加模型路径检查
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        # 安全加载模型
        self.model = VGG19Emotion().to(device)
        self.model.load_state_dict(
            torch.load(model_path,
                       map_location=device,
                       weights_only=True)  # 修复点
        )
        self.model.eval()

        # 预处理配置
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 人脸检测器初始化检查
        cascade_path = 'data/haarcascade_files/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"级联分类器文件 {cascade_path} 不存在")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.device = device

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (224, 224))

            # 修复通道转换方式
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)  # 正确转换方式

            tensor_img = self.transform(face_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_img)
                _, pred = torch.max(outputs, 1)
                emotion = EMOTIONS[pred.item()]

            # 绘制结果
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame


if __name__ == '__main__':
    # 添加OpenCV版本检查
    print("OpenCV版本:", cv2.__version__)

    detector = EmotionDetector('models/vgg19_emotion.pth')

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