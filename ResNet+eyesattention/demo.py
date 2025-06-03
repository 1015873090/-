# ================ demo.py ================
import cv2
import torch
import numpy as np
from model import EMOTIONS, ResNetEmotion
import torchvision.transforms as transforms


class EmotionDetector:
    def __init__(self, model_path, device='cuda'):
        if not torch.cuda.is_available():
            device = 'cpu'

        self.model = ResNetEmotion().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.face_cascade = cv2.CascadeClassifier('data/haarcascade_files/haarcascade_frontalface_default.xml')
        self.device = device

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_roi, (224, 224))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)

            tensor_img = self.transform(face_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_img)
                prob = torch.nn.functional.softmax(outputs, dim=1)
                _, pred = torch.max(prob, 1)
                emotion = EMOTIONS[pred.item()]
                confidence = prob[0][pred.item()].item()

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.0%})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame


if __name__ == '__main__':
    detector = EmotionDetector('models/resnet_emotion.pth')

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.detect(frame)
        cv2.imshow('ResNet Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()