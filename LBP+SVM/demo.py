# demo.py
import cv2
import numpy as np
import joblib
from model import EMOTIONS, extract_features

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def format_image(image):
    """人脸检测和特征提取"""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(
        image, scaleFactor=1.3, minNeighbors=5
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    try:
        face_img = image[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))
        features = extract_features(face_img)
        return features, (x, y, w, h)
    except Exception as e:
        print(f"特征提取失败：{str(e)}")
        return None, None


def draw_emotion_bars(frame, proba, emotions):
    """绘制情绪概率条"""
    for i, (emotion, prob) in enumerate(zip(emotions, proba)):
        cv2.putText(frame, emotion, (10, i * 20 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (130, i * 20 + 10),
                      (130 + int(prob * 100), (i + 1) * 20 + 4), (255, 0, 0), -1)


def overlay_emoji(frame, emoji_face, position):
    """叠加Emoji表情"""
    x1, y1, x2, y2 = position
    if emoji_face is None:
        return
    try:
        emoji_face = cv2.resize(emoji_face, (x2 - x1, y2 - y1))
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + \
                                     frame[y1:y2, x1:x2, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
    except Exception as e:
        print(f"Emoji叠加失败: {str(e)}")


def demo(model_path, show_box=True):
    model = joblib.load(f'{model_path}/lbp_svm_model.pkl')
    feelings_faces = []
    for emotion in EMOTIONS:
        emoji_path = f'./data/emojis/{emotion}.png'
        emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
        if emoji is None:
            print(f"⚠️ 无法加载Emoji: {emoji_path}")
            feelings_faces.append(None)
        else:
            feelings_faces.append(emoji)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_face, face_coor = format_image(frame)
        if detected_face is not None:
            try:
                result = model.predict_proba([detected_face])[0]
                pred_emotion = EMOTIONS[np.argmax(result)]

                if show_box and face_coor is not None:
                    x, y, w, h = face_coor
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                draw_emotion_bars(frame, result, EMOTIONS)
                overlay_emoji(frame, feelings_faces[np.argmax(result)], (10, 200, 130, 320))

            except Exception as e:
                print(f"预测失败: {str(e)}")

        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()