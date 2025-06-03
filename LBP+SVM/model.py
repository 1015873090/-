# model.py
import cv2
import os
import joblib
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Parallel, delayed
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast,
    ShiftScaleRotate, ToFloat
)

EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']

# LBP参数配置
LBP_PARAMS = {
    'radius': 3,
    'n_points': 24,
    'method': 'uniform'
}

# 数据增强配置
AUGMENTATIONS = Compose([
    HorizontalFlip(p=0.3),
    RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.3
    )
])


def extract_features(data):
    """统一特征提取入口"""
    if isinstance(data, (str, bytes)):
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='ignore')
        pixel_list = list(map(int, data.split()))
        image = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)
    elif isinstance(data, np.ndarray):
        image = data.reshape(48, 48) if data.size == 2304 else data
    else:
        raise ValueError("不支持的数据类型")
    return extract_lbp_features(image)


def extract_lbp_features(image):
    """LBP特征提取实现"""
    try:
        lbp_image = local_binary_pattern(
            image,
            P=LBP_PARAMS['n_points'],
            R=LBP_PARAMS['radius'],
            method=LBP_PARAMS['method']
        )

        # 分块直方图统计
        hist_features = []
        cell_size = 16  # 48/3=16
        for i in range(0, 48, cell_size):
            for j in range(0, 48, cell_size):
                cell = lbp_image[i:i + cell_size, j:j + cell_size]
                hist, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256))
                hist_features.extend(hist)

        return np.array(hist_features, dtype=np.float32)
    except Exception as e:
        print(f"LBP特征提取失败：{str(e)}")
        return None


def load_data(data_path):
    """加载训练数据"""
    print("⏳ 加载数据并提取特征...")
    df = pd.read_csv(data_path)
    training_data = df[df['Usage'] == 'Training']

    X = Parallel(n_jobs=-1, verbose=10)(
        delayed(extract_features)(pixels)
        for pixels in training_data['pixels']
    )

    X = np.array(X)
    y = training_data['emotion'].values
    print(f"\n✅ LBP特征提取完成，共加载 {len(X)} 个训练样本")
    return X, y


def train_model(data_path, save_dir):
    try:
        X, y = load_data(data_path)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42,
                verbose=1
            ))
        ])

        print("🚀 开始训练SVM模型...")
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)

        public_test = pd.read_csv(data_path).query("Usage == 'PublicTest'")
        X_public = np.array([extract_features(p) for p in public_test['pixels']])
        y_public = public_test['emotion'].values
        public_score = model.score(X_public, y_public)

        print(f"\n训练集准确率：{train_score:.4f}")
        print(f"验证集准确率：{val_score:.4f}")
        print(f"公开测试集准确率：{public_score:.4f}")

        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(model, f'{save_dir}/lbp_svm_model.pkl')
        print(f"💾 模型已保存至：{save_dir}/lbp_svm_model.pkl")

    except Exception as e:
        print(f"❌ 训练失败：{str(e)}")
        raise


def predict(features):
    """预测情绪概率"""
    model = joblib.load('models/lbp_svm_model.pkl')
    return model.predict_proba([features])[0]


def valid_model(model_path, valid_dir):
    """验证模型"""
    model = joblib.load(f'{model_path}/lbp_svm_model.pkl')
    if not os.path.exists(valid_dir):
        print(f"❌ 验证目录不存在：{valid_dir}")
        return

    for fname in os.listdir(valid_dir):
        if fname.endswith('.jpg'):
            image_path = os.path.join(valid_dir, fname)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"⚠️ 无法读取图像：{image_path}")
                continue

            features = extract_features(image)
            if features is None:
                print(f"⚠️ 特征提取失败：{image_path}")
                continue

            result = model.predict_proba([features])
            pred_emotion = EMOTIONS[np.argmax(result)]
            print(f"{fname}: {pred_emotion}")