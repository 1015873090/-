# model.py
import cv2
import os
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog
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

# HOG参数配置
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True
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
    return extract_hog_features(image)


def extract_hog_features(data):
    """HOG特征提取实现"""
    if isinstance(data, (str, bytes)):
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='ignore')
        pixel_list = list(map(int, data.split()))
        image = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            image = data.reshape(48, 48)
        else:
            image = data
    else:
        raise ValueError("不支持的数据类型")

    try:
        features = hog(
            image,
            orientations=HOG_PARAMS['orientations'],
            pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
            cells_per_block=HOG_PARAMS['cells_per_block'],
            block_norm=HOG_PARAMS['block_norm']
        )
        return features
    except Exception as e:
        print(f"HOG特征提取失败：{str(e)}")
        return None


def load_data(data_path):
    """加载指定数据子集（Training + PublicTest）"""
    print("⏳ 加载数据并提取特征...")
    df = pd.read_csv(data_path)
    selected_data = df[df['Usage'].isin(['Training', 'PublicTest'])]

    X = Parallel(n_jobs=-1, verbose=10)(
        delayed(extract_features)(pixels)
        for pixels in selected_data['pixels']
    )

    X = np.array(X)
    y = selected_data['emotion'].values
    print(f"\n✅ 特征提取完成，共加载 {len(X)} 个样本")
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
        joblib.dump(model, f'{save_dir}/hog_svm_model.pkl')
        print(f"💾 模型已保存至：{save_dir}/hog_svm_model.pkl")

    except Exception as e:
        print(f"❌ 训练失败：{str(e)}")
        raise


def predict(features):
    """预测情绪概率"""
    model = joblib.load('models/hog_svm_model.pkl')
    return model.predict_proba([features])[0]


def valid_model(model_path, valid_dir):
    """验证模型"""
    model = joblib.load(f'{model_path}/hog_svm_model.pkl')
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
            result = model.predict_proba([features])
            pred_emotion = EMOTIONS[np.argmax(result)]
            print(f"{fname}: {pred_emotion}")