# from sklearnex import patch_sklearn
# patch_sklearn()

EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']

import cv2
import os
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Parallel, delayed
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast,
    ShiftScaleRotate, ToFloat
)
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from joblib import parallel_backend

# LBP多尺度参数配置
LBP_SCALES = [
    {'radius': 1, 'n_points': 8},   # 小尺度捕捉细节
    {'radius': 2, 'n_points': 16},  # 中尺度
    {'radius': 3, 'n_points': 24}   # 大尺度
]

# HOG参数配置
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True  # 伽马校正
}

# 修改数据增强配置（避免破坏LBP纹理）
AUGMENTATIONS = Compose([
    HorizontalFlip(p=0.3),
    RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.3
    )
])

def extract_lbp_features(image):
    """多尺度 LBP 特征提取"""
    scales = [
        {'radius': 1, 'n_points': 8},  # 小尺度
        {'radius': 2, 'n_points': 16},  # 中尺度
        {'radius': 3, 'n_points': 24}  # 大尺度
    ]
    hist = []
    for params in scales:
        lbp = local_binary_pattern(
            image,
            P=params['n_points'],
            R=params['radius'],
            method='uniform'
        )
        scale_hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, params['n_points'] + 3),
            density=True
        )
        hist.extend(scale_hist)

    # 添加全局特征（如灰度直方图）
    global_hist, _ = np.histogram(image.ravel(), bins=16, density=True)
    hist.extend(global_hist)

    return np.array(hist)

def extract_hog_features(data):
    """从像素数据中提取HOG特征"""
    if isinstance(data, (str, bytes)):
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='ignore')
        pixel_list = list(map(int, data.split()))
        if len(pixel_list) != 48 * 48:
            raise ValueError(f"像素数量错误：{len(pixel_list)}（应=2304）")
        image = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)
    elif isinstance(data, np.ndarray):
        image = data.reshape(48, 48) if data.size == 2304 else data
    else:
        raise ValueError("不支持的数据类型")

    try:
        augmented = image
    except Exception as e:
        print(f"⚠️ 数据增强失败：{str(e)}")
        augmented = image

    # 提取HOG特征
    features = hog(
        augmented,
        orientations=HOG_PARAMS['orientations'],
        pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
        cells_per_block=HOG_PARAMS['cells_per_block'],
        block_norm=HOG_PARAMS['block_norm']
    )
    return features

def extract_hybrid_features(image):
    """混合特征提取（HOG + LBP）"""
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    return np.concatenate([hog_feat, lbp_feat])

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

    return extract_hybrid_features(image)

def load_data(data_path):
    """加载数据并提取特征"""
    print("⏳ 加载数据并提取特征...")
    df = pd.read_csv(data_path)

    # 数据验证
    invalid_samples = []
    for idx, pixels in enumerate(df['pixels']):
        if len(pixels.split()) != 2304:
            invalid_samples.append(idx)

    # 特征提取
    X = Parallel(n_jobs=-1, verbose=10)(
        delayed(extract_features)(pixels)
        for pixels in df['pixels']
    )

    X = np.array(X)
    y = df['emotion'].values
    print(f"\n✅ 特征提取完成，特征矩阵形状：{X.shape}")
    return X, y

def train_model(data_path, save_dir):
    try:
        X, y = load_data(data_path)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 使用 LinearSVC 作为基础模型
        base_model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            dual=False,  # 当 n_samples > n_features 时设为 False
            max_iter=5000,
            random_state=42
        )

        # 使用 CalibratedClassifierCV 支持概率输出
        model = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', CalibratedClassifierCV(base_model, method='sigmoid', cv=5))
        ])

        # 启用多线程训练
        print("🚀 开始训练...")
        with parallel_backend('threading', n_jobs=-1):  # 使用多线程
            model.fit(X_train, y_train)

        # 评估模型
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        print(f"训练集分数：{train_score:.4f}, 验证集分数：{val_score:.4f}")

        # 保存模型
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(model, f'{save_dir}/hog_lbp_svm_model.pkl')
        print(f"💾 模型已保存至：{save_dir}/hog_lbp_svm_model.pkl")

    except Exception as e:
        print(f"❌ 训练失败：{str(e)}")
        raise

def predict(features):
    """预测情绪概率"""
    model = joblib.load('models/hog_lbp_svm_model.pkl')
    return model.predict_proba([features])[0]

def valid_model(model_path, valid_data_dir):
    """验证模型"""
    try:
        model = joblib.load(f'{model_path}/hog_lbp_svm_model.pkl')
        y_true, y_pred = [], []

        for emotion_idx, emotion in enumerate(EMOTIONS):
            emotion_dir = os.path.join(valid_data_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue

            for fname in os.listdir(emotion_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(emotion_dir, fname)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        logging.warning(f"无法读取图像：{img_path}")
                        continue

                    try:
                        features = extract_hybrid_features(image)
                        pred = model.predict([features])[0]
                        y_true.append(emotion_idx)
                        y_pred.append(pred)
                    except Exception as e:
                        logging.error(f"处理图像出错：{img_path}", exc_info=True)

        # 生成分类报告
        from sklearn.metrics import classification_report
        print("\n分类报告：")
        print(classification_report(
            y_true, y_pred,
            target_names=EMOTIONS,
            digits=4
        ))

    except Exception as e:
        logging.error("验证过程出错", exc_info=True)
        raise