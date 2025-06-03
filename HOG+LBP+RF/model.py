#from sklearnex import patch_sklearn
#patch_sklearn()

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
from sklearn.ensemble import RandomForestClassifier

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

"""
16,8,3:训练集分数：0.5583, 验证集分数：0.4136
16,8,2:训练集分数：0.5278, 验证集分数：0.4081
16,6,3:训练集分数：0.6985, 验证集分数：0.4147
12,6,3:训练集分数：0.6498, 验证集分数：0.4235
12,4,4:训练集分数：0.9196, 验证集分数：0.4198
9,8,2:训练集分数：0.4792, 验证集分数：0.4081
9,8,2;9,16,1混合：训练集分数：0.4899, 验证集分数：0.4146(log_loss)
982:训练集分数：0.4764, 验证集分数：0.3977(hinge)
"""







# 修改数据增强配置（避免破坏LBP纹理）
AUGMENTATIONS = Compose([
    HorizontalFlip(p=0.3),
    RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.3
    )
])


# 添加多尺度 LBP 和全局特征
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


# 测试验证
test_image = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
features = extract_lbp_features(test_image)
print(f"特征维度: {features.shape}")  # 应输出 (8+2)+(16+2)+(24+2)=54

# 确保训练和测试时使用相同的特征提取函数
def extract_features(data):
    """统一特征提取入口"""
    # 解析输入数据
    if isinstance(data, (str, bytes)):
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='ignore')
        pixel_list = list(map(int, data.split()))
        image = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)
    elif isinstance(data, np.ndarray):
        image = data.reshape(48, 48) if data.size == 2304 else data
    else:
        raise ValueError("不支持的数据类型")

    # 提取多尺度 LBP 特征
    return extract_hybrid_features(image)



def extract_hog_features(data):
    """
    从像素数据中提取HOG特征
    此函数支持两种输入格式：
      - 字符串（或bytes）：空格分隔的像素值字符串，例如 CSV 文件中的内容。
      - numpy.ndarray：直接传入图像数组（1D或2D）。
    """
    # 如果输入是字符串或bytes，按空格分割解析为整数
    if isinstance(data, (str, bytes)):
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='ignore')
        pixel_list = list(map(int, data.split()))
        if len(pixel_list) != 48 * 48:
            raise ValueError(f"像素数量错误：{len(pixel_list)}（应=2304）")
        image = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)
    # 如果输入是 numpy 数组，直接使用
    elif isinstance(data, np.ndarray):
        # 如果是扁平数组，则尝试重塑为 (48,48)
        if data.ndim == 1:
            if data.size != 48 * 48:
                raise ValueError(f"数组大小错误：{data.size} (应为2304)")
            image = data.reshape(48, 48)
        else:
            image = data
    else:
        raise ValueError("不支持的数据类型")

    try:
        # 应用数据增强
        #augmented = AUGMENTATIONS(image=image)['image']
        #augmented = np.clip(augmented, 0, 255).astype(np.uint8)
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

"""
    def extract_multi_scale_hog(image):
        #多尺度HOG特征融合
        # 第一尺度：8x8单元
        hog_small = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

        # 第二尺度：16x16单元
        hog_large = hog(
            image,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            block_norm='L2-Hys'
        )

        return np.concatenate([hog_small, hog_large])

    features = extract_multi_scale_hog(image)
    return features
"""

def extract_hybrid_features(image):
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    return np.concatenate([hog_feat, lbp_feat])



def image_to_tensor(image):
    """将图像转换为 LBP 特征"""
    # 输入应为 48x48 灰度图
    if image.shape != (48, 48):
        image = cv2.resize(image, (48, 48))
    features = extract_hybrid_features(image)
    return features.reshape(1, -1)


def load_data(data_path):
    """加载数据并提取 LBP 特征"""
    print("⏳ 加载数据并提取 LBP 特征...")
    df = pd.read_csv(data_path)

    # 数据验证（保持原有逻辑）
    invalid_samples = []
    for idx, pixels in enumerate(df['pixels']):
        if len(pixels.split()) != 2304:
            invalid_samples.append(idx)

    # 特征提取（修改为调用 extract_features）
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

        # 修改分类器配置
        model = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ))
        ])

        print("🚀 开始训练...")
        # 一次性拟合整个流水线
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        print(f"训练集分数：{train_score:.4f}, 验证集分数：{val_score:.4f}")

        # 保存模型
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(model, f'{save_dir}/lbp_svm_model.pkl')
        print(f"💾 模型已保存至：{save_dir}/lbp_svm_model.pkl")

        # 绘制学习曲线等（如果需要单独记录训练过程，可以考虑自定义回调）
        # 这里就不再使用循环增量训练了

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

            # 提取HOG特征并预测
            features = image_to_tensor(image)
            result = model.predict_proba(features)
            pred_emotion = EMOTIONS[np.argmax(result)]
            print(f"{fname}: {pred_emotion}")