#用fer2013的验证集来评估
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from model import EMOTIONS, extract_hybrid_features
from joblib import Parallel, delayed


def load_fer2013_data(csv_path):
    """
    加载FER2013验证集数据
    数据格式：
    emotion | pixels | Usage
    0 | 70 80 45 ... | PrivateTest
    """
    df = pd.read_csv(csv_path)

    # 筛选验证集数据
    val_df = df[df['Usage'] == 'PrivateTest']

    X, y = [], []

    # 并行处理像素数据
    def process_pixels(pixels_str):
        try:
            pixels = list(map(int, pixels_str.split()))
            if len(pixels) != 48 * 48:
                return None
            return np.array(pixels, dtype=np.uint8).reshape(48, 48)
        except:
            return None

    # 使用多核加速处理
    images = Parallel(n_jobs=-1)(
        delayed(process_pixels)(pixels)
        for pixels in val_df['pixels']
    )

    # 过滤无效数据并收集标签
    valid_indices = [i for i, img in enumerate(images) if img is not None]
    X = [images[i] for i in valid_indices]
    y = val_df['emotion'].iloc[valid_indices].values

    print(f"成功加载验证集样本：{len(X)}/{len(val_df)}")
    return np.array(X), np.array(y)


def evaluate_on_fer2013(model_path, data_path='./data/fer2013/fer2013.csv'):
    """在FER2013验证集上评估模型"""
    # 加载模型
    model = joblib.load(model_path)

    # 加载验证数据
    X, y_true = load_fer2013_data(data_path)
    print(f"✅ 验证集样本数量：{len(X)}")

    # 批量特征提取
    X_features = Parallel(n_jobs=-1, verbose=1)(
        delayed(extract_hybrid_features)(img)
        for img in X
    )
    X_features = np.array(X_features)

    # 维度验证
    if X_features.shape[1] != model.n_features_in_:
        raise ValueError(
            f"特征维度不匹配！模型需要 {model.n_features_in_} 维，"
            f"实际提取到 {X_features.shape[1]} 维"
        )

    # 执行预测
    y_pred = model.predict(X_features)
    probas = model.predict_proba(X_features)

    # 输出分类报告
    print("\n=== 分类报告（PrivateTest） ===")
    print(classification_report(y_true, y_pred,
                                target_names=EMOTIONS,
                                digits=4))



    # 输出原始计数的混淆矩阵
    print("\n=== 混淆矩阵 ===")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)
    print(cm_df)


if __name__ == "__main__":
    # 配置文件路径
    model_path = "./models/lbp_svm_model.pkl"
    data_path = "./data/fer2013/fer2013.csv"

    # 执行评估
    evaluate_on_fer2013(model_path, data_path)