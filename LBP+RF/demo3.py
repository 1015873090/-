# 用FER2013的私有测试集评估
import os
import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from model import EMOTIONS, extract_features  # 使用统一特征提取接口


def load_fer2013_PrivateTest(data_path):
    """
    加载FER2013测试集
    数据格式：
    emotion | pixels | Usage
    其中Usage包含"PrivateTest"标识
    """
    df = pd.read_csv(data_path)
    private_test = df[df['Usage'] == 'PrivateTest']

    X, y = [], []

    for idx, row in private_test.iterrows():
        try:
            # 使用统一的特征提取方法
            features = extract_features(row['pixels'])
            X.append(features)
            y.append(int(row['emotion']))
        except Exception as e:
            print(f"跳过无效样本（行{idx}）: {str(e)}")
            continue

    return np.array(X), np.array(y)


def evaluate_on_fer2013(model_path, fer2013_path='./data/fer2013/fer2013.csv'):
    """在FER2013私有测试集上评估"""
    # 加载模型
    model = joblib.load(model_path)

    # 加载数据
    X_features, y_true = load_fer2013_PrivateTest(fer2013_path)
    print(f"✅ 成功加载 {len(X_features)} 个测试样本")

    # 预测
    y_pred = model.predict(X_features)
    probas = model.predict_proba(X_features)

    # 分类报告
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTIONS,
        output_dict=True,
        digits=4
    )

    # 置信区间计算
    conf_intervals = {}
    for idx, emotion in enumerate(EMOTIONS):
        class_probs = probas[:, idx]
        if len(class_probs) == 0:
            conf_intervals[emotion] = (0, (0, 0))
            continue

        mean = np.mean(class_probs)
        sem = stats.sem(class_probs)
        ci = stats.t.interval(0.95, len(class_probs) - 1, loc=mean, scale=sem)
        conf_intervals[emotion] = (mean, ci)

    # 打印结果
    print("\n=== 分类报告（PrivateTest）===")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4))


    # 混淆矩阵
    print("\n=== 混淆矩阵 ===")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)
    print(cm_df)


# 使用示例
if __name__ == "__main__":
    model_path = "./models/lbp_forest_model.pkl"
    fer2013_path = "./data/fer2013/fer2013.csv"  # 确保路径正确
    evaluate_on_fer2013(model_path, fer2013_path)