# demo2.py
import os
import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from model import EMOTIONS, extract_features


def load_fer2013_PrivateTest(data_path):
    df = pd.read_csv(data_path)
    private_test = df[df['Usage'] == 'PrivateTest']

    X, y = [], []
    for idx, row in private_test.iterrows():
        try:
            features = extract_features(row['pixels'])
            if features is not None:
                X.append(features)
                y.append(int(row['emotion']))
        except Exception as e:
            print(f"跳过无效样本（行{idx}）: {str(e)}")

    return np.array(X), np.array(y)


def evaluate_on_fer2013(model_path, fer2013_path='./data/fer2013/fer2013.csv'):
    model = joblib.load(model_path)
    X_features, y_true = load_fer2013_PrivateTest(fer2013_path)

    if len(X_features) == 0:
        print("❌ 无有效测试数据")
        return

    print(f"✅ 成功加载 {len(X_features)} 个测试样本")

    y_pred = model.predict(X_features)
    probas = model.predict_proba(X_features)

    report = classification_report(
        y_true, y_pred,
        target_names=EMOTIONS,
        output_dict=True,
        digits=4
    )

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

    print("\n=== 分类报告（PrivateTest）===")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4))


    print("\n=== 混淆矩阵 ===")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)
    print(cm_df)


if __name__ == "__main__":
    model_path = "./models/lbp_svm_model.pkl"
    fer2013_path = "./data/fer2013/fer2013.csv"
    evaluate_on_fer2013(model_path, fer2013_path)