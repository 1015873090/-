import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import EMOTIONS, VGG19Emotion, FERDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class FER2013Evaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        self.model = VGG19Emotion().to(self.device)
        try:
            load_args = {'map_location': self.device}
            if 'weights_only' in torch.load.__code__.co_varnames:
                load_args['weights_only'] = True
            self.model.load_state_dict(torch.load(model_path, **load_args))
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def evaluate(self, csv_path):
        val_set = FERDataset(
            csv_path=csv_path,
            usage='PrivateTest',
            transform=self.transform
        )
        print(f"✅ 验证集样本数: {len(val_set)}")

        if len(val_set) == 0:
            raise ValueError("验证集为空")

        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("\n=== FER2013验证集分类报告 ===")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=EMOTIONS,
            labels=range(len(EMOTIONS)),
            zero_division=0,
            digits=4
        ))

        # 修改后的混淆矩阵输出
        print("\n=== 混淆矩阵 ===")
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(EMOTIONS)))
        cm_df = pd.DataFrame(
            cm,  # 直接使用原始计数
            index=EMOTIONS,
            columns=EMOTIONS
        )
        print(cm_df)


if __name__ == '__main__':
    model_path = 'models/vgg19_emotion.pth'
    data_path = 'data/fer2013/fer2013.csv'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集 {data_path} 不存在")

    evaluator = FER2013Evaluator(model_path)
    evaluator.evaluate(data_path)