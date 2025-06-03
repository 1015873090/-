import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import EMOTIONS, ResNetEmotion, FERDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class ResNetFER2013Evaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        self.model = ResNetEmotion(num_classes=7, pretrained=False).to(self.device)

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def evaluate(self, csv_path):
        try:
            full_dataset = FERDataset(csv_path, transform=self.transform)
            private_indices = full_dataset.df[full_dataset.df['Usage'] == 'PrivateTest'].index.tolist()

            from torch.utils.data import Subset
            private_set = Subset(full_dataset, private_indices)

            if len(private_set) == 0:
                raise ValueError("筛选后的PrivateTest集为空")

        except Exception as e:
            raise RuntimeError(f"数据集加载失败: {str(e)}")

        val_loader = DataLoader(private_set, batch_size=64, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 分类报告
        print("\n=== 分类报告（PrivateTest）===")
        print(classification_report(
            all_labels, all_preds,
            target_names=EMOTIONS,
            labels=range(len(EMOTIONS)),
            zero_division=0,
            digits=4
        ))

        # 修改后的混淆矩阵
        print("\n=== 混淆矩阵 ===")
        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(
            cm,
            index=EMOTIONS,
            columns=EMOTIONS
        )
        print(cm_df)


if __name__ == '__main__':
    try:
        model_path = 'models/resnet_emotion.pth'
        data_path = 'data/fer2013/fer2013.csv'

        evaluator = ResNetFER2013Evaluator(model_path)
        evaluator.evaluate(data_path)

    except Exception as e:
        print(f"评估失败: {str(e)}")
        exit(1)