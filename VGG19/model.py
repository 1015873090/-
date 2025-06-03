# ================ model.py ================
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.models import vgg19
import pandas as pd
import numpy as np
import cv2

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
IMG_SIZE = 224


class FERDataset(Dataset):
    """FER2013数据集加载器（严格划分训练/验证集）"""

    def __init__(self, csv_path, usage='Training', transform=None):
        """
        :param usage: 'Training'或'PublicTest'（原验证集）
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        self.df = pd.read_csv(csv_path)
        # 严格按官方划分选择数据
        self.df = self.df[self.df['Usage'] == usage].reset_index(drop=True)

        # 数据清洗
        self.df['pixels'] = self.df['pixels'].str.replace(r'[^0-9 ]', '', regex=True)
        valid_mask = self.df['pixels'].apply(
            lambda x: len(x.split()) == 2304 and x.replace(' ', '').isdigit()
        )
        self.df = self.df[valid_mask]

        if len(self.df) == 0:
            raise ValueError(f"{usage}数据集为空")

        self.transform = transform or self.default_transform()
        print(f"✅ 成功加载 {len(self.df)} 个{usage}样本")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = self.df.iloc[idx]['pixels']
        label = int(self.df.iloc[idx]['emotion'])

        # 图像预处理流程
        img = np.array(list(map(int, pixels.split()))).reshape(48, 48).astype(np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 转换为RGB三通道

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class VGG19Emotion(nn.Module):
    """VGG19表情识别模型"""

    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        base_model = vgg19(weights='IMAGENET1K_V1' if pretrained else None)

        # 冻结特征提取层
        for param in base_model.parameters():
            param.requires_grad = False

        # 修改分类器
        base_model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        self.model = base_model

    def forward(self, x):
        return self.model(x)