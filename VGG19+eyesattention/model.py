import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19
import pandas as pd
import numpy as np
import cv2

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
IMG_SIZE = 224


class EyeAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class VGG19Emotion(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        base_model = vgg19(weights='IMAGENET1K_V1' if pretrained else None)
        self.features = base_model.features
        self.eye_attention = EyeAttention(512)
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes))
        self.eye_roi = (slice(30, 90), slice(50, 150))

    def forward(self, x):
        x = self.features(x)
        attention_mask = self.eye_attention(x)
        x = x * (1 + 0.5 * attention_mask)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FERDataset(Dataset):
    def __init__(self, csv_path, transform=None, usage=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        self.df = pd.read_csv(csv_path, dtype={'emotion': 'int32', 'pixels': 'string', 'Usage': 'string'})

        if 'Usage' not in self.df.columns:
            raise KeyError("数据集缺少必要的'Usage'列")

        if usage is not None:
            valid_usages = ['Training', 'PublicTest', 'PrivateTest']
            if usage not in valid_usages:
                raise ValueError(f"usage参数必须是 {valid_usages} 之一")
            self.df = self.df[self.df['Usage'] == usage]

        self.df = self.df.dropna().reset_index(drop=True)
        self.df['pixels'] = self.df['pixels'].str.replace(r'[^0-9 ]', '', regex=True)
        valid_mask = self.df['pixels'].apply(
            lambda x: len(x.split()) == 2304 and x.replace(' ', '').isdigit()
        )
        self.df = self.df[valid_mask]

        if len(self.df) == 0:
            raise ValueError("清洗后数据集为空")

        self.transform = transform or self.default_transform()
        print(f"✅ 成功加载 {len(self.df)} 个{usage if usage else '所有'}样本")

        self.eye_roi = (slice(10, 40), slice(20, 60))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = self.df.iloc[idx]['pixels']
        label = self.df.iloc[idx]['emotion']

        img = np.array(list(map(int, pixels.split()))).reshape(48, 48).astype(np.uint8)
        eye_region = img[self.eye_roi]
        eye_region = cv2.equalizeHist(eye_region)
        img[self.eye_roi] = cv2.addWeighted(img[self.eye_roi], 0.7,
                                            eye_region, 0.3, 0)

        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            img = self.transform(img)

        return img.float(), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ], p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])