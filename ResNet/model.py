# ================ model.py ================
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.models import resnet50
import pandas as pd
import numpy as np
import cv2

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc[2].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * (avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attention)


# ================ model.py ================
class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        original_model = resnet50(weights='IMAGENET1K_V2' if pretrained else None)

        # 特征提取层（layer1-3）
        self.backbone = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3
        )

        # 注意力层（layer4 + 注意力机制）
        self.layer4 = original_model.layer4  # 关键修改：直接使用layer4
        self.ca = ChannelAttention(1024)  # 通道注意力
        self.sa = SpatialAttention()  # 空间注意力

        # 池化层和分类头
        self.avgpool = original_model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.6),
            nn.Linear(2048, num_classes)
        )

        # 梯度控制
        self._freeze_layers()

    def _freeze_layers(self):
        # 冻结backbone参数（layer1-3）
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻layer4和注意力
        for param in self.layer4.parameters():
            param.requires_grad = True
        for param in self.ca.parameters():
            param.requires_grad = True
        for param in self.sa.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)  # [BS, 1024, 14, 14]
        x = self.ca(x)  # 通道注意力
        x = self.layer4(x)  # [BS, 2048, 7, 7]
        x = self.sa(x)  # 空间注意力
        x = self.avgpool(x)  # [BS, 2048, 1, 1]
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# FERDataset类和FocalLoss保持原样...


# ----------------- 数据集类（保持不变） -----------------
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        return loss.mean()