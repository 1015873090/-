# ================ main.py ================
import argparse
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from model import FERDataset, ResNetEmotion, FocalLoss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler


def train(csv_path, save_path, device):
    print(f"🛠️ 初始化训练环境（设备: {device}）")
    scaler = GradScaler()  # 混合精度训练

    try:
        # 数据增强配置
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 加载官方数据集
        train_set = FERDataset(csv_path, transform=train_transform, usage='Training')
        public_test_set = FERDataset(csv_path, transform=train_transform, usage='PublicTest')
        combined_train = ConcatDataset([train_set, public_test_set])

        # 验证集配置
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_set = FERDataset(csv_path, transform=val_transform, usage='PrivateTest')

        # 数据加载器
        train_loader = DataLoader(combined_train,
                                  batch_size=128,  # 增大batch_size
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True)
        val_loader = DataLoader(val_set,
                                batch_size=256,
                                shuffle=False)

        # 初始化模型
        model = ResNetEmotion().to(device)

        # 损失函数（带类别权重）
        class_weights = torch.tensor([1.5, 3.0, 1.8, 0.8, 1.2, 0.7, 1.0]).to(device)
        criterion = FocalLoss(gamma=3, alpha=class_weights)

        # 优化器配置
        optimizer = torch.optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.ca.parameters(), 'lr': 1e-4},
            {'params': model.sa.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-4)

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            steps_per_epoch=len(train_loader),
            epochs=50,
            pct_start=0.2
        )

        # 训练循环
        best_acc = 0.0
        print(f"🚀 开始训练，样本数：训练集={len(combined_train)} 验证集={len(val_set)}")
        for epoch in range(60):
            model.train()
            total_loss = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                # 混合精度训练
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                scheduler.step()

            # 验证
            val_acc = validate(model, val_loader, device)

            # 保存最佳模型
            if val_acc > best_acc:
                torch.save(model.state_dict(), save_path)
                best_acc = val_acc
                print(f"🎯 最佳准确率 {val_acc:.2%} @ Epoch {epoch + 1}")

            print(
                f"Epoch {epoch + 1}/60 | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2%} | LR: {scheduler.get_last_lr()[0]:.2e}")

    except Exception as e:
        print(f"❌ 训练出错: {str(e)}")


def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet表情识别训练')
    parser.add_argument('--data', default='data/fer2013/fer2013.csv')
    parser.add_argument('--model', default='models/resnet_emotion.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args.data, args.model, device)