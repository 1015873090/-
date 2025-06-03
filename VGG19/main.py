# ================ main.py ================
import argparse
import torch
import torch.nn as nn
from model import FERDataset, VGG19Emotion
from torch.utils.data import DataLoader


def train(csv_path, save_path, device):
    # 初始化
    print(f"🛠️ 初始化训练环境（设备: {device}）")

    try:
        # 仅加载训练集
        train_set = FERDataset(csv_path, usage='Training')
        # 加载官方验证集
        val_set = FERDataset(csv_path, usage='PublicTest')

        train_loader = DataLoader(
            train_set,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_set,
            batch_size=128,
            shuffle=False
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return

    # 模型配置
    try:
        model = VGG19Emotion().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=3
        )
    except Exception as e:
        print(f"❌ 模型初始化失败: {str(e)}")
        return

    # 训练循环
    best_acc = 0.0
    print("🚀 开始训练...")
    for epoch in range(50):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 在官方验证集上评估
        val_acc = validate(model, val_loader, device)
        scheduler.step(val_acc)

        if val_acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc
            print(f"💾 保存最佳模型，准确率: {val_acc:.2%}")

        print(f"Epoch {epoch + 1}/50 | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2%}")


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
    parser = argparse.ArgumentParser(description='VGG19表情识别训练')
    parser.add_argument('--data', default='data/fer2013/fer2013.csv', help='数据集路径')
    parser.add_argument('--model', default='models/vgg19_emotion.pth', help='模型保存路径')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args.data, args.model, device)