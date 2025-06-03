import argparse
import torch
import torch.nn as nn
from model import FERDataset, VGG19Emotion
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms

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

def train(csv_path, save_path, device):
    print(f"🛠️ 初始化训练环境（设备: {device}）")

    try:
        # 加载训练集和验证集
        train_transform = FERDataset.default_transform()
        train_set = FERDataset(csv_path, transform=train_transform, usage='Training')
        public_test_set = FERDataset(csv_path, transform=train_transform, usage='PublicTest')
        combined_train = ConcatDataset([train_set, public_test_set])

        # 单独加载验证集
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_set = FERDataset(csv_path, transform=val_transform, usage='PrivateTest')

        train_loader = DataLoader(combined_train, batch_size=32, shuffle=True,
                                num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        # 模型配置保持不变...
        model = VGG19Emotion().to(device)
        criterion = FocalLoss(gamma=2, alpha=torch.tensor([1.0, 1.5, 1.5, 0.8, 1.2, 0.8, 1.0]).to(device))
        optimizer = torch.optim.Adam([
            {'params': model.features.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-4},
            {'params': model.eye_attention.parameters(), 'lr': 1e-4}
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        # 训练循环保持不变...
        best_acc = 0.0
        for epoch in range(70):
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

            val_acc = validate(model, val_loader, device)
            scheduler.step(total_loss)

            if val_acc > best_acc:
                torch.save(model.state_dict(), save_path)
                best_acc = val_acc
                print(f"💾 保存最佳模型，准确率: {val_acc:.2%}")

            print(f"Epoch {epoch + 1}/70 | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2%}")

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
    parser = argparse.ArgumentParser(description='PyTorch表情识别训练')
    parser.add_argument('--data', default='data/fer2013/fer2013.csv', help='数据集路径')
    parser.add_argument('--model', default='models/vgg19_emotion.pth', help='模型保存路径')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args.data, args.model, device)