import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from model_vit import ViT


class ModelTrainer:
    def __init__(self, model_name, num_classes=5):
        self.model_name = model_name
        self.num_classes = num_classes

    def get_model(self):
        if self.model_name == 'lenet':
            return MyLeNet(num_classes=self.num_classes)
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier[-1] = nn.Linear(4096, self.num_classes)
            return model
        elif self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, self.num_classes)
            return model
        elif self.model_name == 'vit':
            model = ViT(image_size=224, patch_size=7, num_classes=5, channels=3,
                        dim=64, depth=6, num_heads=8, mlp_dim=128)
            return model
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")


class MyLeNet(nn.Module):
    def __init__(self, num_classes=5):
        super(MyLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 61 * 61, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 使用之前相同的TinySegDataset类
class TinySegDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'JPEGImages')

        # 类别映射
        self.classes = ['person', 'cat', 'plane', 'car', 'bird']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.seg_to_class = {1: 'person', 2: 'cat', 3: 'plane', 4: 'car', 5: 'bird'}

        # 读取数据集划分
        split_file = os.path.join(root_dir, 'ImageSets', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = f.read().splitlines()

        # 读取分割标注中的类别信息
        self.labels = {}
        annotations_dir = os.path.join(root_dir, 'Annotations')

        for img_id in self.image_ids:
            ann_path = os.path.join(annotations_dir, f'{img_id}.png')
            if os.path.exists(ann_path):
                seg_img = Image.open(ann_path)
                seg_array = np.array(seg_img)
                unique_ids = np.unique(seg_array)
                classes = [self.seg_to_class[id] for id in unique_ids if id in self.seg_to_class]
                if classes:
                    self.labels[img_id] = classes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.zeros(len(self.classes))
        if img_id in self.labels:
            for cls in self.labels[img_id]:
                label[self.class_to_idx[cls]] = 1

        return image, label



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir,
                model_name):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.numel()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total


        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_accuracy,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, f'{model_name}_best_model.pth'))
            # logger.info(f'保存最佳模型，验证准确率: {val_accuracy:.4f}')
            print(f'保存最佳模型，验证准确率: {val_accuracy:.4f}')

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy,
        }, os.path.join(save_dir, f'{model_name}_latest.pth'))


def main():
    # 设置参数
    data_root = 'data/TinySeg'
    batch_size = 16
    num_epochs = 15
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 要训练的模型列表
    models_to_train = ['lenet', 'vgg16', 'resnet18', 'vit']

    for model_name in models_to_train:
        print(f"\n开始训练 {model_name}...")

        # 创建保存目录
        save_dir = f'checkpoints_{model_name}'
        os.makedirs(save_dir, exist_ok=True)

        print(f'设备: {device}')

        # 数据预处理
        if model_name == 'vit':
            input_size = 224  # ViT的标准输入大小
        else:
            input_size = 256

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 创建数据集和数据加载器
        train_dataset = TinySegDataset(data_root, 'train', transform)
        val_dataset = TinySegDataset(data_root, 'val', transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f'训练集大小: {len(train_dataset)}')
        print(f'验证集大小: {len(val_dataset)}')

        # 创建模型
        trainer = ModelTrainer(model_name)
        model = trainer.get_model().to(device)

        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        print('开始训练...')
        train_model(model, train_loader, val_loader, criterion, optimizer,
                    num_epochs, device, save_dir, model_name)
        print(f'{model_name} 训练完成!')


if __name__ == '__main__':
    main()