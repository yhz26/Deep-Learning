import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
from model_resnet18 import MyResNet18
from model_lenet import MyLeNet5
from model_vgg import MyVGG16
import torchvision
from tqdm import tqdm
from model_Vit import ViT
import os


class ModelTrainer:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        if self.model_name == 'lenet':
            model = MyLeNet5()
            return model
        elif self.model_name == 'vgg16':
            model = MyVGG16()
            return model
        elif self.model_name == 'resnet18':
            model = MyResNet18()
            return model
        elif self.model_name == 'vit':
            model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                        dim=64, depth=6, num_heads=8, mlp_dim=128)
            return model
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device,save_dir,
                model_name):
    best_val_acc = 0
    for epoch in range(num_epochs):
        # 训练阶段
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
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
            _,predicted = torch.max(outputs, 1)
            predicted = predicted.to(labels.dtype)
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
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.to(labels.dtype)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

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
    num_epochs = 5
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 要训练的模型列表
    models_to_train = ['lenet', 'vgg16', 'resnet18', 'vit']

    for model_name in models_to_train:
        print(f"\n开始训练 {model_name}...")
        print(f'设备: {device}')

        # 创建保存目录
        save_dir = f'checkpoints_{model_name}'
        os.makedirs(save_dir, exist_ok=True)


        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # 创建数据集和数据加载器
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

        print(f'训练集大小: {len(trainset)}')
        print(f'验证集大小: {len(testset)}')

        # 创建模型
        trainer = ModelTrainer(model_name)
        model = trainer.get_model().to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        print('开始训练...')
        train_model(model, trainloader, testloader, criterion, optimizer,
                    num_epochs, device, save_dir, model_name)
        print(f'{model_name} 训练完成!')


if __name__ == '__main__':
    main()