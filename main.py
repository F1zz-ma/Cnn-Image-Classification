import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils.data as dataloader
from PIL import Image
from torchvision.datasets import CIFAR10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3main.py)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((32,32)),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010))])
trainset = CIFAR10(root="./data", train=True, transform=transform, download=True)
testset = CIFAR10(root="./data", train=False, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, trailoader, criterion,
          optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(trainloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

torch.no_grad()
def evaluate(model, testloader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

def main():

    print("当前运行设备:", device)
    epochs = 30
    best_accuracy = 0.0

    for epoch in range(epochs):

        train_loss, train_acc = train(
            model,
            trainloader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            testloader,
            criterion,
            device
        )

        optimizer.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc:.2f}%"
        )

        # 保存测试准确率最高的模型
        if test_acc > best_accuracy:

            best_accuracy = test_acc

            torch.save(
                model.state_dict(),
                "cifar10_cnn_best.pth"
            )

            print("已保存当前最佳模型！")

    print("训练结束！")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
