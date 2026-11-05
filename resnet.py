import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils.data as dataloader
from PIL import Image
from torchvision.datasets import CIFAR10

#残差模块
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        y=x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(y)
        out = self.relu(out)
        return out

 #ResNet18
class ResNet18(nn.Module):
    def make_layer(self, out_channels, blocks, stride):
        layers = []

        # 第一个 BasicBlock 可能改变尺寸
        layers.append(
            BasicBlock(
                self.in_channels,
                out_channels,
                stride
            )
        )

        self.in_channels = out_channels

        # 后面的 Block 不改变尺寸
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)
    def __init__(self,num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(32, blocks=2, stride=1)
        self.layer2 = self.make_layer(64, blocks=2, stride=2)
        self.layer3 = self.make_layer(128, blocks=2, stride=2)
        self.layer4 = self.make_layer(256, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out=self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224)),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010))])
trainset = CIFAR10(root="./data", train=True, transform=transform, download=True)
testset = CIFAR10(root="./data", train=False, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, trainloader, criterion,
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

@torch.no_grad()
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
                "cifar10_resnet_best.pth"
            )

            print("已保存当前最佳模型！")

    print("训练结束！")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()

