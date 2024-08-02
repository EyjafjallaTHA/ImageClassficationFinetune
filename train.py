import random
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

random.seed(42)

# 加载VGGNet16模型
vgg_model = models.vgg16(pretrained=True)

# 冻结卷积层参数
for param in vgg_model.features.parameters():
    param.requires_grad = False

# 具体任务数据集类别为2，据此调整全连接层
num_classes = 2
vgg_model.classifier[6] = nn.Linear(vgg_model.classifier[6].in_features, num_classes)

train_folder_path1 = './黑甲皮肤镜照片和列表/甲下出血皮肤镜图像'
train_folder_path2 = './黑甲皮肤镜照片和列表/甲母质痣皮肤镜图像'
eval_folder_path1 = './黑甲皮肤镜照片和列表-北大-验证/甲下出血'
eval_folder_path2 = './黑甲皮肤镜照片和列表-北大-验证/甲母痣'


# 按照设定比例随机分配验证集，测试集
def split_file_paths(folder_path, val_ratio, test_ratio):

    assert val_ratio + test_ratio == 1

    # 存储文件路径的列表
    val_paths = []
    test_paths = []

    # 遍历子文件夹
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            # 获取所有文件路径
            file_paths = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if
                          os.path.isfile(os.path.join(subdir_path, f)) and f.lower().endswith('.jpg')]
            random.shuffle(file_paths)  # 随机打乱文件顺序

            # 计算分割点
            val_end = int(len(file_paths) * val_ratio)

            # 分配文件路径
            val_paths.extend(file_paths[:val_end])
            test_paths.extend(file_paths[val_end:])

    return val_paths, test_paths


val_paths1, test_paths1 = split_file_paths(eval_folder_path1, 0.5, 0.5)
val_paths2, test_paths2 = split_file_paths(eval_folder_path2, 0.5, 0.5)

train_paths1 = []
train_paths2 = []

for subdir in os.listdir(train_folder_path1):
    subdir_path = os.path.join(train_folder_path1, subdir)
    if os.path.isdir(subdir_path):
        # 获取所有文件路径
        file_paths = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if
                      os.path.isfile(os.path.join(subdir_path, f))]
        train_paths1.extend(file_paths)
for subdir in os.listdir(train_folder_path2):
    subdir_path = os.path.join(train_folder_path2, subdir)
    if os.path.isdir(subdir_path):
        # 获取所有文件路径
        file_paths = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if
                      os.path.isfile(os.path.join(subdir_path, f))]
        train_paths2.extend(file_paths)

# 预处理输入图像
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, path1, path2, transform=None):
        self.img_labels = []
        for file_path in path1:
            self.img_labels.append((file_path, 0))
        for file_path in path2:
            self.img_labels.append((file_path, 1))
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# 实例化训练集，验证集，测试集
train_dataset = CustomDataset(path1=train_paths1, path2=train_paths2, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32)
val_dataset = CustomDataset(path1=val_paths1, path2=val_paths2, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32)
test_dataset = CustomDataset(path1=test_paths1, path2=test_paths2, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg_model.classifier.parameters(), lr=0.001, momentum=0.9)

patience = 3
# 如果验证准确率在连续三个epoch后没有改善，则停止训练
best_val_accuracy = 0
last_val_accuracy = 0
patience_counter = 0

num_epochs = 50
for epoch in range(num_epochs):
    vgg_model.train()
    # 微调模型
    for images, labels in train_loader:
        outputs = vgg_model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1} finish training.')

    vgg_model.eval()
    correct = 0
    total = 0
    # 使用验证集测试模型
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = vgg_model(images)
            _, predicted = torch.max(outputs.data, 1)
            for index, value in enumerate(predicted):
                total += 1
                if value.item() == labels[index]:
                    correct += 1

    val_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy}%')
    if val_accuracy >= best_val_accuracy:
        best_val_accuracy = val_accuracy
        patience_counter = 0
        # 保存最佳模型
        torch.save(vgg_model.state_dict(), 'vgg_model_best.pth')
    elif val_accuracy < last_val_accuracy:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break  # 如果准确率连续几个epoch没有改善，停止训练
    else:
        patience_counter = 0
    last_val_accuracy = val_accuracy

# 使用测试集验证微调后模型效果
test_model = models.vgg16(pretrained=True)
test_model.classifier[6] = nn.Linear(vgg_model.classifier[6].in_features, 2)
test_model.load_state_dict(torch.load('vgg_model_best.pth'))


test_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = test_model(images)
        _, predicted = torch.max(outputs.data, 1)
        for index, value in enumerate(predicted):
            total += 1
            if value.item() == labels[index]:
                correct += 1
    val_accuracy = 100 * correct / total
    print(f'Test set: Validation Accuracy: {val_accuracy}%')
