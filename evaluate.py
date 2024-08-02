import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

folder_path1 = './黑甲皮肤镜照片和列表/甲下出血皮肤镜图像'  # 甲下出血图像文件位置
folder_path2 = './黑甲皮肤镜照片和列表/甲母质痣皮肤镜图像'  # 甲母质痣图像文件位置


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.img_labels = []
        for root, dirs, files in os.walk(folder_path1):
            for file in files:
                file_path = os.path.join(root, file)
                self.img_labels.append((file_path, 0))
        for root, dirs, files in os.walk(folder_path2):
            for file in files:
                file_path = os.path.join(root, file)
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


vgg_model = models.vgg16(pretrained=True)
vgg_model.classifier[6] = nn.Linear(4096, 2)
vgg_model.load_state_dict(torch.load('vgg_model_best.pth'))

dataset = CustomDataset(transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

vgg_model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loader:
        outputs = vgg_model(images)
        _, predicted = torch.max(outputs.data, 1)
        for index, value in enumerate(predicted):
            print(f'predicted:{value.item()}')
            print(f'labels:{labels[index]}')
            total += 1
            if value.item() == labels[index]:
                correct += 1
    print(f'correct: {correct}, total: {total}')
