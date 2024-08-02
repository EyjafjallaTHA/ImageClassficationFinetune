# 微调VGG16模型分类皮肤病变

本项目微调了预训练的 VGG16 模型，用于将皮肤病变图像分类为两类：甲下出血和甲母质痣。

## 数据集

数据集包含两类图像：
•  甲下出血

•  甲母质痣


在微调中，将数据集数据集分为训练集、验证集和测试集。

## 模型

使用 VGG16 模型作为基础模型。冻结卷积层，并将全连接层修改为输出两类。

## 训练

模型使用以下设置进行训练：
•  优化器：SGD，学习率 0.001，动量 0.9

•  损失函数：CrossEntropyLoss

•  使用早停机制控制训练停止，耐心值为 3 个 epoch


## 结果

模型在训练期间达到了以下验证准确率：
•  最佳验证准确率：**X%**

模型在测试集上的准确率为：
•  测试集准确率：**Y%**


## 分析

模型在区分甲下出血和甲母质痣方面显示出良好的结果。然而，可以通过以下方式进一步改进：
•  增加数据集大小

•  试验不同的架构

•  微调超参数


## 使用方法

1. 安装所需的包：
```bash
pip install -r requirements.txt

运行训练脚本：

python train.py

在其他测试集上评估模型：

python evaluate.py

文件结构
•  train.py：训练模型的脚本

•  evaluate.py：在测试集上评估模型的脚本

•  requirements.txt：所需包列表

•  README.md：项目文档

致谢
本项目使用了 torchvision 库中的 VGG16 模型。