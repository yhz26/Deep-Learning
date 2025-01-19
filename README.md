# Deep-Learning

## mnist multi-class classification

| model_name   |   val_accuracy |
|:-------------|---------------:|
| lenet        |          97.96 |
| vgg16        |          99.06 |
| resnet18     |          99.07 |
| vit          |          97.38 |

## tiny_voc multi—label classification

| model_name   |   val_accuracy |
|:-------------|---------------:|
| lenet        |        86.0256 |
| vgg16        |        85.8974 |
| resnet18     |        93.5897 |
| vit          |        88.0769 |

tiny_voc需要将数据集下载解压后以data/tiny_voc存储

## 说明
每个文件夹中的model_xx.py中是对应的模型

调用train.py会生成对应的模型checkpoint

调用check.py汇总各个模型的最好的验证集准确度
