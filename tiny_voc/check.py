import os
import torch
import pandas as pd
from train import ModelTrainer


def analyze_model(model_name, device='cuda'):
    """分析单个模型的性能"""
    results = {
        'model_name': model_name,
        'val_accuracy': 0
    }

    try:
        # 加载模型
        trainer = ModelTrainer(model_name)
        model = trainer.get_model().to(device)

        # 加载最佳模型权重
        checkpoint_path = f'checkpoints_{model_name}/{model_name}_best_model.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path,weights_only=True)
            results['val_accuracy'] = checkpoint['val_acc'] * 100


    except Exception as e:
        print(f"分析 {model_name} 时出错: {str(e)}")

    return results



if __name__ == '__main__':
    models = ['lenet', 'vgg16', 'resnet18', 'vit']
    results = []

    for model_name in models:
        results.append(analyze_model(model_name))

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 保存为CSV
    df.to_csv('model_performance.csv', index=False)

    # 创建Markdown表格
    markdown_table = df.to_markdown(index=False)
    with open('model_performance.md', 'w') as f:
        f.write("model performance\n\n")
        f.write(markdown_table)