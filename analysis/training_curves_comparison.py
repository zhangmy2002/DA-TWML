# -*- coding: utf-8 -*-
"""
training_curves_comparison.py - 四种方法训练过程对比可视化

生成内容：
1. 双轴折线图（上图为测试准确率，下图为元损失）
2. 标注各方法的最佳轮次
3. 置信区间带状区域
4. 四种方法横向对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_training_data():
    """加载四种方法的训练日志数据"""
    methods = {
        'ProtoNet_equal_training': {
            'file': r'runs_comparison\01_ProtoNet_equal_training\log_all_1777382038.csv',
            'color': '#1f77b4',
            'label': 'ProtoNet（等权）',
            'best_epoch': 25
        },
        'Random_TopK': {
            'file': r'runs_comparison\02_Random_TopK\log_random_topk_1777409427.csv',
            'color': '#ff7f0e',
            'label': 'Random Top-k',
            'best_epoch': 26
        },
        'Loss_only_TSML': {
            'file': r'runs_comparison\03_Loss_only_TSML\log_loss_topk_1777440025.csv',
            'color': '#2ca02c',
            'label': 'Loss-only TSML',
            'best_epoch': 25
        },
        'DA_TWML_full': {
            'file': r'runs_comparison\04_DA_TWML_full\log_full_1777467064.csv',
            'color': '#d62728',
            'label': 'DA-TWML full',
            'best_epoch': 25
        }
    }
    
    base_dir = Path(r'D:\南京大学岩石教学薄片显微图像数据集')
    data = {}
    
    for method, info in methods.items():
        file_path = base_dir / info['file']
        df = pd.read_csv(file_path)
        data[method] = {
            'df': df,
            'color': info['color'],
            'label': info['label'],
            'best_epoch': info['best_epoch']
        }
    
    return data

def plot_training_curves(data, output_dir):
    """绘制训练曲线对比图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    epochs = np.arange(1, 31)
    
    # 上图：测试准确率
    for method, info in data.items():
        df = info['df']
        color = info['color']
        label = info['label']
        best_epoch = info['best_epoch']
        
        # 测试准确率
        test_acc = df['test_acc'].values * 100  # 转换为百分比
        test_ci95 = df['test_ci95'].values * 100
        
        # 绘制折线
        ax1.plot(epochs, test_acc, color=color, label=label, linewidth=2)
        
        # 绘制置信区间带
        ax1.fill_between(epochs, 
                         test_acc - test_ci95, 
                         test_acc + test_ci95, 
                         color=color, alpha=0.1)
        
        # 标注最佳轮次
        best_acc = test_acc[best_epoch - 1]
        ax1.scatter(best_epoch, best_acc, color=color, marker='*', s=100, zorder=5)
        ax1.text(best_epoch, best_acc + 1.5, f'epoch {best_epoch}', 
                 ha='center', va='bottom', fontsize=9, color=color)
    
    ax1.set_ylabel('测试准确率 (%)', fontsize=12)
    ax1.set_title('四种方法测试准确率对比', fontsize=14, pad=15, loc='center')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(45, 70)
    
    # 下图：元损失
    for method, info in data.items():
        df = info['df']
        color = info['color']
        label = info['label']
        best_epoch = info['best_epoch']
        
        # 训练损失（作为元损失的近似）
        train_loss = df['train_loss'].values
        
        # 绘制折线
        ax2.plot(epochs, train_loss, color=color, label=label, linewidth=2)
        
        # 标注最佳轮次
        best_loss = train_loss[best_epoch - 1]
        ax2.scatter(best_epoch, best_loss, color=color, marker='*', s=100, zorder=5)
    
    ax2.set_xlabel('训练轮次 (epoch)', fontsize=12)
    ax2.set_ylabel('元损失 (L_meta)', fontsize=12)
    ax2.set_title('四种方法元损失收敛轨迹', fontsize=14, pad=15, loc='center')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(1.15, 1.55)
    ax2.set_xticks(np.arange(0, 31, 5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_interval_summary(data, output_dir):
    """绘制每5个epoch的间隔汇总图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    interval_epochs = [5, 10, 15, 20, 25, 30]
    bar_width = 0.2
    
    x = np.arange(len(interval_epochs))
    
    for i, (method, info) in enumerate(data.items()):
        df = info['df']
        color = info['color']
        label = info['label']
        
        # 获取特定epoch的数据（每5个epoch）
        acc_values = []
        ci_values = []
        for epoch in interval_epochs:
            idx = epoch - 1
            acc_values.append(df['test_acc'].iloc[idx] * 100)
            ci_values.append(df['test_ci95'].iloc[idx] * 100)
        
        ax.bar(x + i * bar_width, acc_values, width=bar_width, 
               label=label, color=color, alpha=0.7, yerr=ci_values, capsize=4)
    
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels([f'epoch {e}' for e in interval_epochs])
    ax.set_ylabel('测试准确率 (%)', fontsize=12)
    ax.set_xlabel('训练轮次', fontsize=12)
    ax.set_title('四种方法测试准确率间隔对比', fontsize=14, pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(45, 70)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'interval_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(data, output_dir):
    """生成训练曲线汇总表格"""
    summary = []
    
    for method, info in data.items():
        df = info['df']
        best_epoch = info['best_epoch']
        
        best_row = df.iloc[best_epoch - 1]
        
        summary.append({
            '方法': info['label'],
            '最佳轮次': best_epoch,
            '最佳测试准确率(%)': f"{best_row['test_acc'] * 100:.2f}",
            '95%置信区间(%)': f"±{best_row['test_ci95'] * 100:.2f}",
            '最终测试准确率(%)': f"{df['test_acc'].iloc[-1] * 100:.2f}",
            '最终训练损失': f"{df['train_loss'].iloc[-1]:.4f}",
            '验证准确率峰值(%)': f"{df['val_acc'].max() * 100:.2f}"
        })
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(output_dir / 'training_summary.csv', index=False, encoding='utf-8-sig')
    
    # 打印表格
    print("\n表4-6 四种方法训练过程汇总")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)
    
    return df_summary

def main():
    """主函数"""
    base_dir = r'D:\南京大学岩石教学薄片显微图像数据集'
    output_dir = Path(base_dir) / 'figures_final' / 'comparison'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("生成四种方法训练过程对比图")
    print("=" * 70)
    
    # 加载训练数据
    data = load_training_data()
    print("✓ 加载四种方法的训练日志")
    
    # 绘制训练曲线对比图
    plot_training_curves(data, output_dir)
    print("✓ 绘制训练曲线对比图（测试准确率+元损失）")
    
    # 绘制间隔汇总图
    plot_interval_summary(data, output_dir)
    print("✓ 绘制间隔汇总图")
    
    # 生成汇总表格
    generate_summary_table(data, output_dir)
    print("✓ 生成训练汇总表格")
    
    print("=" * 70)
    print(f"所有图表已保存到: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

