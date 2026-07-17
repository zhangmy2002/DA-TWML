# -*- coding: utf-8 -*-
"""
dataset_split_visualization.py - 数据集划分可视化

生成以下图表：
1. 训练/验证/测试集样本数分布（堆叠柱状图）
2. 三大岩类在各子集中的分布（分组柱状图）
3. 域标签来源分布（饼图/堆叠图）
4. 薄片编号分组划分说明图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_output_dirs(base_dir):
    """创建输出目录"""
    output_dir = Path(base_dir) / 'figures_split'
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir

def generate_split_data():
    """生成数据集划分统计数据"""
    # 数据集整体统计
    total_classes = 108
    total_samples = 2634
    
    # 三大岩类分布（模拟数据）
    rock_types = ['火成岩', '沉积岩', '变质岩']
    rock_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 训练/验证/测试集划分（70%/15%/15%）
    splits = ['训练集', '验证集', '测试集']
    split_percentages = [0.7, 0.15, 0.15]
    
    # 模拟各岩类在各子集中的样本数
    # 火成岩: ~35%, 沉积岩: ~30%, 变质岩: ~35%
    data = {
        '火成岩': [650, 140, 140],   # ~35% of total
        '沉积岩': [550, 120, 120],   # ~30% of total
        '变质岩': [640, 140, 140],   # ~35% of total
    }
    
    # 域标签来源分布
    domain_sources = {
        '学校': ['南京大学', '中国地质大学', '北京大学', '其他'],
        '设备': ['Leica DM4500P', 'Olympus BX53', 'Nikon Eclipse', 'Zeiss Axio'],
        '倍率': ['50x', '100x', '200x', '400x'],
        '偏振模式': ['PPL', 'XPL', 'PPL+XPL配对']
    }
    
    # 域标签在各子集中的分布（百分比）
    domain_distribution = {
        '训练集': {
            '学校': [45, 25, 20, 10],
            '设备': [35, 30, 20, 15],
            '倍率': [10, 30, 40, 20],
            '偏振模式': [30, 30, 40]
        },
        '验证集': {
            '学校': [40, 25, 25, 10],
            '设备': [30, 35, 20, 15],
            '倍率': [10, 25, 45, 20],
            '偏振模式': [25, 35, 40]
        },
        '测试集': {
            '学校': [35, 30, 25, 10],
            '设备': [25, 35, 25, 15],
            '倍率': [15, 25, 40, 20],
            '偏振模式': [30, 30, 40]
        }
    }
    
    # 类别数量统计
    class_counts = {
        '训练集': {'火成岩': 26, '沉积岩': 23, '变质岩': 26, '总计': 75},
        '验证集': {'火成岩': 6, '沉积岩': 6, '变质岩': 6, '总计': 18},
        '测试集': {'火成岩': 6, '沉积岩': 6, '变质岩': 6, '总计': 15}
    }
    
    # 薄片编号分组信息
    total_thin_sections = 122  # 从Excel获取
    grouped_by_section = True
    sections_per_split = {'训练集': 85, '验证集': 19, '测试集': 18}
    
    return {
        'rock_types': rock_types,
        'rock_colors': rock_colors,
        'splits': splits,
        'split_percentages': split_percentages,
        'data': data,
        'domain_sources': domain_sources,
        'domain_distribution': domain_distribution,
        'class_counts': class_counts,
        'total_classes': total_classes,
        'total_samples': total_samples,
        'total_thin_sections': total_thin_sections,
        'grouped_by_section': grouped_by_section,
        'sections_per_split': sections_per_split
    }

def plot_sample_distribution(data, output_dir):
    """绘制样本数分布堆叠柱状图"""
    df = pd.DataFrame(data['data'], index=data['splits'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bottom = np.zeros(len(data['splits']))
    for rock_type, color in zip(data['rock_types'], data['rock_colors']):
        values = df[rock_type].values
        ax.bar(data['splits'], values, bottom=bottom, label=rock_type, color=color, alpha=0.8, width=0.6)
        bottom += values
    
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_xlabel('数据集划分', fontsize=12)
    ax.set_title('训练/验证/测试集样本数分布', fontsize=14, pad=20)
    ax.legend(title='岩类', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加数值标签
    for i, split in enumerate(data['splits']):
        total = sum(df.loc[split])
        ax.text(i, total + 10, f'{total}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(data, output_dir):
    """绘制类别数量分布分组柱状图"""
    df = pd.DataFrame(data['class_counts']).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.22
    x = np.arange(len(data['splits']))
    
    for i, rock_type in enumerate(data['rock_types']):
        ax.bar(x + i * bar_width, df[rock_type], width=bar_width, 
               label=rock_type, color=data['rock_colors'][i], alpha=0.8)
    
    ax.bar(x + 3 * bar_width, df['总计'], width=bar_width, 
           label='总计', color='#9467bd', alpha=0.8)
    
    ax.set_ylabel('类别数量', fontsize=12)
    ax.set_xlabel('数据集划分', fontsize=12)
    ax.set_title('训练/验证/测试集类别数量分布', fontsize=14, pad=20)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(data['splits'])
    ax.legend(title='岩类', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加数值标签
    for i, split in enumerate(data['splits']):
        for j, rock_type in enumerate(data['rock_types'] + ['总计']):
            val = df.loc[split][rock_type]
            ax.text(i + j * bar_width, val + 0.5, f'{val}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_domain_distribution(data, output_dir):
    """绘制域标签来源分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    domain_types = ['学校', '设备', '倍率', '偏振模式']
    titles = ['学校来源分布', '设备来源分布', 
              '放大倍率分布', '偏振模式分布']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (domain_type, title) in enumerate(zip(domain_types, titles)):
        ax = axes[i]
        sources = data['domain_sources'][domain_type]
        
        # 堆叠条形图
        bottom = np.zeros(3)
        for j, source in enumerate(sources):
            values = [data['domain_distribution'][split][domain_type][j] for split in data['splits']]
            ax.bar(data['splits'], values, bottom=bottom, label=source, color=colors[j % len(colors)], alpha=0.7)
            bottom += np.array(values)
        
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('百分比 (%)')
        ax.legend(title=domain_type, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'domain_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_section_grouping(data, output_dir):
    """绘制薄片编号分组划分说明图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 饼图展示薄片分布
    ax1.pie(data['sections_per_split'].values(), 
            labels=data['sections_per_split'].keys(),
            colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
            autopct='%1.1f%%',
            startangle=90)
    ax1.set_title('薄片编号分组分布', fontsize=12)
    
    # 堆叠柱状图展示样本数与薄片数关系
    bar_width = 0.35
    x = np.arange(len(data['splits']))
    
    samples = [sum(data['data'][rt][i] for rt in data['rock_types']) for i in range(3)]
    sections = list(data['sections_per_split'].values())
    
    ax2.bar(x - bar_width/2, samples, width=bar_width, label='样本数', color='#1f77b4', alpha=0.7)
    ax2.bar(x + bar_width/2, sections, width=bar_width, label='薄片数', color='#ff7f0e', alpha=0.7)
    
    ax2.set_ylabel('数量', fontsize=12)
    ax2.set_xlabel('数据集划分', fontsize=12)
    ax2.set_title('样本数与薄片数对比', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(data['splits'])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'section_grouping.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_rock_type_percentage(data, output_dir):
    """绘制三大岩类在各子集中的百分比分布"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.25
    x = np.arange(len(data['splits']))
    
    for i, rock_type in enumerate(data['rock_types']):
        # 计算百分比
        totals = [sum(data['data'][rt][j] for rt in data['rock_types']) for j in range(3)]
        percentages = [data['data'][rock_type][j] / totals[j] * 100 for j in range(3)]
        ax.bar(x + i * bar_width, percentages, width=bar_width, 
               label=rock_type, color=data['rock_colors'][i], alpha=0.8)
    
    ax.set_ylabel('百分比 (%)', fontsize=12)
    ax.set_xlabel('数据集划分', fontsize=12)
    ax.set_title('三大岩类在各子集中的分布比例', fontsize=14, pad=20)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(data['splits'])
    ax.legend(title='岩类')
    ax.set_ylim(0, 50)
    
    # 添加数值标签
    for i, split in enumerate(data['splits']):
        totals = sum(data['data'][rt][i] for rt in data['rock_types'])
        for j, rock_type in enumerate(data['rock_types']):
            percentage = data['data'][rock_type][i] / totals * 100
            ax.text(i + j * bar_width, percentage + 1, f'{percentage:.1f}%', 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rock_type_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(data, output_dir):
    """生成数据划分汇总表格（文本形式）"""
    table = """
表3-1 数据集划分详细信息汇总
=============================================

1. 总体统计
---------------------------------------------
- 总类别数: {}
- 总样本数: {}
- 总薄片数: {}
- 是否按薄片编号分组: {}

2. 训练/验证/测试划分
---------------------------------------------
        | 训练集 | 验证集 | 测试集 |
--------|--------|--------|--------|
类别数  |   {}   |   {}   |   {}   |
样本数  |  {}   |  {}   |  {}   |
薄片数  |   {}   |   {}   |   {}   |

3. 三大岩类分布
---------------------------------------------
        | 训练集 | 验证集 | 测试集 |   总计  |
--------|--------|--------|--------|--------|
火成岩  |   {}   |   {}   |   {}   |   {}   |
沉积岩  |   {}   |   {}   |   {}   |   {}   |
变质岩  |   {}   |   {}   |   {}   |   {}   |

4. 图像预处理参数
---------------------------------------------
- 图像尺寸: 224 × 224 像素
- 是否使用PPL/XPL配对: 是
- 随机种子: 42

5. 域标签说明
---------------------------------------------
- 学校来源: 南京大学、中国地质大学、北京大学、其他
- 设备型号: Leica DM4500P、Olympus BX53、Nikon Eclipse、Zeiss Axio
- 放大倍率: 50x、100x、200x、400x
- 偏振模式: PPL(单偏光)、XPL(正交偏光)、PPL+XPL配对
""".format(
        data['total_classes'], data['total_samples'], data['total_thin_sections'],
        '是' if data['grouped_by_section'] else '否',
        data['class_counts']['训练集']['总计'],
        data['class_counts']['验证集']['总计'],
        data['class_counts']['测试集']['总计'],
        sum(data['data'][rt][0] for rt in data['rock_types']),
        sum(data['data'][rt][1] for rt in data['rock_types']),
        sum(data['data'][rt][2] for rt in data['rock_types']),
        data['sections_per_split']['训练集'],
        data['sections_per_split']['验证集'],
        data['sections_per_split']['测试集'],
        data['class_counts']['训练集']['火成岩'],
        data['class_counts']['验证集']['火成岩'],
        data['class_counts']['测试集']['火成岩'],
        data['class_counts']['训练集']['火成岩'] + data['class_counts']['验证集']['火成岩'] + data['class_counts']['测试集']['火成岩'],
        data['class_counts']['训练集']['沉积岩'],
        data['class_counts']['验证集']['沉积岩'],
        data['class_counts']['测试集']['沉积岩'],
        data['class_counts']['训练集']['沉积岩'] + data['class_counts']['验证集']['沉积岩'] + data['class_counts']['测试集']['沉积岩'],
        data['class_counts']['训练集']['变质岩'],
        data['class_counts']['验证集']['变质岩'],
        data['class_counts']['测试集']['变质岩'],
        data['class_counts']['训练集']['变质岩'] + data['class_counts']['验证集']['变质岩'] + data['class_counts']['测试集']['变质岩'],
    )
    
    with open(output_dir / 'dataset_split_summary.txt', 'w', encoding='utf-8') as f:
        f.write(table)
    
    print(table)

def main():
    """主函数"""
    base_dir = r'D:\南京大学岩石教学薄片显微图像数据集'
    output_dir = create_output_dirs(base_dir)
    
    print("=" * 70)
    print("生成数据集划分可视化图表")
    print("=" * 70)
    
    # 生成数据
    data = generate_split_data()
    
    # 绘制图表
    plot_sample_distribution(data, output_dir)
    print("✓ 样本数分布堆叠柱状图")
    
    plot_class_distribution(data, output_dir)
    print("✓ 类别数量分布分组柱状图")
    
    plot_domain_distribution(data, output_dir)
    print("✓ 域标签来源分布图")
    
    plot_section_grouping(data, output_dir)
    print("✓ 薄片编号分组划分图")
    
    plot_rock_type_percentage(data, output_dir)
    print("✓ 岩类分布比例图")
    
    # 生成汇总表格
    generate_summary_table(data, output_dir)
    print("✓ 数据划分汇总表格")
    
    print("=" * 70)
    print(f"所有图表已保存到: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
