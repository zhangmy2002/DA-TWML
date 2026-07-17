# -*- coding: utf-8 -*-
"""
method_framework_diagram.py - DA-TWML方法框架完整流程图

生成内容：
1. 将核心模块关系示意图和训练流程图合并为完整的端到端流程图
2. 清晰标注MTVS→TKWTS→SWMU的信息流
3. 标注DAES在episode构造阶段的作用位置
4. 用不同颜色区分内环适应、任务评估、任务筛选、外环更新四个阶段
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_box(ax, x, y, width, height, text, color, text_color='black', fontsize=9, style='round'):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle=f"round,pad=0.02,rounding_size={height*0.3}",
                         linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=text_color, fontweight='bold', wrap=True)

def draw_arrow(ax, start, end, color='#333333', style='->', connectionstyle='arc3,rad=0'):
    """绘制箭头连接"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=2,
                               connectionstyle=connectionstyle))

def draw_dashed_arrow(ax, start, end, color='#666666'):
    """绘制虚线箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                               linestyle='dashed', connectionstyle='arc3,rad=0'))

def main():
    """主函数"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'input': '#4CAF50',      # 输入数据 - 绿色
        'episode': '#2196F3',    # Episode构造 - 蓝色
        'evaluation': '#FF9800', # 任务评估 - 橙色
        'selection': '#9C27B0',  # 任务筛选 - 紫色
        'inner_loop': '#E91E63', # 内环适应 - 粉红
        'outer_loop': '#00BCD4', # 外环更新 - 青色
        'output': '#795548',     # 输出 - 棕色
        'background': '#F5F5F5'  # 背景色
    }
    
    # 添加背景
    background = FancyBboxPatch((0.2, 0.2), 19.6, 15.6,
                                 boxstyle="round,pad=0.02",
                                 linewidth=2, edgecolor='#CCCCCC', 
                                 facecolor=colors['background'], alpha=0.3)
    ax.add_patch(background)
    
    # ==================== 标题 ====================
    ax.text(10, 15.3, 'DA-TWML: 域感知Top-k加权任务选择元学习框架', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(10, 14.9, 'Domain-Aware Top-k Weighted Meta-Learning for Rock Thin Section Classification', 
            ha='center', va='center', fontsize=11, style='italic')
    
    # ==================== 左侧：数据输入层 ====================
    # 数据输入框
    draw_box(ax, 2, 13, 2.5, 1, '岩石薄片\n图像数据集', colors['input'], 'white', 9)
    
    # 特征提取 backbone
    draw_box(ax, 2, 11.5, 2.5, 1, '特征提取\n(Conv4)', colors['input'], 'white', 9)
    
    # 域编码器
    draw_box(ax, 2, 10, 2.5, 1, '域编码器\n(Domain Encoder)', colors['input'], 'white', 9)
    
    # 连接箭头
    draw_arrow(ax, (2, 12.5), (2, 12))
    draw_arrow(ax, (2, 11), (2, 10.5))
    
    # ==================== 中上：Episode构造阶段（蓝色） ====================
    # Episode构造标题
    ax.text(6.5, 13.5, 'Episode构造阶段 (DAES)', ha='center', va='center', 
            fontsize=11, fontweight='bold', color=colors['episode'],
            bbox=dict(boxstyle='round', facecolor=colors['episode'], alpha=0.2))
    
    # Support Set
    draw_box(ax, 5.5, 12, 2.2, 1.2, 'Support Set\n(5-way-k-shot)', colors['episode'], 'white', 9)
    
    # Query Set
    draw_box(ax, 7.5, 12, 2.2, 1.2, 'Query Set\n(15 samples/class)', colors['episode'], 'white', 9)
    
    # 域标签分配
    draw_box(ax, 6.5, 10.5, 2.2, 1, '域标签分配\n(Domain Label)', colors['episode'], 'white', 9)
    
    # 连接
    draw_arrow(ax, (5.5, 11.4), (5.5, 11))
    draw_arrow(ax, (7.5, 11.4), (7.5, 11))
    draw_arrow(ax, (6.5, 11), (6.5, 11))
    draw_dashed_arrow(ax, (2, 10), (5, 12))
    draw_dashed_arrow(ax, (2, 10), (7, 12))
    
    # ==================== 中左：MTVS - 任务价值评估（橙色） ====================
    ax.text(5, 8.5, 'MTVS\n(多任务价值评估)', ha='center', va='center', 
            fontsize=10, fontweight='bold', color=colors['evaluation'],
            bbox=dict(boxstyle='round', facecolor=colors['evaluation'], alpha=0.2))
    
    # 四个评估指标
    draw_box(ax, 3.5, 7, 1.8, 0.9, 'L\n任务难度', '#FFB74D', 'black', 8)
    draw_box(ax, 5, 7, 1.8, 0.9, 'U\n预测不确定性', '#FFB74D', 'black', 8)
    draw_box(ax, 6.5, 7, 1.8, 0.9, 'G\n域偏移度', '#FFB74D', 'black', 8)
    draw_box(ax, 8, 7, 1.8, 0.9, 'C\n类混淆度', '#FFB74D', 'black', 8)
    
    # 连接评估指标
    for x_pos in [3.5, 5, 6.5, 8]:
        draw_arrow(ax, (x_pos, 8), (x_pos, 7.45))
    
    # S = 组合分数
    draw_box(ax, 5.75, 5.5, 2, 0.9, 'S = αL+βU+γG+δC\n综合价值分数', '#FF9800', 'white', 8)
    draw_arrow(ax, (3.5, 6.55), (4.5, 5.95))
    draw_arrow(ax, (5, 6.55), (5.25, 5.95))
    draw_arrow(ax, (6.5, 6.55), (6.25, 5.95))
    draw_arrow(ax, (8, 6.55), (7, 5.95))
    
    # ==================== 中右：TKWTS - Top-k任务选择（紫色） ====================
    ax.text(13, 8.5, 'TKWTS\n(Top-k加权任务选择)', ha='center', va='center', 
            fontsize=10, fontweight='bold', color=colors['selection'],
            bbox=dict(boxstyle='round', facecolor=colors['selection'], alpha=0.2))
    
    # 任务价值排序
    draw_box(ax, 11.5, 7, 2, 0.9, '任务价值排序\n(Sorting)', '#CE93D8', 'black', 8)
    
    # Top-k选择
    draw_box(ax, 13, 5.5, 2, 0.9, 'Top-k Selection\n(k=3)', '#9C27B0', 'white', 8)
    
    # 任务权重计算
    draw_box(ax, 14.5, 7, 2, 0.9, '任务权重计算\n(Softmax τs)', '#CE93D8', 'black', 8)
    
    # 连接
    draw_arrow(ax, (12.75, 8), (12.75, 7.45))
    draw_arrow(ax, (11.5, 6.55), (12, 5.95))
    draw_arrow(ax, (13, 6.55), (13, 5.95))
    draw_arrow(ax, (14.5, 6.55), (13.5, 5.95))
    draw_arrow(ax, (14.5, 7.45), (14.5, 7))
    
    # ==================== 右侧：内环适应 + 外环更新 ====================
    # 内环适应阶段标题
    ax.text(17, 13.5, '内环适应 (Inner Loop)', ha='center', va='center', 
            fontsize=11, fontweight='bold', color=colors['inner_loop'],
            bbox=dict(boxstyle='round', facecolor=colors['inner_loop'], alpha=0.2))
    
    # 梯度计算
    draw_box(ax, 17, 12, 2.2, 1, '梯度计算\n(Weighted Loss)', colors['inner_loop'], 'white', 9)
    
    # 快速权重更新
    draw_box(ax, 17, 10.5, 2.2, 1, '快速权重更新\n(θ <- θ - α×grad L)', colors['inner_loop'], 'white', 9)
    
    # Query预测
    draw_box(ax, 17, 9, 2.2, 1, 'Query预测\n& 损失计算', colors['inner_loop'], 'white', 9)
    
    # 连接内环
    draw_arrow(ax, (17, 11.5), (17, 11))
    draw_arrow(ax, (17, 10), (17, 9.5))
    draw_dashed_arrow(ax, (5.75, 5.5), (16, 12))
    draw_dashed_arrow(ax, (13, 5), (16, 12))
    
    # 外环更新阶段标题
    ax.text(17, 6.5, '外环更新 (Outer Loop)', ha='center', va='center', 
            fontsize=11, fontweight='bold', color=colors['outer_loop'],
            bbox=dict(boxstyle='round', facecolor=colors['outer_loop'], alpha=0.2))
    
    # 累积梯度
    draw_box(ax, 17, 5.5, 2.2, 1, '累积梯度\n(Aggregate)', colors['outer_loop'], 'white', 9)
    
    # 元权重更新
    draw_box(ax, 17, 4, 2.2, 1, '元权重更新\n(φ <- φ - β×grad)', colors['outer_loop'], 'white', 9)
    
    # 连接外环
    draw_arrow(ax, (17, 8.5), (17, 6))
    draw_arrow(ax, (17, 5), (17, 4.5))
    draw_arrow(ax, (17, 3.5), (17, 3))
    
    # ==================== SWMU - 域感知权重更新 ====================
    ax.text(13, 2.5, 'SWMU\n(域感知权重更新模块)', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#607D8B',
            bbox=dict(boxstyle='round', facecolor='#607D8B', alpha=0.2))
    
    # 域权重
    draw_box(ax, 11, 1.5, 2, 0.9, '域权重调整\n(Domain-aware)', '#90A4AE', 'black', 8)
    
    # 元学习器权重
    draw_box(ax, 15, 1.5, 2, 0.9, '元学习器权重\n(φ update)', '#90A4AE', 'black', 8)
    
    # 连接
    draw_arrow(ax, (13, 2), (12, 1.95))
    draw_arrow(ax, (13, 2), (14, 1.95))
    draw_dashed_arrow(ax, (17, 3), (15, 2))
    
    # ==================== 输出 ====================
    draw_box(ax, 10, 1.5, 2.5, 0.9, '任务价值分布\n更新完成', colors['output'], 'white', 9)
    draw_arrow(ax, (11, 1.5), (10, 1.5))
    draw_arrow(ax, (15, 1.5), (10, 1.5))
    
    # ==================== 图例 ====================
    legend_x = 0.8
    legend_y = 2
    
    legend_items = [
        (colors['input'], '数据输入'),
        (colors['episode'], 'Episode构造 (DAES)'),
        (colors['evaluation'], '任务评估 (MTVS)'),
        (colors['selection'], '任务筛选 (TKWTS)'),
        (colors['inner_loop'], '内环适应'),
        (colors['outer_loop'], '外环更新'),
        ('#607D8B', '权重更新 (SWMU)'),
    ]
    
    ax.text(legend_x, legend_y + 0.8, '图例:', fontsize=10, fontweight='bold')
    
    for i, (color, label) in enumerate(legend_items):
        y_pos = legend_y - i * 0.4
        box = FancyBboxPatch((legend_x, y_pos - 0.12), 0.3, 0.24,
                             boxstyle="round,pad=0.01",
                             linewidth=1, edgecolor=color, facecolor=color, alpha=0.5)
        ax.add_patch(box)
        ax.text(legend_x + 0.5, y_pos, label, fontsize=8, va='center')
    
    # ==================== 信息流标注 ====================
    # MTVS → TKWTS
    ax.annotate('', xy=(10, 5.95), xytext=(8.5, 5.95),
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=2.5))
    ax.text(9.25, 6.2, 'S (任务价值)', fontsize=8, ha='center', color='#E91E63', fontweight='bold')
    
    # TKWTS → 内环
    ax.annotate('', xy=(16, 11), xytext=(14.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=2,
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(15, 8, 'Top-k Tasks', fontsize=8, ha='center', color='#9C27B0', fontweight='bold')
    
    # 内环 → 外环
    ax.annotate('', xy=(17, 6.5), xytext=(17, 8.5),
                arrowprops=dict(arrowstyle='->', color='#00BCD4', lw=2))
    ax.text(17.8, 7.5, 'Loss', fontsize=8, ha='center', color='#00BCD4', fontweight='bold')
    
    # 添加阶段分隔框
    # Episode构造阶段
    episode_box = FancyBboxPatch((4.5, 9.2), 4.5, 4.5,
                                  boxstyle="round,pad=0.05",
                                  linewidth=2, edgecolor=colors['episode'], 
                                  facecolor=colors['episode'], alpha=0.1)
    ax.add_patch(episode_box)
    
    # MTVS阶段
    mtvs_box = FancyBboxPatch((2.5, 4.3), 7, 4.5,
                                boxstyle="round,pad=0.05",
                                linewidth=2, edgecolor=colors['evaluation'], 
                                facecolor=colors['evaluation'], alpha=0.1)
    ax.add_patch(mtvs_box)
    
    # TKWTS阶段
    tkwts_box = FancyBboxPatch((10.2, 4.3), 5.5, 4.5,
                                boxstyle="round,pad=0.05",
                                linewidth=2, edgecolor=colors['selection'], 
                                facecolor=colors['selection'], alpha=0.1)
    ax.add_patch(tkwts_box)
    
    # 保存图表
    base_dir = r'D:\南京大学岩石教学薄片显微图像数据集'
    output_dir = Path(base_dir) / 'figures_final' / 'method'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'DA_TWML_framework_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'DA_TWML_framework_diagram.pdf', bbox_inches='tight')
    plt.close()
    
    print("=" * 70)
    print("生成DA-TWML方法框架完整流程图")
    print("=" * 70)
    print(f"✓ 已保存到: {output_dir}")
    print("  - DA_TWML_framework_diagram.png")
    print("  - DA_TWML_framework_diagram.pdf")
    print("=" * 70)

if __name__ == "__main__":
    main()
