#!/usr/bin/env python3
"""
可视化 time_score_data.csv 数据的工具
支持在同一张图上显示多个数据列，每列使用独立的纵坐标刻度
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_data(csv_path: str) -> pd.DataFrame:
    """加载 CSV 数据文件，并将数值列转换为浮点数"""
    df = pd.read_csv(csv_path)
    
    # 尝试将除了第一列外的所有列转换为数值类型
    # 无法转换的值（如 "--"）会被转换为 NaN
    for col in df.columns:
        if col != df.columns[0]:  # 跳过第一列（通常是索引列）
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def plot_multi_axis(df: pd.DataFrame, x_column: str, y_columns: list[str], 
                    title: str = "Data Visualization", x_label: str = None):
    """
    在同一张图上绘制多个数据列，每列使用独立的 y 轴
    
    参数:
        df: 数据框
        x_column: x 轴使用的列名
        y_columns: 要绘制的 y 轴列名列表
        title: 图表标题
        x_label: x 轴标签（默认使用 x_column）
    """
    if not y_columns:
        print("错误：没有选择要显示的数据列")
        return
    
    if x_label is None:
        x_label = x_column
    
    # 创建图形和主轴
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 颜色列表
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    axes = [ax1]  # 存储所有轴
    lines = []    # 存储所有线条
    labels = []   # 存储所有标签
    
    y_label_dict = {
        "score_positional_dir_raw": "Raw Positional Direction Score",
        "score_positional_dis_raw": "Raw Positional Distance Score",
    }
    
    # 绘制第一个数据系列
    color = colors[0 % len(colors)]
    line1 = ax1.plot(df[x_column], df[y_columns[0]], color=color, 
                     marker='o', linewidth=2, markersize=6, label=y_columns[0])
    ax1.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel(y_label_dict.get(y_columns[0], y_columns[0]), color=color, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3, linestyle='--')
    lines.extend(line1)
    labels.append(y_label_dict.get(y_columns[0], y_columns[0]))
    
    # 为其他数据系列创建额外的 y 轴
    for i, y_col in enumerate(y_columns[1:], start=1):
        color = colors[i % len(colors)]
        
        # 创建新的 y 轴
        ax_new = ax1.twinx()
        
        # 如果有多个额外轴，需要调整位置避免重叠
        if i > 1:
            # 将轴向右移动
            ax_new.spines['right'].set_position(('outward', 60 * (i - 1)))
        
        line = ax_new.plot(df[x_column], df[y_col], color=color, 
                          marker='s', linewidth=2, markersize=6, label=y_col)
        ax_new.set_ylabel(y_label_dict.get(y_col, y_col), color=color, fontsize=11, fontweight='bold')
        ax_new.tick_params(axis='y', labelcolor=color)
        
        # 设置y轴从0开始（忽略NaN值）
        # y_max = df[y_col].dropna().max() * 1.1  # 留出10%的空间
        # ax_new.set_ylim(0, y_max)
        
        # 格式化y轴刻度为2位小数
        ax_new.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        
        axes.append(ax_new)
        lines.extend(line)
        labels.append(y_label_dict.get(y_col, y_col))
    
    # 添加标题
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 创建图例（将所有线条的图例放在一起）
    ax1.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.98, 0.02), 
              framealpha=0.9, fontsize=10)
    
    # 调整布局以避免标签被裁剪
    fig.tight_layout()
    
    # 如果有多个额外轴，需要额外的右边距
    if len(y_columns) > 2:
        plt.subplots_adjust(right=0.85 - 0.05 * (len(y_columns) - 2))
    
    plt.show()


def interactive_selection(df: pd.DataFrame):
    """交互式选择要显示的列"""
    # 获取所有可用的列（排除 Point Count 作为 x 轴）
    available_columns = [col for col in df.columns if col != 'Point Count']
    
    print("\n可用的数据列:")
    for i, col in enumerate(available_columns, 1):
        print(f"  {i}. {col}")
    
    print("\n请输入要显示的列编号（用逗号分隔，例如：1,2,3）")
    print("或输入 'all' 显示所有列")
    
    user_input = input("您的选择: ").strip()
    
    if user_input.lower() == 'all':
        return available_columns
    
    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(',')]
        selected = [available_columns[i] for i in indices if 0 <= i < len(available_columns)]
        return selected
    except (ValueError, IndexError):
        print("输入无效，将显示所有列")
        return available_columns


def main():
    parser = argparse.ArgumentParser(
        description='可视化 time_score_data.csv 数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互式选择列
  python show_time_score_data.py
  
  # 指定要显示的列
  python show_time_score_data.py -c "Time(second)" "Best score"
  
  # 显示所有列
  python show_time_score_data.py --all
        """
    )
    
    parser.add_argument('-f', '--file', 
                       default='python/data_process/time_score_data.csv',
                       help='CSV 文件路径（默认: python/data_process/time_score_data.csv）')
    
    parser.add_argument('-c', '--columns', nargs='+',
                        default=["Time(second)", "Best score", "score_positional_dir_raw", "score_positional_dis_raw"],
                        # default=["Time(second)", "score_positional_dir_raw", "score_positional_dis_raw"],
                       help='要显示的列名（空格分隔）')
    
    parser.add_argument('--all', action='store_true',
                       help='显示所有数据列')
    
    parser.add_argument('-x', '--x-column', default='Point Count',
                       help='x 轴使用的列名（默认: Point Count）')
    
    parser.add_argument('-t', '--title', 
                       default='Time and Score Data Visualization',
                       help='图表标题')
    
    args = parser.parse_args()
    
    # 加载数据
    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"错误：文件不存在: {csv_path}")
        return
    
    print(f"正在加载数据: {csv_path}")
    df = load_data(csv_path)
    print(f"数据加载完成，共 {len(df)} 行")
    
    # 确定要显示的列
    if args.columns:
        # 使用命令行指定的列
        selected_columns = args.columns
        # 验证列名是否存在
        invalid_cols = [col for col in selected_columns if col not in df.columns]
        if invalid_cols:
            print(f"警告：以下列不存在: {invalid_cols}")
            selected_columns = [col for col in selected_columns if col in df.columns]
    elif args.all:
        # 显示所有列（除了 x 轴）
        selected_columns = [col for col in df.columns if col != args.x_column]
    else:
        # 交互式选择
        selected_columns = interactive_selection(df)
    
    if not selected_columns:
        print("没有选择任何列，退出")
        return
    
    print(f"\n将显示以下列: {selected_columns}")
    
    # 绘制图表
    plot_multi_axis(df, args.x_column, selected_columns, 
                   title=args.title, x_label=args.x_column)


if __name__ == "__main__":
    main()
