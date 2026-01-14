#!/usr/bin/env python3
"""统计 valid_mask.csv 中各列的有效点数量"""

import pandas as pd
from pathlib import Path
from typing import Dict


def analyze_valid_mask(csv_path: str) -> Dict[str, int]:
    """
    分析 valid_mask.csv 文件，统计各列中值为1的数量
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        包含统计信息的字典
    """
    df = pd.read_csv(csv_path)
    
    # 统计总数据条目数（不包括表头）
    total_entries = len(df)
    
    # 统计第2、3、4列（step_0, step_1, step_2）中值为1的数量
    step_columns = ['step_0', 'step_1', 'step_2']
    stats = {
        'total_entries': total_entries
    }
    
    for col in step_columns:
        if col in df.columns:
            count_ones = (df[col] == 1).sum()
            stats[f'{col}_ones'] = int(count_ones)
            stats[f'{col}_rate'] = count_ones / total_entries if total_entries > 0 else 0.0
    
    return stats


def print_statistics(stats: Dict[str, int]) -> None:
    """打印统计结果"""
    print("=" * 50)
    print("Valid Mask 统计结果")
    print("=" * 50)
    print(f"总数据条目数: {stats['total_entries']}")
    print("-" * 50)
    
    for step in ['step_0', 'step_1', 'step_2']:
        ones_key = f'{step}_ones'
        rate_key = f'{step}_rate'
        if ones_key in stats:
            print(f"{step:8s}: {stats[ones_key]:4d} 个1  ({stats[rate_key]:6.2%} 有效率)")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    
    # 默认路径或从命令行参数获取
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # 默认使用 output/OmegaG_11/valid_mask.csv
        csv_path = "output/OmegaG_11/valid_mask.csv"
    
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        print(f"错误: 文件不存在 - {csv_path}")
        sys.exit(1)
    
    stats = analyze_valid_mask(str(csv_file))
    print_statistics(stats)
