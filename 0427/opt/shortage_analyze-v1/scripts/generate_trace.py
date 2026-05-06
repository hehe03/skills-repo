#!/usr/bin/env python3
"""为欠料归因分析生成带预测结果的 Excel 和逐行 trace。"""

from __future__ import annotations

import sys
from pathlib import Path

from analyze_shortage import analyze_excel


def main() -> None:
    if len(sys.argv) < 2:
        print('使用方法: python generate_trace.py <input_excel_path> [output_excel_path] [trace_output_path]')
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_name(f'{input_path.stem}_trace结果.xlsx')
    trace_output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else input_path.with_name(f'{input_path.stem}_trace.jsonl')

    analyze_excel(str(input_path), str(output_path), trace_output_path=str(trace_output_path))


if __name__ == '__main__':
    main()
