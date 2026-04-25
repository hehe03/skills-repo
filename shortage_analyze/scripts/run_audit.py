#!/usr/bin/env python3
"""
欠料归因分析审计工作流脚本

功能：
1. 运行分析脚本生成带trace的输出
2. 运行审计脚本生成审计报告
3. 总结审计结果

使用方法：
    python run_audit.py <input_excel_path>
"""

import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("使用方法: python run_audit.py <input_excel_path>")
        print("\n示例:")
        print("  python run_audit.py 欠料数据.xlsx")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"文件不存在: {input_path}")
        sys.exit(1)

    base_dir = input_path.parent

    output_path = base_dir / f"{input_path.stem}_with_trace.xlsx"
    trace_path = base_dir / f"{input_path.stem}_with_trace.jsonl"
    audit_excel_path = base_dir / f"{input_path.stem}_audit.xlsx"
    audit_md_path = base_dir / f"{input_path.stem}_audit_report.md"

    print("=" * 60)
    print("Step 1: 运行分析脚本（带 trace 输出）")
    print("=" * 60)
    
    analyze_cmd = [
        sys.executable,
        "scripts/analyze_shortage.py",
        str(input_path),
        str(output_path),
        "--trace"
    ]
    
    subprocess.run(analyze_cmd, cwd=base_dir)
    
    print("\n" + "=" * 60)
    print("Step 2: 运行审计脚本")
    print("=" * 60)
    
    skill_ops_dir = base_dir.parent / "skill-ops"
    if not skill_ops_dir.exists():
        print(f"skill-ops 目录不存在: {skill_ops_dir}")
        sys.exit(1)
    
    audit_cmd = [
        sys.executable,
        "scripts/label_audit.py",
        str(output_path),
        "--human-col", "人工标注_L2分类结果",
        "--pred-col", "L2分类结果",
        "--candidate-col", "候选标签",
        "--low-confidence-labels", "替代交付异常",
        "--output-path", str(audit_excel_path),
        "--report-path", str(audit_md_path)
    ]
    
    subprocess.run(audit_cmd, cwd=skill_ops_dir)
    
    print("\n" + "=" * 60)
    print("Step 3: 生成 Trace 摘要")
    print("=" * 60)
    
    if trace_path.exists():
        trace_summary_cmd = [
            sys.executable,
            "scripts/trace_jsonl_summary.py",
            str(trace_path),
            "--output-path", str(base_dir / f"{input_path.stem}_trace_summary.md")
        ]
        subprocess.run(trace_summary_cmd, cwd=skill_ops_dir)
    
    print("\n" + "=" * 60)
    print("完成！输出文件：")
    print("=" * 60)
    print(f"- 分析结果: {output_path}")
    print(f"- Trace文件: {trace_path}")
    print(f"- 审计Excel: {audit_excel_path}")
    print(f"- 审计报告: {audit_md_path}")
    
    if audit_md_path.exists():
        print("\n" + "=" * 60)
        print("审计报告摘要：")
        print("=" * 60)
        with audit_md_path.open('r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')[:15]
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()