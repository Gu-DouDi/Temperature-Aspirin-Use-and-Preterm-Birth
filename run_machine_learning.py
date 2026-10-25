"""读取Excel数据，准备分析队列并完成模型训练、评估和SHAP分析。"""

import argparse
from pathlib import Path
import sys
import pandas as pd


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))


INPUT_EXCEL = Path(__file__).resolve().parent
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def main():
    # 1. 读取Excel路径、输出路径和分析参数。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "excel", nargs="?", type=Path, default=INPUT_EXCEL, help="输入Excel数据文件路径"
    )
    parser.add_argument(
        "output", nargs="?", type=Path, default=OUTPUT_DIR, help="分析结果输出文件夹"
    )
    parser.add_argument("--sheet-name", default=0, help="工作表名称，默认读取第一张表")
    parser.add_argument("--jobs", type=int, default=4, help="训练和SHAP的并行任务数")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=BUNDLE / "config/selected_model.json",
        help="固定ExtraTrees模型参数的JSON文件路径",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # 2. 读取Excel并准备内存中的开发集和测试集。
    import prepare_cohort
    import train_models
    import evaluate_models
    import explain_models

    print("步骤1/4：数据准备", flush=True)
    data = prepare_cohort.main(args.excel.expanduser().resolve(), args.sheet_name)

    # 3. 在开发集训练固定参数ExtraTrees模型。
    print("步骤2/4：模型训练", flush=True)
    bundle = train_models.main(
        data, args.jobs, args.model_config.expanduser().resolve()
    )

    # 4. 评估同一个内存模型，计算SHAP及温度与阿司匹林的成对贡献。
    print("步骤3/4：测试集评估", flush=True)
    tables = evaluate_models.main(data, bundle)
    print("步骤4/4：SHAP解释", flush=True)
    tables.update(explain_models.main(data, bundle, args.jobs))

    # 5. 仅输出Performance和SHAP相关工作表。
    result_file = output / "ml_analysis_results.xlsx"
    sheet_names = [
        "Performance",
        "SHAP_Importance",
        "SHAP_Values",
        "SHAP_Pairs",
        "Pair_Bin_Summary",
    ]
    with pd.ExcelWriter(result_file, engine="openpyxl") as writer:
        for name in sheet_names:
            table = tables[name]
            table.to_excel(writer, sheet_name=name, index=False, freeze_panes=(1, 0))
            sheet = writer.sheets[name]
            sheet.auto_filter.ref = sheet.dimensions
            for cells in sheet.columns:
                width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in cells)
                sheet.column_dimensions[cells[0].column_letter].width = min(max(width + 2, 12), 45)
    print("结果已保存至：", result_file, flush=True)


if __name__ == "__main__":
    main()
