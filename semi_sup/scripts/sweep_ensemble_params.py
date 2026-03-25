import argparse
import csv
import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List


# ======================================================
# 参数扫描脚本（针对伪标注融合）
#
# 优化点：
# 1) 自动扫描 tau_f / theta / min_votes / soft_nms_iou 组合。
# 2) 每组参数自动产出独立伪标注文件，避免覆盖。
# 3) 记录每组的 annotation 数量到 CSV，便于后续筛选。
# 4) 可选执行评估命令模板，把每组结果都跑一遍并记录。
# ======================================================


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Grid sweep for ensemble pseudo-annotation parameters")

    parser.add_argument(
        "--generator_script",
        type=str,
        default=str(Path("semi_sup") / "scripts" / "generate_ensemble_pseudo_annotations.py"),
        help="Path to pseudo label generation script",
    )

    parser.add_argument(
        "--input_json",
        type=str,
        default=str(Path("data") / "pseudo_labels" / "zerowaste-s_consolidated_pseudo_annotations.json"),
        help="Consolidated pseudo annotation json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("data") / "pseudo_labels" / "sweeps"),
        help="Directory to save sweep outputs",
    )

    # 重点扫描四个参数
    parser.add_argument("--tau_f_list", type=str, default="0.30,0.35,0.40,0.45")
    parser.add_argument("--theta_list", type=str, default="0.55,0.60,0.65,0.70")
    parser.add_argument("--min_votes_list", type=str, default="2,3")
    parser.add_argument("--soft_nms_iou_list", type=str, default="0.45,0.50,0.55")

    # 其余参数保持固定值（也可按需改）
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--min_area", type=float, default=16.0)
    parser.add_argument("--max_ar", type=float, default=12.0)
    parser.add_argument("--soft_nms_sigma", type=float, default=0.5)
    parser.add_argument("--soft_nms_min_score", type=float, default=0.2)
    parser.add_argument("--model_weights_json", type=str, default="")

    # 可选：每组参数生成后自动执行评估命令（模板）
    # 用法示例：
    # --eval_cmd_template "python external_modules/mmdetection/tools/test.py <CONFIG> <CKPT> --cfg-options test_evaluator.ann_file={pseudo_json}"
    parser.add_argument(
        "--eval_cmd_template",
        type=str,
        default="",
        help="Optional eval command template. You can use {pseudo_json} placeholder.",
    )

    return parser.parse_args()


def run_command(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def count_annotations(coco_path: Path) -> int:
    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data.get("annotations", []))


def main():
    args = parse_args()

    tau_f_list = parse_float_list(args.tau_f_list)
    theta_list = parse_float_list(args.theta_list)
    min_votes_list = parse_int_list(args.min_votes_list)
    soft_nms_iou_list = parse_float_list(args.soft_nms_iou_list)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"sweep_results_{timestamp}.csv"

    rows: List[Dict] = []
    combos = list(itertools.product(tau_f_list, theta_list, min_votes_list, soft_nms_iou_list))

    print(f"[INFO] total combinations: {len(combos)}")

    for idx, (tau_f, theta, min_votes, soft_nms_iou) in enumerate(combos, start=1):
        # 优化点：每个参数组合一个独立输出，便于可追溯复现
        # 文件名只替换参数中的小数点，扩展名保持 .json
        out_name = (
            f"ensemble_tauF{tau_f:.2f}_theta{theta:.2f}_votes{min_votes}_snmsIou{soft_nms_iou:.2f}"
            .replace(".", "p")
            + ".json"
        )
        output_json = output_dir / out_name

        cmd = [
            "python",
            args.generator_script,
            "--input",
            args.input_json,
            "--output",
            str(output_json),
            "--tau",
            str(args.tau),
            "--tau_f",
            str(tau_f),
            "--theta",
            str(theta),
            "--min_votes",
            str(min_votes),
            "--alpha",
            str(args.alpha),
            "--beta",
            str(args.beta),
            "--gamma",
            str(args.gamma),
            "--min_area",
            str(args.min_area),
            "--max_ar",
            str(args.max_ar),
            "--soft_nms_iou",
            str(soft_nms_iou),
            "--soft_nms_sigma",
            str(args.soft_nms_sigma),
            "--soft_nms_min_score",
            str(args.soft_nms_min_score),
        ]

        if args.model_weights_json.strip():
            cmd.extend(["--model_weights_json", args.model_weights_json])

        print(f"\n[RUN {idx}/{len(combos)}] tau_f={tau_f}, theta={theta}, min_votes={min_votes}, soft_nms_iou={soft_nms_iou}")
        proc = run_command(cmd)

        generation_ok = proc.returncode == 0 and output_json.exists()
        ann_count = count_annotations(output_json) if generation_ok else -1

        eval_return_code = ""
        eval_stdout_tail = ""
        eval_stderr_tail = ""

        # 可选评估步骤
        if generation_ok and args.eval_cmd_template.strip():
            eval_cmd = args.eval_cmd_template.format(pseudo_json=str(output_json))
            eval_proc = subprocess.run(eval_cmd, check=False, capture_output=True, text=True, shell=True)
            eval_return_code = str(eval_proc.returncode)
            eval_stdout_tail = eval_proc.stdout[-500:]
            eval_stderr_tail = eval_proc.stderr[-500:]

        rows.append(
            {
                "tau_f": tau_f,
                "theta": theta,
                "min_votes": min_votes,
                "soft_nms_iou": soft_nms_iou,
                "generation_ok": generation_ok,
                "generator_return_code": proc.returncode,
                "annotation_count": ann_count,
                "output_json": str(output_json),
                "generator_stdout_tail": proc.stdout[-500:],
                "generator_stderr_tail": proc.stderr[-500:],
                "eval_return_code": eval_return_code,
                "eval_stdout_tail": eval_stdout_tail,
                "eval_stderr_tail": eval_stderr_tail,
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] sweep finished. results: {csv_path}")


if __name__ == "__main__":
    main()
