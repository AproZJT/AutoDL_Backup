import argparse
import csv
import itertools
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# ======================================================
# 参数扫描脚本（针对伪标注融合）
#
# 优化点：
# 1) 自动扫描 tau_f / theta / min_votes / soft_nms_iou 组合。
# 2) 每组参数自动产出独立伪标注文件，避免覆盖。
# 3) 记录每组 annotation 数量与耗时到 CSV。
# 4) 可选执行评估命令模板，把每组结果都跑一遍并记录。
# 5) 支持自动二阶段：Phase1 粗搜 -> Top-K -> Phase2 细搜。
# ======================================================


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def unique_sorted_floats(values: List[float]) -> List[float]:
    return sorted(set(round(v, 4) for v in values))


def clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


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

    # 单阶段/Phase1 扫描参数
    parser.add_argument("--tau_f_list", type=str, default="0.30,0.35,0.40,0.45")
    parser.add_argument("--theta_list", type=str, default="0.55,0.60,0.65,0.70")
    parser.add_argument("--min_votes_list", type=str, default="2,3")
    parser.add_argument("--soft_nms_iou_list", type=str, default="0.45,0.50,0.55")

    # 二阶段开关 + 细化策略
    parser.add_argument("--auto_two_stage", action="store_true", help="Enable automatic two-stage sweep")
    parser.add_argument("--top_k", type=int, default=6, help="Select top-k from phase1 for phase2 expansion")
    parser.add_argument("--phase2_float_delta", type=float, default=0.03, help="Float expansion delta for tau_f/theta/soft_nms_iou")

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


def extract_processed_images(stdout_text: str) -> int:
    if not stdout_text:
        return -1

    patterns = [
        r"processed\s+images\s*[:=]\s*(\d+)",
        r"images\s+processed\s*[:=]\s*(\d+)",
        r"processed\s*(\d+)\s*images",
    ]
    lower_text = stdout_text.lower()

    for p in patterns:
        m = re.search(p, lower_text)
        if m:
            return int(m.group(1))

    return -1


def save_phase_csv(csv_path: Path, rows: List[Dict]):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def run_phase(
    args,
    phase_name: str,
    combos: List[Tuple[float, float, int, float]],
    output_dir: Path,
    log_dir: Path,
    timestamp: str,
) -> Tuple[List[Dict], Path]:
    rows: List[Dict] = []
    csv_path = output_dir / f"sweep_results_{phase_name}_{timestamp}.csv"

    print(f"\n[INFO] {phase_name} total combinations: {len(combos)}")

    sweep_start_time = time.time()

    for idx, (tau_f, theta, min_votes, soft_nms_iou) in enumerate(combos, start=1):
        out_name = (
            f"{phase_name}_ensemble_tauF{tau_f:.2f}_theta{theta:.2f}_votes{min_votes}_snmsIou{soft_nms_iou:.2f}".replace(".", "p")
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

        run_start_time = time.time()
        print(
            f"\n[{phase_name} RUN {idx}/{len(combos)}] "
            f"tau_f={tau_f}, theta={theta}, min_votes={min_votes}, soft_nms_iou={soft_nms_iou}"
        )
        proc = run_command(cmd)

        run_tag = f"{phase_name}_tauF{tau_f:.2f}_theta{theta:.2f}_votes{min_votes}_snmsIou{soft_nms_iou:.2f}".replace(".", "p")
        log_path = log_dir / f"{run_tag}.log"
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"[PHASE] {phase_name}\n")
            lf.write(f"[RUN] {idx}/{len(combos)}\n")
            lf.write(f"[PARAMS] tau_f={tau_f}, theta={theta}, min_votes={min_votes}, soft_nms_iou={soft_nms_iou}\n")
            lf.write(f"[COMMAND] {' '.join(cmd)}\n")
            lf.write(f"[RETURN_CODE] {proc.returncode}\n\n")
            lf.write("[STDOUT]\n")
            lf.write(proc.stdout if proc.stdout else "<EMPTY>\n")
            lf.write("\n[STDERR]\n")
            lf.write(proc.stderr if proc.stderr else "<EMPTY>\n")

        generation_ok = proc.returncode == 0 and output_json.exists()
        ann_count = count_annotations(output_json) if generation_ok else -1

        run_elapsed_sec = time.time() - run_start_time
        sweep_elapsed_sec = time.time() - sweep_start_time
        avg_per_run_sec = sweep_elapsed_sec / idx
        remaining_runs = len(combos) - idx
        eta_remaining_sec = avg_per_run_sec * remaining_runs
        processed_images = extract_processed_images(proc.stdout)

        print(
            f"[{phase_name} PROGRESS] run_time={run_elapsed_sec:.1f}s | "
            f"avg={avg_per_run_sec:.1f}s/run | "
            f"eta={eta_remaining_sec/60:.1f}min | "
            f"processed_images={processed_images}"
        )

        eval_return_code = ""
        eval_stdout_tail = ""
        eval_stderr_tail = ""

        if generation_ok and args.eval_cmd_template.strip():
            eval_cmd = args.eval_cmd_template.format(pseudo_json=str(output_json))
            eval_proc = subprocess.run(eval_cmd, check=False, capture_output=True, text=True, shell=True)
            eval_return_code = str(eval_proc.returncode)
            eval_stdout_tail = eval_proc.stdout[-500:]
            eval_stderr_tail = eval_proc.stderr[-500:]

        rows.append(
            {
                "phase": phase_name,
                "tau_f": tau_f,
                "theta": theta,
                "min_votes": min_votes,
                "soft_nms_iou": soft_nms_iou,
                "generation_ok": generation_ok,
                "generator_return_code": proc.returncode,
                "annotation_count": ann_count,
                "processed_images": processed_images,
                "run_elapsed_sec": round(run_elapsed_sec, 3),
                "avg_per_run_sec": round(avg_per_run_sec, 3),
                "eta_remaining_sec": round(eta_remaining_sec, 3),
                "output_json": str(output_json),
                "run_log": str(log_path),
                "generator_stdout_tail": proc.stdout[-500:],
                "generator_stderr_tail": proc.stderr[-500:],
                "eval_return_code": eval_return_code,
                "eval_stdout_tail": eval_stdout_tail,
                "eval_stderr_tail": eval_stderr_tail,
            }
        )

    save_phase_csv(csv_path, rows)
    print(f"\n[INFO] {phase_name} finished. results: {csv_path}")
    return rows, csv_path


def build_phase2_combos(top_rows: List[Dict], delta: float) -> List[Tuple[float, float, int, float]]:
    tau_f_vals: List[float] = []
    theta_vals: List[float] = []
    min_votes_vals: List[int] = []
    soft_nms_iou_vals: List[float] = []

    for row in top_rows:
        tau_f = float(row["tau_f"])
        theta = float(row["theta"])
        min_votes = int(row["min_votes"])
        soft_nms_iou = float(row["soft_nms_iou"])

        tau_f_vals.extend([
            clamp(tau_f - delta, 0.05, 0.95),
            clamp(tau_f, 0.05, 0.95),
            clamp(tau_f + delta, 0.05, 0.95),
        ])
        theta_vals.extend([
            clamp(theta - delta, 0.30, 0.95),
            clamp(theta, 0.30, 0.95),
            clamp(theta + delta, 0.30, 0.95),
        ])
        soft_nms_iou_vals.extend([
            clamp(soft_nms_iou - delta, 0.30, 0.90),
            clamp(soft_nms_iou, 0.30, 0.90),
            clamp(soft_nms_iou + delta, 0.30, 0.90),
        ])
        min_votes_vals.append(min_votes)

    tau_f_vals = unique_sorted_floats(tau_f_vals)
    theta_vals = unique_sorted_floats(theta_vals)
    soft_nms_iou_vals = unique_sorted_floats(soft_nms_iou_vals)
    min_votes_vals = sorted(set(min_votes_vals))

    combos = list(itertools.product(tau_f_vals, theta_vals, min_votes_vals, soft_nms_iou_vals))
    return combos


def select_top_k(rows: List[Dict], top_k: int) -> List[Dict]:
    ok_rows = [r for r in rows if r["generation_ok"]]
    if not ok_rows:
        return []

    # 主排序：annotation_count 降序；次排序：run_elapsed_sec 升序
    ok_rows.sort(key=lambda r: (int(r["annotation_count"]), -float(r["run_elapsed_sec"])), reverse=True)
    return ok_rows[: max(1, top_k)]


def main():
    args = parse_args()

    tau_f_list = parse_float_list(args.tau_f_list)
    theta_list = parse_float_list(args.theta_list)
    min_votes_list = parse_int_list(args.min_votes_list)
    soft_nms_iou_list = parse_float_list(args.soft_nms_iou_list)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    phase1_combos = list(itertools.product(tau_f_list, theta_list, min_votes_list, soft_nms_iou_list))
    phase1_rows, phase1_csv = run_phase(
        args=args,
        phase_name="phase1",
        combos=phase1_combos,
        output_dir=output_dir,
        log_dir=log_dir,
        timestamp=timestamp,
    )

    all_rows = list(phase1_rows)

    if args.auto_two_stage:
        top_rows = select_top_k(phase1_rows, args.top_k)
        if not top_rows:
            print("[WARN] phase1 has no successful run. skip phase2.")
        else:
            phase2_combos = build_phase2_combos(top_rows, delta=args.phase2_float_delta)
            print(f"[INFO] phase2 generated combinations from top-{args.top_k}: {len(phase2_combos)}")

            phase2_rows, phase2_csv = run_phase(
                args=args,
                phase_name="phase2",
                combos=phase2_combos,
                output_dir=output_dir,
                log_dir=log_dir,
                timestamp=timestamp,
            )
            all_rows.extend(phase2_rows)
            print(f"[INFO] phase2 csv: {phase2_csv}")

    merged_csv = output_dir / f"sweep_results_merged_{timestamp}.csv"
    save_phase_csv(merged_csv, all_rows)

    print(f"\n[INFO] phase1 csv: {phase1_csv}")
    print(f"[INFO] merged csv: {merged_csv}")


if __name__ == "__main__":
    main()
