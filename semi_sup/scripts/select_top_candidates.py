import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Select top candidate parameter combos from merged sweep csv")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(Path("data") / "pseudo_labels" / "sweeps" / "merged_96_summary.csv"),
        help="Path to merged summary CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(Path("data") / "pseudo_labels" / "sweeps" / "top10_candidates_for_train.csv"),
        help="Path to output candidate CSV",
    )
    parser.add_argument("--top_n", type=int, default=10, help="Number of candidates to select")
    parser.add_argument("--mid_low_quantile", type=float, default=0.20, help="Lower quantile for annotation_count filtering")
    parser.add_argument("--mid_high_quantile", type=float, default=0.80, help="Upper quantile for annotation_count filtering")
    parser.add_argument("--min_per_tau", type=int, default=2, help="Minimum picks per tau_f for diversity")
    return parser.parse_args()


def to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def load_success_rows(input_csv: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not to_bool(r.get("generation_ok", "False")):
                continue
            try:
                r["annotation_count"] = int(r["annotation_count"])
                r["tau_f"] = float(r["tau_f"])
                r["theta"] = float(r["theta"])
                r["min_votes"] = int(r["min_votes"])
                r["soft_nms_iou"] = float(r["soft_nms_iou"])
                rows.append(r)
            except Exception:
                continue
    return rows


def select_candidates(rows: List[Dict], top_n: int, low_q: float, high_q: float, min_per_tau: int) -> List[Dict]:
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda x: x["annotation_count"])
    n = len(sorted_rows)
    lo = max(0, min(n - 1, int(n * low_q)))
    hi = max(lo + 1, min(n, int(n * high_q)))

    mid_rows = sorted_rows[lo:hi] if hi > lo else sorted_rows
    mid_rows = sorted(mid_rows, key=lambda x: x["annotation_count"], reverse=True)

    by_tau = defaultdict(list)
    for r in mid_rows:
        by_tau[r["tau_f"]].append(r)

    selected: List[Dict] = []
    used = set()

    for tau in sorted(by_tau.keys()):
        take = 0
        for r in by_tau[tau]:
            key = (r["tau_f"], r["theta"], r["min_votes"], r["soft_nms_iou"])
            if key in used:
                continue
            selected.append(r)
            used.add(key)
            take += 1
            if take >= min_per_tau:
                break

    if len(selected) < top_n:
        for r in mid_rows:
            key = (r["tau_f"], r["theta"], r["min_votes"], r["soft_nms_iou"])
            if key in used:
                continue
            selected.append(r)
            used.add(key)
            if len(selected) >= top_n:
                break

    return selected[:top_n]


def save_output(output_csv: Path, selected: List[Dict]):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tau_f",
        "theta",
        "min_votes",
        "soft_nms_iou",
        "annotation_count",
        "json_path",
        "run_dir",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in selected:
            writer.writerow({k: r.get(k, "") for k in fields})


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    rows = load_success_rows(input_csv)
    selected = select_candidates(
        rows=rows,
        top_n=args.top_n,
        low_q=args.mid_low_quantile,
        high_q=args.mid_high_quantile,
        min_per_tau=args.min_per_tau,
    )

    save_output(output_csv, selected)

    print(f"[INFO] input: {input_csv}")
    print(f"[INFO] success rows: {len(rows)}")
    print(f"[INFO] selected: {len(selected)}")
    print(f"[INFO] output: {output_csv}")

    for i, r in enumerate(selected, start=1):
        print(
            f"{i:02d}. tau_f={r['tau_f']}, theta={r['theta']}, "
            f"min_votes={r['min_votes']}, soft_nms_iou={r['soft_nms_iou']}, "
            f"annotation_count={r['annotation_count']}"
        )


if __name__ == "__main__":
    main()
