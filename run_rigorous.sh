#!/usr/bin/env bash
set -uo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND"' ERR

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH=.

DATA_ROOT="./data/zerowaste-f"
RAW_PREFIX="train/data/"
UNLABELED_JSON="$DATA_ROOT/unlabeled/labels.json"
TEST_JSON="$DATA_ROOT/test/labels.json"
SOTA_WEIGHT="weights/final_sota/best_swa_0.545.pth"
TRAIN_BASE_CFG="external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste_finetune.py"

echo "===== Precheck ====="
for f in "$TRAIN_BASE_CFG" "$SOTA_WEIGHT" "$TEST_JSON"; do
  [ -f "$f" ] || { echo "[FATAL] missing file: $f"; exit 1; }
done
mkdir -p "$DATA_ROOT/unlabeled" data/pseudo_labels semi_sup/scripts

# 如果无 unlabeled/labels.json，则用 train/labels.json 复制一份并清空 annotations
if [ ! -f "$UNLABELED_JSON" ]; then
  if [ ! -f "$DATA_ROOT/train/labels.json" ]; then
    echo "[FATAL] missing $UNLABELED_JSON and $DATA_ROOT/train/labels.json"
    exit 1
  fi
  python - <<'PY'
import json
from pathlib import Path
src = Path('data/zerowaste-f/train/labels.json')
dst = Path('data/zerowaste-f/unlabeled/labels.json')
data = json.loads(src.read_text(encoding='utf-8'))
data['annotations'] = []
if 'licenses' not in data: data['licenses'] = []
if 'info' not in data: data['info'] = {}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(data), encoding='utf-8')
print('[OK] created unlabeled labels:', dst)
print('images=', len(data.get('images', [])), 'annotations=', len(data.get('annotations', [])))
PY
fi

echo "===== Step 1: generate unlabeled infer cfg ====="
cat > generate_unlabeled_infer_cfg.py <<PY_EOF
from mmengine.config import Config
cfg = Config.fromfile('$TRAIN_BASE_CFG')
cfg.test_dataloader.dataset.ann_file = 'unlabeled/labels.json'
cfg.test_dataloader.dataset.data_prefix = dict(img='$RAW_PREFIX')
cfg.dump('unlabeled_infer_cfg.py')
print('[OK] unlabeled_infer_cfg.py')
PY_EOF
python generate_unlabeled_infer_cfg.py || true

echo "===== Step 2: infer on unlabeled ====="
python external_modules/mmdetection/tools/test.py \
  unlabeled_infer_cfg.py "$SOTA_WEIGHT" \
  --cfg-options model.test_cfg.rcnn.score_thr=0.01 test_evaluator.outfile_prefix=./data/pseudo_labels/unlabeled_raw || true

python - <<'PY'
from pathlib import Path
ps = sorted(Path('data/pseudo_labels').glob('unlabeled_raw*'))
print('[CHECK] unlabeled_raw files:')
for p in ps: print(' -', p)
PY

echo "===== Step 3: make pseudo coco ====="
cat > semi_sup/scripts/make_pseudo_coco.py <<'PY_EOF'
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_gt_json', required=True)
    parser.add_argument('--pred_json', required=True)
    parser.add_argument('--out_json', required=True)
    parser.add_argument('--thr_metal', type=float, default=0.20)
    parser.add_argument('--thr_others', type=float, default=0.30)
    args = parser.parse_args()

    with open(args.test_gt_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    with open(args.pred_json, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    cat_id_to_name = {c['id']: c['name'] for c in coco.get('categories', [])}
    pseudo_anns = []
    ann_id = 1
    filtered = 0
    kept_cls = {}

    for p in preds:
        cid = p['category_id']
        cname = cat_id_to_name.get(cid, '')
        thr = args.thr_metal if cname == 'metal' else args.thr_others
        if p.get('score', 0.0) >= thr:
            x, y, w, h = p['bbox']
            pseudo_anns.append({
                "id": ann_id,
                "image_id": p['image_id'],
                "category_id": cid,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0
            })
            kept_cls[cname] = kept_cls.get(cname, 0) + 1
            ann_id += 1
        else:
            filtered += 1

    coco['annotations'] = pseudo_anns
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print("===== 正在提取高纯度伪标签 =====")
    print(f"[OK] 过滤掉的低分噪点数: {filtered}")
    print("[OK] 成功提取的高质量伪标签:")
    for k,v in kept_cls.items():
        print(f"  - {k}: {v} 个")
    print(f"[OK] 伪标签集已保存至: {args.out_json}")

if __name__ == '__main__':
    main()
PY_EOF

python semi_sup/scripts/make_pseudo_coco.py \
  --test_gt_json "$UNLABELED_JSON" \
  --pred_json ./data/pseudo_labels/unlabeled_raw.bbox.json \
  --out_json "$DATA_ROOT/unlabeled_pseudo.json" \
  --thr_metal 0.20 \
  --thr_others 0.30 || true

echo "===== Step 4: generate rigorous finetune cfg ====="
cat > generate_rigorous_finetune_cfg_v3.py <<PY_EOF
from mmengine.config import Config
cfg = Config.fromfile('$TRAIN_BASE_CFG')

orig = cfg.train_dataloader.dataset
pseudo = cfg.train_dataloader.dataset.copy()

inner = pseudo
while isinstance(inner, dict) and 'dataset' in inner:
    inner = inner['dataset']

inner['ann_file'] = 'unlabeled_pseudo.json'
inner['data_prefix'] = dict(img='$RAW_PREFIX')

cfg.train_dataloader.dataset = dict(
    type='ConcatDataset',
    datasets=[orig, pseudo]
)

cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1
if 'optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper:
    cfg.optim_wrapper.optimizer.lr = 1e-5

cfg.load_from = '$SOTA_WEIGHT'
cfg.work_dir = './work_dirs/rigorous_pseudo_finetune_v3'
cfg.dump('rigorous_finetune_cfg_v3.py')
print('[OK] rigorous_finetune_cfg_v3.py')
PY_EOF

python generate_rigorous_finetune_cfg_v3.py || true

echo "===== Step 5: train 1 epoch ====="
python external_modules/mmdetection/tools/train.py rigorous_finetune_cfg_v3.py || true

echo "===== Step 6: blind test + postproc ====="
if [ -f "./work_dirs/rigorous_pseudo_finetune_v3/epoch_1.pth" ]; then
  python external_modules/mmdetection/tools/test.py \
    external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py \
    ./work_dirs/rigorous_pseudo_finetune_v3/epoch_1.pth \
    --cfg-options model.test_cfg.rcnn.score_thr=0.01 test_evaluator.outfile_prefix=./data/pseudo_labels/final_test_raw || true

  python semi_sup/scripts/apply_best_postproc.py \
    --gt_json "$TEST_JSON" \
    --pred_json ./data/pseudo_labels/final_test_raw.bbox.json \
    --out_json ./data/pseudo_labels/final_test_post.bbox.json \
    --default_thr 0.0 --default_nms_iou 1.0 --metal_nms_iou 0.72 || true
else
  echo "[WARN] epoch_1.pth not found, skip blind test."
fi

echo "[DONE] pipeline finished."
