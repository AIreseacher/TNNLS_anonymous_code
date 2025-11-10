#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------
# Train ResNet101 Teacher model on each dataset with multiple seeds
#   - Merge pretrain + train subsets
#   - Automatically resume from last.pth if available
#   - Save checkpoints/logs under /nas/unas15/chentao/TNNLS/checkpoints/{dataset}/Teacher101/seed_{seed}
# ----------------------------------------------------------

PY=python
JSON_DIR="/nas/unas15/chentao/TNNLS/json"

# ---------------------- Dataset list ----------------------
# DATASETS=("Breast" "Chaoyang" "Kidney")
DATASETS=("Brain")

# ---------------------- Batch size per dataset ----------------------
declare -A BATCH_SIZES=(
  ["Brain"]=32
)

# ---------------------- Common hyper-parameters ----------------------
SEEDS=(142 33 6888 95)
EPOCHS=100
LR=1e-3
WEIGHT_DECAY=1e-3
NUM_WORKERS=4
LOG_MODE="w"    # r: train+val,  w: train+val+test,  b: both

# ==========================================================
# Main loop
# ==========================================================
for DS in "${DATASETS[@]}"; do
  SRC_JSON="${JSON_DIR}/${DS}.json"
  if [[ ! -f "${SRC_JSON}" ]]; then
    echo "[WARN] Skip ${DS}: JSON not found -> ${SRC_JSON}"
    continue
  fi

  for SEED in "${SEEDS[@]}"; do
    # 动态设置 TMP_ROOT（与 save_dir 保持一致）
    TMP_ROOT="/nas/unas15/chentao/TNNLS/checkpoints/${DS}/Teacher101/seed_${SEED}"
    mkdir -p "${TMP_ROOT}"

    MERGED_JSON="${TMP_ROOT}/${DS}_merged_train.json"

    # ---------------- Merge pretrain + train ----------------
    echo "[INFO] Merging JSON for dataset ${DS} (seed ${SEED}) ..."
    DS="${DS}" SRC_JSON="${SRC_JSON}" MERGED_JSON="${MERGED_JSON}" \
    ${PY} - <<'PY'
import os, json
SRC_JSON = os.environ["SRC_JSON"]
OUT_JSON = os.environ["MERGED_JSON"]
DS       = os.environ["DS"]

with open(SRC_JSON, "r") as f:
    raw = json.load(f)

def pick(raw, key):
    if isinstance(raw, dict):
        if key in raw: return raw[key]
        for k,v in raw.items():
            if isinstance(v, dict) and k.lower()==DS.lower():
                for kk,vv in v.items():
                    if kk.lower()==key.lower(): return vv
        for k in list(raw.keys()):
            if k.lower()==key.lower(): return raw[k]
    elif isinstance(raw, list):
        return raw
    return []

pre = pick(raw, "pretrain")
trn = pick(raw, "train")

def norm(x):
    if isinstance(x, (list,tuple)) and len(x)>=2:
        return {"name": str(x[0]), "label": int(x[1])}
    if isinstance(x, dict):
        pkey = next((k for k in ["name","image","img","path","file","file_path","filepath","image_path","im","fname"] if k in x), None)
        lkey = next((k for k in ["label","target","cls","class","y","category_id"] if k in x), None)
        if pkey is None or lkey is None: return None
        d = dict(x); d["name"] = str(d.pop(pkey)); d["label"] = int(d.pop(lkey))
        return d
    return None

merged_train = []
for s in (pre or []) + (trn or []):
    ns = norm(s)
    if ns: merged_train.append(ns)

def grab_val(raw):
    for k in ["val","valid","validation"]:
        v = pick(raw, k)
        if v: return v
    return None

out = {"train": merged_train}
v = grab_val(raw); t = pick(raw, "test")
if v: out["val"] = v
if t: out["test"] = t

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"[MERGED] {OUT_JSON} | train={len(merged_train)} val={len(out.get('val',[]))} test={len(out.get('test',[]))}")
PY

    # ---------------- Training / Resume logic ----------------
    BS=${BATCH_SIZES[$DS]:-128}
    RUN_DIR="${TMP_ROOT}"
    LAST_PTH="${RUN_DIR}/last.pth"

    if [[ -f "${LAST_PTH}" ]]; then
      echo "[INFO] Found checkpoint -> ${LAST_PTH}"
      RESUME_FLAG="--resume ${LAST_PTH}"
    else
      echo "[INFO] No checkpoint found. Start fresh training for ${DS} (seed ${SEED})."
      RESUME_FLAG=""
    fi

    echo "[RUN] Dataset=${DS} | Seed=${SEED} | BatchSize=${BS}"
    ${PY} teacher101_train_eval.py \
        --json_path "${MERGED_JSON}" \
        --dataset "${DS}" \
        --save_dir "${RUN_DIR}" \
        --seed "${SEED}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --batch_size "${BS}" \
        --num_workers "${NUM_WORKERS}" \
        --log_mode "${LOG_MODE}" \
        ${RESUME_FLAG}
  done
done

echo "[DONE] All Teacher101 trainings finished. Results saved under /nas/unas15/chentao/TNNLS/checkpoints/{dataset}/Teacher101/seed_{seed}"
