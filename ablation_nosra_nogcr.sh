#!/usr/bin/env bash
# ===============================================================
# Ablation: No SRA + No GCR (plain KD + CE, no relations, no GCR)
# - Auto resume / auto skip via checkpoints/.../last.pth
# - Layout: checkpoints/{DS}/{TEACHER_STUDENT}/nosra_nogcr/seed_{SEED}
# ===============================================================

set -euo pipefail

# ---------- Basics ----------
CONFIG="./configs/Train.yaml"
JSON_DIR="./json"
GPUS="0"
EPOCHS=""          # Optional -> train.epochs
BATCH_SIZE=""      # Optional -> data.batch_size
NUM_WORKERS=""     # Optional -> data.num_workers
LOG_MODE="w"

# Datasets to run; 
# DATASETS=("Chaoyang" "Kidney" "Breast")
DATASETS=("Brain")

# Teacher-Student pairs
TEACHER_STUDENT_PAIRS=(
  "ResNet101 ResNet18"
  # "ViT-B ResNet18"
  # "WideResNet101 ResNet18"
  # "ResNet101 ShuffleNetV2"
)

# ---------- Parse CLI ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)       CONFIG="$2"; shift 2 ;;
    -j|--json_dir)     JSON_DIR="$2"; shift 2 ;;
    -g|--gpus)         GPUS="${2:-0}"; shift 2 ;;
    -e|--epochs)       EPOCHS="$2"; shift 2 ;;
    -b|--batch_size)   BATCH_SIZE="$2"; shift 2 ;;
    -w|--num_workers)  NUM_WORKERS="$2"; shift 2 ;;
    -m|--log_mode)     LOG_MODE="$2"; shift 2 ;;
    -D|--datasets)     IFS=',' read -r -a DATASETS <<< "$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash ablation_nosra_nogcr.sh [options]
  -c, --config        YAML path (default: ${CONFIG})
  -j, --json_dir      JSON folder (default: ${JSON_DIR})
  -g, --gpus          CUDA_VISIBLE_DEVICES (default: ${GPUS})
  -e, --epochs        override train.epochs
  -b, --batch_size    override data.batch_size
  -w, --num_workers   override data.num_workers
  -m, --log_mode      r/w/b (default: ${LOG_MODE})
  -D, --datasets      comma-separated list of datasets
"; exit 0 ;;
    *) echo "[ERR] Unknown arg: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPUS}"

# ---------- Read seed list from YAML（兼容 int / list） ----------
SEEDS_CSV="$(
python3 - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r')) or {}
train = cfg.get('train', {}) or {}
seeds = train.get('seeds') or [train.get('seed', 42)]
if isinstance(seeds, int): seeds = [seeds]
print(",".join(map(str, seeds)))
PY
)"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS_CSV}"
echo "[INFO] Seeds: ${SEEDS_CSV}"

# ---------- Helper: target epochs ----------
read_target_epochs() {
  if [[ -n "${EPOCHS}" ]]; then
    echo "${EPOCHS}"
  else
    python3 - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r')) or {}
print((cfg.get('train', {}) or {}).get('epochs', 0))
PY
  fi
}
TARGET_EPOCHS="$(read_target_epochs)"
echo "[INFO] Target epochs: ${TARGET_EPOCHS}"

# ---------- Loop ----------
for DS in "${DATASETS[@]}"; do
  for PAIR in "${TEACHER_STUDENT_PAIRS[@]}"; do
    set -- $PAIR
    TEACHER="$1"; STUDENT="$2"
    TEACHER_STUDENT="${TEACHER}_${STUDENT}"

    for SEED in "${SEED_ARR[@]}"; do
      SAVE_DIR="checkpoints/${DS}/${TEACHER_STUDENT}/nosra_nogcr/seed_${SEED}"
      mkdir -p "${SAVE_DIR}"
      LAST_PTH="${SAVE_DIR}/last.pth"

      # ----- Auto resume / skip -----
      if [[ -f "${LAST_PTH}" ]]; then
        LAST_EPOCH="$(python3 - <<PY
import torch
ckpt = torch.load("${LAST_PTH}", map_location="cpu")
print(ckpt.get("epoch", -1))
PY
)"
        if [[ "${LAST_EPOCH}" -ge "${TARGET_EPOCHS}" ]]; then
          echo "[SKIP] ${DS} ${TEACHER_STUDENT} nosra_nogcr seed=${SEED}: finished (${LAST_EPOCH}/${TARGET_EPOCHS})"
          continue
        else
          echo "[RESUME] ${DS} ${TEACHER_STUDENT} nosra_nogcr seed=${SEED}: resume at ${LAST_EPOCH}/${TARGET_EPOCHS}"
        fi
      fi

      # ----- Build cmd -----
      CMD=(python main_ACC.py
        --config "${CONFIG}"
        --json_dir "${JSON_DIR}"
        --seed "${SEED}"
        --log_mode "${LOG_MODE}"
        --override "data.dataset=${DS}"
        --override "output.save_dir=${SAVE_DIR}"

        # ---- turn off SRA mechanism ----
        --override "scope.with_relations=False"
        --override "scope.enable_sra=False"

        # ---- turn off GCR mechanism ----
        --override "gcr.variant=sum"
      )

      [[ -n "${EPOCHS}" ]]      && CMD+=(--override "train.epochs=${EPOCHS}")
      [[ -n "${BATCH_SIZE}" ]]  && CMD+=(--override "data.batch_size=${BATCH_SIZE}")
      [[ -n "${NUM_WORKERS}" ]] && CMD+=(--override "data.num_workers=${NUM_WORKERS}")

      echo "==== NoSRA+NoGCR | ${DS} | ${TEACHER_STUDENT} | seed ${SEED} ===="
      "${CMD[@]}"
    done
  done
done
