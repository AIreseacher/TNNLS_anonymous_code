#!/usr/bin/env bash
# ===============================================================
# Ablation experiment: Teacher-Only mode across multiple datasets
# with auto resume / skip based on checkpoints/last.pth
# ===============================================================

set -euo pipefail

# ---------- Basic configuration ----------
CONFIG="./configs/Train.yaml"      # Main YAML config file
JSON_DIR="./json"                  # JSON directory path
GPUS="0"                           # GPU device(s)
EPOCHS=""                          # Optional override to train.epochs
BATCH_SIZE=""                      # Optional override
NUM_WORKERS=""                     # Optional override
LOG_MODE="w"                       # r/w/b; here default to print train/val/test
# DATASETS=("Chaoyang" "Kidney" "Breast")
DATASETS=("Brain")  # Target datasets

# ---------- Define teacher–student pairs ----------
TEACHER_STUDENT_PAIRS=(
  "ResNet101 ResNet18"
  "ViT-B ResNet18"
  "WideResNet101 ResNet18"
  "ResNet101 ShuffleNetV2"
)

# ---------- Extract seed list from YAML (compatible with int/list) ----------
SEEDS_CSV="$(
python3 - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r')) or {}
train = cfg.get('train', {}) or {}
seeds = train.get('seeds')
if seeds is None:
    seeds = [train.get('seed', 42)]
elif isinstance(seeds, int):
    seeds = [seeds]
print(",".join(map(str, seeds)))
PY
)"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS_CSV}"
echo "[INFO] Seeds detected: ${SEEDS_CSV}"

# ---------- GPU setup ----------
export CUDA_VISIBLE_DEVICES="${GPUS}"

# ---------- Helper: read target epochs from YAML unless overridden ----------
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

# ---------- Loop over datasets ----------
for DS in "${DATASETS[@]}"; do
  # Loop over teacher–student pairs
  for PAIR in "${TEACHER_STUDENT_PAIRS[@]}"; do
    set -- $PAIR
    TEACHER="$1"
    STUDENT="$2"
    TEACHER_STUDENT="${TEACHER}_${STUDENT}"

    # Loop over random seeds
    for SEED in "${SEED_ARR[@]}"; do
      RUN_DIR_BASE="checkpoints/${DS}/${TEACHER_STUDENT}/teacher_only"
      SAVE_DIR="${RUN_DIR_BASE}/seed_${SEED}"
      mkdir -p "${SAVE_DIR}"
      LAST_PTH="${SAVE_DIR}/last.pth"

      # ---------- Auto resume / skip ----------
      if [[ -f "${LAST_PTH}" ]]; then
        LAST_EPOCH="$(python3 - <<PY
import torch
ckpt = torch.load("${LAST_PTH}", map_location="cpu")
print(ckpt.get("epoch", -1))
PY
)"
        if [[ "${LAST_EPOCH}" -ge "${TARGET_EPOCHS}" ]]; then
          echo "[SKIP] ${DS} ${TEACHER_STUDENT} teacher_only seed=${SEED}: finished (${LAST_EPOCH}/${TARGET_EPOCHS})"
          continue
        else
          echo "[RESUME] ${DS} ${TEACHER_STUDENT} teacher_only seed=${SEED}: resume at ${LAST_EPOCH}/${TARGET_EPOCHS}"
        fi
      fi

      # ---------- Build command ----------
      CMD=(python main_ACC.py
        --config "${CONFIG}"
        --json_dir "${JSON_DIR}"
        --seed "${SEED}"
        --log_mode "${LOG_MODE}"
        --override "data.dataset=${DS}"
        --override "output.save_dir=${SAVE_DIR}"

        # Teacher-only ablation configuration
        --override "scope.with_relations=False"
        --override "scope.enable_sra=False"
        # Only KD loss (disable CE by weighting); adjust to your loss wiring
        --override "scope.beta=0.6"
        --override "scope.gamma=1.0"
      )

      # Optional overrides
      [[ -n "${EPOCHS}" ]]      && CMD+=(--override "train.epochs=${EPOCHS}")
      [[ -n "${BATCH_SIZE}" ]]  && CMD+=(--override "data.batch_size=${BATCH_SIZE}")
      [[ -n "${NUM_WORKERS}" ]] && CMD+=(--override "data.num_workers=${NUM_WORKERS}")

      # ---------- Run ----------
      echo "======================================================"
      echo "   Teacher-Only Experiment (auto-resume)"
      echo "   Dataset : ${DS}"
      echo "   Pair    : ${TEACHER_STUDENT}"
      echo "   Seed    : ${SEED}"
      echo "   SaveDir : ${SAVE_DIR}"
      echo "======================================================"
      "${CMD[@]}"
    done
  done
done
