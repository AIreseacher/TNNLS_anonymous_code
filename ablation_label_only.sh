#!/usr/bin/env bash
# ===============================================================
# Ablation experiment: SRA "Label-Only" across target datasets
# with auto resume / auto skip based on checkpoints/last.pth
# Datasets: Chaoyang, Kidney, Breast
# Teacher–Student pairs: same as train.sh
# ===============================================================

set -euo pipefail

# ---------- Basic configuration ----------
CONFIG="./configs/Train.yaml"      # Main YAML config file
JSON_DIR="./json"                  # JSON directory path
GPUS="0"                           # GPU device(s), e.g., "0" or "0,1"
EPOCHS=""                          # Optional override to train.epochs
BATCH_SIZE=""                      # Optional override
NUM_WORKERS=""                     # Optional override
LOG_MODE="w"                       # Log both train/val/test results
# DATASETS=("Chaoyang" "Kidney" "Breast")
DATASETS=("Brain")

# ---------- Teacher–student pairs (kept consistent with train.sh) ----------
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
      SAVE_DIR="checkpoints/${DS}/${TEACHER_STUDENT}/label_only/seed_${SEED}"
      mkdir -p "${SAVE_DIR}"
      LAST_PTH="${SAVE_DIR}/last.pth"

      # ---------- Auto resume / auto skip ----------
      if [[ -f "${LAST_PTH}" ]]; then
        LAST_EPOCH="$(python3 - <<PY
import torch
ckpt = torch.load("${LAST_PTH}", map_location="cpu")
print(ckpt.get("epoch", -1))
PY
)"
        if [[ "${LAST_EPOCH}" -ge "${TARGET_EPOCHS}" ]]; then
          echo "[SKIP] ${DS} ${TEACHER_STUDENT} label_only seed=${SEED}: finished (${LAST_EPOCH}/${TARGET_EPOCHS})"
          continue
        else
          echo "[RESUME] ${DS} ${TEACHER_STUDENT} label_only seed=${SEED}: resume at ${LAST_EPOCH}/${TARGET_EPOCHS}"
        fi
      fi

      # ---------- Build command ----------
      # In Label-Only, we keep KD and CE as usual, but SRA supervision
      # uses ONLY label/ontology relations (no teacher relations).
      CMD=(python main_ACC.py
        --config "${CONFIG}"
        --json_dir "${JSON_DIR}"
        --seed "${SEED}"
        --log_mode "${LOG_MODE}"
        --override "data.dataset=${DS}"
        --override "output.save_dir=${SAVE_DIR}"

        # SRA configuration: label-only relational supervision
        --override "scope.with_relations=True"
        --override "scope.enable_sra=True"
        --override "scope.sra.variant=label_only"
        --override "scope.sra.alpha_mode=entropy"  # harmless here; kept consistent

        # GCR configuration (on, as in full framework)
        --override "gcr.variant=gcr"
        --override "gcr.log_cos=True"
        --override "gcr.sample_epochs=0.1,0.5,0.9"
        --override "gcr.sample_batches=16"
      )

      # Optional overrides
      [[ -n "${EPOCHS}" ]]      && CMD+=(--override "train.epochs=${EPOCHS}")
      [[ -n "${BATCH_SIZE}" ]]  && CMD+=(--override "data.batch_size=${BATCH_SIZE}")
      [[ -n "${NUM_WORKERS}" ]] && CMD+=(--override "data.num_workers=${NUM_WORKERS}")

      # ---------- Run ----------
      echo "======================================================"
      echo "   SRA Label-Only Ablation (auto-resume)"
      echo "   Dataset : ${DS}"
      echo "   Pair    : ${TEACHER_STUDENT}"
      echo "   Seed    : ${SEED}"
      echo "   SaveDir : ${SAVE_DIR}"
      echo "======================================================"
      "${CMD[@]}"
    done
  done
done
