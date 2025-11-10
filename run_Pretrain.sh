#!/bin/bash
# ======================================================
# Batch pretraining over multiple datasets and backbones
# Backbones: vit-b / wideresnet101 / (optional) resnet101
# JSON expected at: ${JSON_DIR}/${DATASET}.json
# Output saved at:  ${SAVE_DIR}/Pretrain_<TAG>_<DATASET>.pth
# ======================================================

set -euo pipefail

PYTHON=python
PRETRAIN_PY=Pretrain.py

# Root folders (adjust to your environment)
JSON_DIR="/nas/unas15/chentao/TNNLS/json"
SAVE_DIR="/nas/unas15/chentao/TNNLS/Pretrain"

# Datasets to loop over
DATASETS=(
  "Brain"
  "Breast"
  "Cataract"
  "Chaoyang"
  "HAM10000"
  "Kidney"
  "LUNG"
  "Myeloma"
  "PAPILA"
)

# Teacher backbones to train
BACKBONES=(
  "vit-b"
  "wideresnet101"
  # "resnet101"   # uncomment if needed
)

# Common training hyper-parameters
BATCH_SIZE=256
NUM_EPOCHS=100
LR=0.0001
EARLY_STOP_ACC=97.0
SUBSET="pretrain"
NUM_WORKERS=4

echo "=========== PRETRAIN LAUNCHER ==========="
echo "[JSON DIR ] ${JSON_DIR}"
echo "[SAVE DIR ] ${SAVE_DIR}"
echo "[BACKBONES] ${BACKBONES[*]}"
echo "[DATASETS ] ${DATASETS[*]}"
echo "========================================="

for DATASET in "${DATASETS[@]}"; do
  JSON_PATH="${JSON_DIR}/${DATASET}.json"

  if [[ ! -f "${JSON_PATH}" ]]; then
    echo "[WARN] Skip ${DATASET}: JSON not found -> ${JSON_PATH}"
    continue
  fi

  for BB in "${BACKBONES[@]}"; do
    TAG=$(echo "${BB}" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
    echo "-----------------------------------------"
    echo "[RUN] Dataset=${DATASET}  Backbone=${BB}"
    echo "[JSON] ${JSON_PATH}"
    echo "[SAVE] ${SAVE_DIR}/Pretrain_${TAG}_${DATASET}.pth"
    echo "-----------------------------------------"

    ${PYTHON} "${PRETRAIN_PY}" \
      --json_path "${JSON_PATH}" \
      --dataset "${DATASET}" \
      --subset "${SUBSET}" \
      --save_dir "${SAVE_DIR}" \
      --backbone "${BB}" \
      --batch_size "${BATCH_SIZE}" \
      --num_epochs "${NUM_EPOCHS}" \
      --learning_rate "${LR}" \
      --early_stop_acc "${EARLY_STOP_ACC}" \
      --num_workers "${NUM_WORKERS}"
  done
done
