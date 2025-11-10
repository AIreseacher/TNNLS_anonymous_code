#!/usr/bin/env bash
# train.sh iterate over all datasets (json basenames) and YAML seeds
set -euo pipefail

# ---------- Defaults ----------
CONFIG="./configs/Train.yaml"
JSON_DIR="./json"
GPUS="0"
OVERRIDES=()
EPOCHS=""
BATCH_SIZE=""
NUM_WORKERS=""
LOG_MODE=""       # optional; if empty, use YAML's output.log_mode
DATASETS_CSV=""   # optional: comma-separated list to override auto-discovery

# ---------- Parse flags ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)        CONFIG="$2"; shift 2 ;;
    -j|--json_dir)      JSON_DIR="$2"; shift 2 ;;
    -g|--gpus)          GPUS="${2:-0}"; shift 2 ;;
    -o|--override)      OVERRIDES+=("$2"); shift 2 ;;
    -e|--epochs)        EPOCHS="$2"; shift 2 ;;
    -b|--batch_size)    BATCH_SIZE="$2"; shift 2 ;;
    -w|--num_workers)   NUM_WORKERS="$2"; shift 2 ;;
    -m|--log_mode)      LOG_MODE="$2"; shift 2 ;;
    -D|--datasets)      DATASETS_CSV="$2"; shift 2 ;;
    -M|--method)        METHOD="$2"; shift 2 ;;
    scope|sra_off|sra_teacher_only|sra_label_only|sra_fixed_alpha|sra_no_ontology|wogcr|cagrad|pcgrad)
        METHOD="$1"; shift 1 ;;

    -h|--help)
      echo "Usage: bash train.sh [options]
  -c, --config         YAML path (default: ${CONFIG})
  -j, --json_dir       JSON folder (default: ${JSON_DIR})
  -g, --gpus           CUDA_VISIBLE_DEVICES (default: ${GPUS})
  -o, --override       KEY=VALUE (repeatable)
  -e, --epochs         -> train.epochs
  -b, --batch_size     -> data.batch_size
  -w, --num_workers    -> data.num_workers
  -m, --log_mode       r/w/b (optional; if omitted, use YAML)
  -D, --datasets       comma-separated datasets; if omitted, auto list *.json
"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;

  esac
done

# ---------- Read teacher/student from YAML (robust) ----------
read -r TEACHER STUDENT <<<"$(
python3 - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r')) or {}
m = cfg.get('model', {}) or {}
teacher = m.get('backbone_teacher') or 'UnknownTeacher'
student = m.get('backbone_student') or 'UnknownStudent'
print(teacher, student)
PY
)"
TEACHER_STUDENT="${TEACHER}_${STUDENT}"

# ---------- Read seeds from YAML ----------
SEEDS_CSV="$(
python3 - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r')) or {}

def pick_seeds(c):
    for key in ('seeds',):
        for path in ([], ['train'], ['output'], ['data']):
            d = c
            for k in path:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            seq = d.get(key)
            if isinstance(seq, (list, tuple)) and seq:
                try: return [int(x) for x in seq]
                except: pass
    for key in ('seed',):
        for path in ([], ['train'], ['output'], ['data']):
            d = c
            for k in path:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            if 'seed' in d:
                try: return [int(d['seed'])]
                except: pass
    return [42]

print(",".join(map(str, pick_seeds(cfg))))
PY
)"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS_CSV}"
echo "[INFO] Seeds from YAML: ${SEEDS_CSV}"

# ---------- Discover datasets if not provided ----------
if [[ -z "${DATASETS_CSV}" ]]; then
  DATASETS_CSV="$(
  python3 - "${JSON_DIR}" <<'PY'
import sys, os, glob
json_dir = sys.argv[1]
names = []
for p in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
    base = os.path.splitext(os.path.basename(p))[0]
    names.append(base)
print(",".join(names))
PY
  )"
fi
IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
echo "[INFO] Datasets to run: ${DATASETS_CSV}"

# ---------- Select launcher ----------
export CUDA_VISIBLE_DEVICES="${GPUS}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NP=${#GPU_ARR[@]}
if [[ "${NP}" -gt 1 ]]; then LAUNCHER=(torchrun --nproc_per_node="${NP}"); else LAUNCHER=(python); fi

# ---------- dataset-specific roots from YAML (optional) ----------
read -r ROOTS_JSON <<<"$(
python3 - "${CONFIG}" <<'PY'
import sys, yaml, json
cfg = yaml.safe_load(open(sys.argv[1], 'r')) or {}
roots = (cfg.get('data', {}) or {}).get('roots', {}) or {}
print(json.dumps(roots))
PY
)"

# ---------- Define teacher-student pairs (unchanged) ----------
TEACHER_STUDENT_PAIRS=(
  "ResNet101 ResNet18"
  "ViT-B ResNet18"
  "WideResNet101 ResNet18"
  "ResNet101 ShuffleNetV2"
)

# ---------- SRA variants ----------
# Keep the loop structure but narrow to 'full' so overall logic doesn't change.
SRA_VARIANTS=("full")
FIXED_ALPHA=0.5

# ---------- Loop over datasets ----------
for DS in "${DATASETS[@]}"; do
  DS_ROOT="$(
  python3 - <<PY
import json
roots = json.loads('''${ROOTS_JSON}''')
print(roots.get('${DS}', ''))
PY
  )"

  # loop teacher-student pairs
  for PAIR in "${TEACHER_STUDENT_PAIRS[@]}"; do
    set -- $PAIR
    TEACHER="$1"; STUDENT="$2"
    TEACHER_STUDENT="${TEACHER}_${STUDENT}"

    # pin pair to tmp config (unchanged)
    TMP_CONFIG="./configs/tmp_${TEACHER}_${STUDENT}.yaml"
    python3 - "$CONFIG" "$TEACHER" "$STUDENT" "$TMP_CONFIG" <<'PY'
import sys, yaml
base_cfg, teacher, student, out = sys.argv[1:]
cfg = yaml.safe_load(open(base_cfg, 'r')) or {}
cfg.setdefault('model', {})
cfg['model']['backbone_teacher'] = teacher
cfg['model']['backbone_student'] = student
with open(out, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    # re-read seeds from tmp (unchanged)
    SEEDS_CSV_TMP="$(
    python3 - "${TMP_CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r')) or {}

def pick_seeds(c):
    for key in ('seeds',):
        for path in ([], ['train'], ['output'], ['data']):
            d = c
            for k in path:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            seq = d.get(key)
            if isinstance(seq, (list, tuple)) and seq:
                try: return [int(x) for x in seq]
                except: pass
    for key in ('seed',):
        for path in ([], ['train'], ['output'], ['data']):
            d = c
            for k in path:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            if 'seed' in d:
                try: return [int(d['seed'])]
                except: pass
    return [42]

print(",".join(map(str, pick_seeds(cfg))))
PY
    )"
    IFS=',' read -r -a SEEDS_THIS <<< "${SEEDS_CSV_TMP}"
    echo "[INFO] ${DS} -> ${TEACHER_STUDENT} Seeds: ${SEEDS_CSV_TMP}"

    # loop seeds
    # ---------- All method variants (SRA+GCR ablations unified) ----------
# METHOD_VARIANTS=(
#   "scope"
#   "sra_off"
#   "sra_teacher_only"
#   "sra_label_only"
#   "sra_fixed_alpha"
#   "sra_no_ontology"
#   "wogcr"
#   "cagrad"
#   "pcgrad"
# )

METHOD_VARIANTS=(
  "scope"
)

# ---------- Loop over method variants (outer to seed) ----------
for METHOD_DIR in "${METHOD_VARIANTS[@]}"; do

  # Map to SRA variant (only for SRA-related)
  case "${METHOD_DIR}" in
    scope)              SRA_MODE="full";  GCR_MODE="gcr"     ;;
    sra_off)            SRA_MODE="off";   GCR_MODE="gcr"     ;;
    sra_teacher_only)   SRA_MODE="teacher_only"; GCR_MODE="gcr" ;;
    sra_label_only)     SRA_MODE="label_only";  GCR_MODE="gcr" ;;
    sra_fixed_alpha)    SRA_MODE="fixed_alpha"; GCR_MODE="gcr" ;;
    sra_no_ontology)    SRA_MODE="no_ontology"; GCR_MODE="gcr" ;;
    wogcr)              SRA_MODE="full";  GCR_MODE="sum"     ;;
    cagrad)             SRA_MODE="full";  GCR_MODE="cagrad"  ;;
    pcgrad)             SRA_MODE="full";  GCR_MODE="pcgrad"  ;;
    *) echo "Unknown METHOD: ${METHOD_DIR}" >&2; exit 1 ;;
  esac

  for SEED in "${SEEDS_THIS[@]}"; do

    RUN_DIR_BASE="checkpoints/${DS}/${TEACHER_STUDENT}/${METHOD_DIR}"
    SAVE_DIR="${RUN_DIR_BASE}/seed_${SEED}"
    mkdir -p "${SAVE_DIR}"
    LAST_PTH="${SAVE_DIR}/last.pth"

    # --- Check resume ---
    if [[ -f "${LAST_PTH}" ]]; then
        LAST_EPOCH=$(python3 - <<PY
import torch; ckpt = torch.load("${LAST_PTH}", map_location="cpu")
print(ckpt.get("epoch", -1))
PY
)
        TARGET_EPOCHS=$(python3 - <<PY
import yaml
with open("${TMP_CONFIG}", "r") as f:
    y = yaml.safe_load(f)
print(y.get("train", {}).get("epochs", 0))
PY
)
        if [[ "${LAST_EPOCH}" -ge "${TARGET_EPOCHS}" ]]; then
          echo "[SKIP] ${DS} ${TEACHER_STUDENT} METH=${METHOD_DIR} seed=${SEED}: finished (${LAST_EPOCH}/${TARGET_EPOCHS})"
          continue
        else
          echo "[RESUME] ${DS} ${TEACHER_STUDENT} METH=${METHOD_DIR} seed=${SEED}: resume at ${LAST_EPOCH}/${TARGET_EPOCHS}"
        fi
    fi

    # --- Construct overrides ---
    THIS_OVR=("${OVERRIDES[@]}")
    [[ -n "${EPOCHS}" ]]      && THIS_OVR+=("train.epochs=${EPOCHS}")
    [[ -n "${BATCH_SIZE}" ]]  && THIS_OVR+=("data.batch_size=${BATCH_SIZE}")
    [[ -n "${NUM_WORKERS}" ]] && THIS_OVR+=("data.num_workers=${NUM_WORKERS}")
    THIS_OVR+=("data.dataset=${DS}")
    THIS_OVR+=("output.save_dir=${SAVE_DIR}")
    [[ -n "${DS_ROOT}" ]] && THIS_OVR+=("data.root=${DS_ROOT}")

    # --- Add SRA config ---
    THIS_OVR+=("scope.with_relations=true")
    THIS_OVR+=("scope.enable_sra=true")
    THIS_OVR+=("scope.sra.variant=${SRA_MODE}")
    [[ "${SRA_MODE}" == "fixed_alpha" ]] && THIS_OVR+=("scope.sra.alpha=${FIXED_ALPHA}")
    THIS_OVR+=("scope.sra.alpha_mode=entropy")

    # --- Add GCR config ---
    THIS_OVR+=("gcr.variant=${GCR_MODE}")
    THIS_OVR+=("gcr.log_cos=true")
    THIS_OVR+=("gcr.sample_epochs=0.1,0.5,0.9")
    THIS_OVR+=("gcr.sample_batches=16")

    # --- Final command ---
    CMD=("${LAUNCHER[@]}" main.py
      --config "${TMP_CONFIG}"
      --json_dir "${JSON_DIR}"
      --seed "${SEED}"
    )
    [[ -n "${LOG_MODE}" ]] && CMD+=(--log_mode "${LOG_MODE}")
    for kv in "${THIS_OVR[@]}"; do CMD+=(--override "${kv}"); done

    echo "== Launch =="
    echo "Dataset     : ${DS}"
    echo "Pair        : ${TEACHER_STUDENT}"
    echo "Method      : ${METHOD_DIR} (gcr.variant=${GCR_MODE}, sra.variant=${SRA_MODE})"
    echo "Seed        : ${SEED}"
    echo "Save dir    : ${SAVE_DIR}"
    echo "============"
    "${CMD[@]}"

  done # seeds
done   # method variants


  done        # pairs
done          # datasets
