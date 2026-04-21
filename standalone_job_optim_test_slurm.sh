#!/bin/bash

# --- DEFAULTS ---
ENV_TYPE=""
ENV_VALUE=""
CONFIG_FILE=""
RESUME_STUDY_FROM_TRIAL_NUMBER=""
BACKEND="nccl"  # Default backend
CPU_CORES=""    # Optional override for number of CPU cores

# Count how many GPUs are in CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES not set, defaulting PPN=1"
    PPN=1
else
    IFS=',' read -ra gpu_array <<< "$CUDA_VISIBLE_DEVICES"
    PPN=${#gpu_array[@]}
fi

echo "Setting PPN=$PPN based on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# --- PARSE ARGS ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_TYPE_VALUE="$2"
            if [[ "$ENV_TYPE_VALUE" == "none" ]]; then
                ENV_TYPE="none"
                ENV_VALUE=""
            else
                IFS=':' read -r ENV_TYPE ENV_VALUE <<< "$ENV_TYPE_VALUE"
                if [[ "$ENV_TYPE" != "conda" && "$ENV_TYPE" != "docker" ]]; then
                    echo "Error: --env must be one of: conda:<env_name>, docker:<image_name>, none"
                    exit 1
                fi
            fi
            shift 2
            ;;
        --config_file_path)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --resume_study_from_trial_number)
            RESUME_STUDY_FROM_TRIAL_NUMBER="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --cpu_cores)
            CPU_CORES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$ENV_TYPE" ]]; then
    echo "Error: --env is required. Use one of: conda:<env_name>, docker:<image_name>, none"
    exit 1
fi

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config <config_file.yaml> is required"
    exit 1
fi

JOB_NAME=$(basename "$CONFIG_FILE" .yaml)
WORKDIR=$(pwd)
JOB_LOG_DIR=$WORKDIR/standalone_job_log/$JOB_NAME
mkdir -p "$JOB_LOG_DIR"

MASTER_PORT=$(for port in $(seq 8000 35000); do nc -z localhost "$port" 2>/dev/null || { echo "$port"; break; }; done)
RUN_TIMESTAMP=$(date +"%Y%m%d%H%M%S")

RESUME_ARG=""
if [[ -n "$RESUME_STUDY_FROM_TRIAL_NUMBER" ]]; then
    RESUME_ARG="--resume_study_from_trial_number=$RESUME_STUDY_FROM_TRIAL_NUMBER"
fi

CPU_ARG=""
if [[ -n "$CPU_CORES" ]]; then
    export OMP_NUM_THREADS=$CPU_CORES
    export MKL_NUM_THREADS=$CPU_CORES
    export OPENBLAS_NUM_THREADS=$CPU_CORES
    export NUMEXPR_NUM_THREADS=$CPU_CORES
    CPU_ARG="--cpu_cores=$CPU_CORES"
    echo "[INFO] Setting thread env vars and passing --cpu_cores=$CPU_CORES"
fi

# --- RUN ---
if [ "$ENV_TYPE" == "conda" ]; then
    echo "Running inside conda env: $ENV_VALUE"
    bash -c "conda run -n \"$ENV_VALUE\" bash -c 'export MASTER_PORT=$MASTER_PORT && python -u -m torch.distributed.run --nproc-per-node=$PPN --master-port=$MASTER_PORT -m icodemix_lang_ident_classifier.cli.lang_ident_classifier_api --config=$CONFIG_FILE $CPU_ARG --backend=$BACKEND --run_timestamp=$RUN_TIMESTAMP $RESUME_ARG >> \"$JOB_LOG_DIR/RUN_$RUN_TIMESTAMP.out\" 2>&1'"
elif [ "$ENV_TYPE" == "docker" ]; then
    echo "Running inside Docker image: $ENV_VALUE"
    MY_UID=$(id -u)
    MY_GID=$(id -g)
    MY_UNAME=$(whoami)

    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Running with GPU support."
        RUNTIME="--runtime=nvidia"
        GPU_FLAG="--gpus \"device=$CUDA_VISIBLE_DEVICES\""
    else
        echo "No NVIDIA GPU detected. Running without GPU support."
        RUNTIME=""
        GPU_FLAG=""
    fi

    eval docker run --rm $RUNTIME $GPU_FLAG \
        -v "$WORKDIR:/app" \
        --ipc=host \
        -w /app \
        -e UID=$MY_UID \
        -e GID=$MY_GID \
        -e USERNAME=$MY_UNAME \
        -e HF_CACHE=/app/.cache \
        "$ENV_VALUE" \
        python -u -m torch.distributed.run \
            --nproc-per-node $PPN \
            --master-port $MASTER_PORT \
            -m icodemix_lang_ident_classifier.cli.lang_ident_classifier_api \
            --config "/app/$CONFIG_FILE" \
            $CPU_ARG \
            --backend $BACKEND \
            --run_timestamp $RUN_TIMESTAMP \
            $RESUME_ARG \
        >> $JOB_LOG_DIR/RUN_$RUN_TIMESTAMP.out 2>&1
elif [ "$ENV_TYPE" == "none" ]; then
    echo "Running directly on host (no env)"
    python -u -m torch.distributed.run \
        --nproc-per-node $PPN \
        --master-port $MASTER_PORT \
        -m icodemix_lang_ident_classifier.cli.lang_ident_classifier_api \
        --config $CONFIG_FILE \
        $CPU_ARG \
        --backend $BACKEND \
        --run_timestamp $RUN_TIMESTAMP \
        $RESUME_ARG \
        >> $JOB_LOG_DIR/RUN_$RUN_TIMESTAMP.out 2>&1
else
    echo "Unknown environment type: $ENV_TYPE"
    exit 1
fi
