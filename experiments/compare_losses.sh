#!/bin/bash
# bash experiments/compare_losses.sh
#
# Two-phase comparison so the anchor (task 0) is IDENTICAL across all 3 cases.
#
# Phase A: train task 0 once with no extra losses (mu_proto=0, mu_attention=0).
#          Save model checkpoint + snapshot_task0.pt to
#          outputs/compare_losses/_shared/.
#
# Phase B: for each loss config, copy Phase A's task-0 checkpoint and
#          snapshot_task0.pt into the case dir, then run with --overwrite 0
#          so the trainer LOADS task 0 (skips training) and only trains
#          task 1 with the case's loss config.
#
#   task old      = task 0  (anchor, same for all cases)
#   task current  = task 1  (different per case)
#
#   attention_0   = softmax(q_0 @ pk_0^T) @ pv_0
#   attention_0*  = softmax(q_0 @ pk_1^T) @ pv_1   (computed per case)
#
#   Cases:
#     1. no_loss     : mu_proto=0,    mu_attention=0
#     2. proto_only  : mu_proto=1e-5, mu_attention=0
#     3. attn_only   : mu_proto=0,    mu_attention=1e-3

cd /home/server2/cuong/softmope

OUT_BASE=outputs/compare_losses
SHARED_DIR=$OUT_BASE/_shared
mkdir -p $OUT_BASE

############################################################
# Phase A: shared task 0
############################################################
if [ ! -f "$SHARED_DIR/snapshots/snapshot_task0.pt" ]; then
    echo "===== Phase A: shared task 0 training ====="
    python -u run.py --config configs/cifar-100_prompt_vapt.yaml \
        --gpuid 0 --repeat 1 --overwrite 1 \
        --learner_type prompt --learner_name SoftMoEPrompt \
        --seeds 0 \
        --crct_epochs 50 --ca_batch_size_ratio 1 \
        --max_task 1 \
        --prompt_param 50 5 1e-5 0 1e-6 1.0 0.1 0 \
        --log_dir $SHARED_DIR \
        --adaptive_pred
else
    echo "===== Phase A already done, skipping ====="
fi

############################################################
# Phase B: per-case task 1 training
############################################################
run_case() {
    local case_name=$1
    local prompt_params=$2

    local case_dir=$OUT_BASE/$case_name
    mkdir -p $case_dir/snapshots

    # Seed the case dir with shared task 0 state
    if [ -d "$SHARED_DIR/models" ]; then
        rm -rf $case_dir/models
        cp -r $SHARED_DIR/models $case_dir/
    fi
    cp -f $SHARED_DIR/snapshots/snapshot_task0.pt $case_dir/snapshots/
    # Remove any stale task-1 snapshot
    rm -f $case_dir/snapshots/snapshot_task1.pt

    python -u run.py --config configs/cifar-100_prompt_vapt.yaml \
        --gpuid 0 --repeat 1 --overwrite 0 \
        --learner_type prompt --learner_name SoftMoEPrompt \
        --seeds 0 \
        --crct_epochs 50 --ca_batch_size_ratio 1 \
        --max_task 2 \
        --prompt_param $prompt_params \
        --log_dir $case_dir \
        --adaptive_pred
}

echo "===== Phase B [1/3]: no_loss ====="
run_case "no_loss"    "50 5 1e-5 0 1e-6 1.0 0.1 0"

echo "===== Phase B [2/3]: proto_only ====="
run_case "proto_only" "50 5 1e-5 1e-5 1e-6 1.0 0.1 0"

echo "===== Phase B [3/3]: attn_only ====="
run_case "attn_only"  "50 5 1e-5 0 1e-6 1.0 0.1 1e-3"

############################################################
# Visualization
############################################################
echo "===== Visualizing attention drift (shared anchor + shared PCA) ====="
python experiments/visualize_attention_drift.py
