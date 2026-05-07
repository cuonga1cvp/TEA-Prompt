# bash experiments/cub-200.sh
# experiment settings
DATASET=cub-200

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0'
REPEAT=5
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR


# SMoPE
#
# prompt parameter args:
#    arg 1 = prompt length, equal to 2 * N_p, where N_p is the number of prompt experts
#    arg 2 = K, number of experts to use for each input
#    arg 3 = alpha_router
#    arg 4 = alpha_proto
#    arg 5 = epsilon

python -u run.py --config configs/cub-200_prompt_vapt.yaml \
    --gpuid 0 --repeat 1 --overwrite 1 \
    --learner_type prompt --learner_name SoftMoEPrompt \
    --prompt_param 50 5 1e-5 5e-5 1e-6 1.0 0.1 1e-4 --seeds 0 \
    --crct_epochs 50 --ca_batch_size_ratio 1 \
    --pretrained_weight sup21k \
    --log_dir outputs/CUB200/10-task/soft-moe-attn-50-5-1e-5-5e-5-1.0-0.1-1e-4 \
    --adaptive_pred


for mu in 0 10 1e-1 1e-2; do
    python -u run.py --config configs/cub-200_prompt_vapt.yaml \
        --gpuid 0 --repeat 1 --overwrite 1 \
        --learner_type prompt --learner_name SoftMoEPrompt \
        --prompt_param 50 5 1e-5 1e-5 1e-6 1.0 0.1 $mu --seeds 0 \
        --crct_epochs 50 --ca_batch_size_ratio 1 \
        --pretrained_weight sup21k \
        --log_dir outputs/CUB200/10-task/soft-moe-attn-${mu} 
done



# VQ-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size, default equal to task number
#    arg 2 = prompt length, default 8
#    arg 3 = temperature

python -u run.py --config configs/cub-200_prompt_vq.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name VQPrompt \
    --prompt_param 10 8 5 --ca_batch_size_ratio 2 --seeds 0 12 24 36 48 \
    --pretrained_weigh sup21k \
    --log_dir ${OUTDIR}/vq-prompt
