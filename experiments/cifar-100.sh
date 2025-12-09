# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100

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

python -u run.py --config configs/cifar-100_prompt_vapt.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name VAPTPrompt \
    --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 3 4 \
    --crct_epochs 50 --ca_batch_size_ratio 1 \
    --log_dir ${OUTDIR}/vapt-prompt


# VQ-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size, default equal to task number
#    arg 2 = prompt length, default 8
#    arg 3 = temperature

python -u run.py --config configs/cifar-100_prompt_vq.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name VQPrompt \
    --prompt_param 10 8 1 --seeds 0 1 2 3 4\
    --log_dir ${OUTDIR}/vq-prompt
sleep 10
