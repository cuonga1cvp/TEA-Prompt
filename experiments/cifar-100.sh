# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0'
REPEAT=1
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR

# soft-moe
# arg1: prompt length
# arg2: threshold rank
# arg3: alpha_router
# arg4: alpha_proto
# arg5: epsilon
# arg6: sigma alpha
# arg7: sigma tau
# python -u run.py --config configs/cifar-100_prompt_vapt.yaml \
#     --gpuid 0 --repeat 1 --overwrite 1 \
#     --learner_type prompt --learner_name SoftMoEPrompt \
#     --prompt_param 50 5 1e-5 1e-5 1e-6 1.0 0.1 1e-4 --seeds 0 \
#     --crct_epochs 50 --ca_batch_size_ratio 1 \
#     --log_dir outputs/CIFAR100/10-task/soft-moe-attn_50_5_1e-5_1e-5_1.0_0.1_1e-4 \
#     --adaptive_pred

for mu in 10; do
    python -u run.py --config configs/cifar-100_prompt_vapt.yaml \
        --gpuid 0 --repeat 1 --overwrite 1 \
        --learner_type prompt --learner_name SoftMoEPrompt \
        --prompt_param 50 5 0 0 1e-6 1.0 0.2 0 $mu 8 --seeds 0 \
        --crct_epochs 50 --ca_batch_size_ratio 1 \
        --pretrained_weight sup21k \
        --log_dir outputs/CIFAR100/10-task/soft-moe-attn-sqrt_0_0_0${mu} \
        --adaptive_pred 
done


# SMoPE
#
# prompt parameter args:
#    arg 1 = prompt length, equal to 2 * N_p, where N_p is the number of prompt experts
#    arg 2 = K, number of experts to use for each input
#    arg 3 = alpha_router
#    arg 4 = alpha_proto
#    arg 5 = epsilon

# python -u run.py --config configs/cifar-100_prompt_vapt.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name VAPTPrompt \
#     --prompt_param 50 -1 0 0 0 --seeds 0 \
#     --crct_epochs 50 --ca_batch_size_ratio 1 \
#     --log_dir ${OUTDIR}/vapt-prompt-no-topk

#baseline_smope
# python run.py \
#   --config configs/cifar-100_prompt_vapt.yaml \
#   --learner_type prompt \
#   --learner_name BaselineSMoPELearner \
#   --prompt_param 50 5 0 0 0 \
#   --log_dir outputs/cifar-100/baseline_smope


# Soft-MoE
#
# prompt parameter args:
#    arg 1 = prompt length, equal to 2 * N_p, where N_p is the number of prompt experts
#    arg 2 = threshold rank used to define s_hat (the default setup uses 5)
#    arg 3 = alpha_router
#    arg 4 = alpha_proto
#    arg 5 = epsilon
#    arg 6 = sigma alpha
#    arg 7 = sigma tau

# python -u run.py --config configs/cifar-100_prompt_vapt.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name SoftMoEPrompt \
#     --prompt_param 50 5 1e-5 1e-5 1e-6 1.0 0.1 --seeds 0 \
#     --crct_epochs 50 --ca_batch_size_ratio 1 \
#     --log_dir ${OUTDIR}/soft-moe



# VQ-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size, default equal to task number
#    arg 2 = prompt length, default 8
#    arg 3 = temperature

# python -u run.py --config configs/cifar-100_prompt_vq.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name VQPrompt \
#     --prompt_param 10 8 1 --seeds 0 1 2 3 4\
#     --log_dir ${OUTDIR}/vq-prompt
# sleep 10
