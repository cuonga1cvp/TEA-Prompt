# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R

# save directory
# OUTDIR_S=outputs/${DATASET}/5-task
OUTDIR=outputs/${DATASET}/10-task
# OUTDIR_L=outputs/${DATASET}/20-task

# training settings
GPUID='0'
REPEAT=1
OVERWRITE=1

###############################################################

# process inputs
[ -n "$OUTDIR_S" ] && mkdir -p $OUTDIR_S
mkdir -p $OUTDIR
[ -n "$OUTDIR_L" ] && mkdir -p $OUTDIR_L



# SMoPE
#
# prompt parameter args:
#    arg 1 = prompt length, equal to 2 * N_p, where N_p is the number of prompt experts
#    arg 2 = K, number of experts to use for each input
#    arg 3 = alpha_router
#    arg 4 = alpha_proto
#    arg 5 = epsilon

# --- 5-task ------
# python -u run.py --config configs/imnet-r_prompt_short_vapt.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name VAPTPrompt \
#     --prompt_param 50 5 1e-4 1e-5 0.7 --seeds 0 1 2 3 4 \
#     --crct_epochs 60 --ca_batch_size_ratio 2 \
#     --log_dir ${OUTDIR_S}/vapt-prompt
# sleep 10

# --- 10-task ------ 
python -u run.py --config configs/imnet-r_prompt_vapt.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name VAPTPrompt \
    --prompt_param 50 5 5e-5 1e-4 0.5 --seeds 0 \
    --crct_epochs 50 --ca_batch_size_ratio 6 \
    --log_dir ${OUTDIR}/vapt-prompt
sleep 10

# --- 20-task ------
# python -u run.py --config configs/imnet-r_prompt_long_vapt.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name VAPTPrompt \
#     --prompt_param 50 5 1e-4 5e-5 0.3 --seeds 0 1 2 3 4 \
#     --crct_epochs 50 --ca_batch_size_ratio 4 \
#     --log_dir ${OUTDIR_L}/vapt-prompt



# VQ-Prompt
#
# prompt parameter args:
#    arg 1 = prompt component pool size, default equal to task number
#    arg 2 = prompt length, default 8
#    arg 3 = temperature

# --- 5-task ------
# python -u run.py --config configs/imnet-r_prompt_short_vq.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name VQPrompt \
#         --prompt_param 10 8 1 --seeds 0 1 2 3 4\
#         --log_dir ${OUTDIR_S}/vq-prompt 
# sleep 10

# # --- 10-task ------  
# python -u run.py --config configs/imnet-r_prompt_vq.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name VQPrompt \
#         --prompt_param 10 8 1 --seeds 0 1 2 3 4\
#         --log_dir ${OUTDIR}/vq-prompt 
# sleep 10

# # --- 20-task ------
# python -u run.py --config configs/imnet-r_prompt_long_vq.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name VQPrompt \
#         --prompt_param 10 8 1 --seeds 0 1 2 3 4\
#         --log_dir ${OUTDIR_L}/vq-prompt 