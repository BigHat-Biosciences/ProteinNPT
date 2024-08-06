#!/bin/bash

export SM_CHANNEL_MODEL_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/model_configs"
export SM_CHANNEL_TARGET_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/target_configs"

# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_3ct_tm"
# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet_382_3ct_random_mut_hsa-display-ml_helper-tm-otf-0_0"

# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_joint_oas_3ct_tm"
# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/oas_and_capulet_382_3ct_tm_aho_aligned_merged"

# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_joint_oas_3ct_tm_imputed"
# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/ft_kdpe_tm_oasis_imputed_sc_on_cfps/checkpoints/final"
export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/ft_kdpe_tm_oasis_imputed_sc_on_cfps_2nd/checkpoints/final"

CONTEXT_DATA="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/cfps_ft/train.csv"
EVAL_DATA="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/cfps_ft/test.csv"
SAVE_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/preds/protnpt"

# TARGET_CONFIG_NAME=fitness.json
TARGET_CONFIG_NAME=fitness_tm_tagg_kdpe_oasis.json
RUN_NAME=predict_ood_cfps_with_iid_cfps_context
AHO_ALIGNED=True

NUM_CLOSEST_ALIGNED_SEQS=0
NUM_EVAL_TRAIN_SEQS=1000

python -m pdb run_predict.py \
    --use_wandb False \
    --target_config_name ${TARGET_CONFIG_NAME} \
    --model_config_name PNPT_final.json \
    --aa_embeddings Linear_embedding \
    --eval_num_training_sequences_per_batch_per_gpu ${NUM_EVAL_TRAIN_SEQS} \
    --eval_num_sequences_to_score_per_batch_per_gpu 100 \
    --load_model_checkpoint True \
    --save_dir ${SAVE_DIR} \
    --eval_num_closest_aligned_sequences ${NUM_CLOSEST_ALIGNED_SEQS} \
    --run_name "${RUN_NAME}_batch_${NUM_EVAL_TRAIN_SEQS}" \
    --aho_aligned ${AHO_ALIGNED} \
    --assay_data_location ${EVAL_DATA} \
    --use_assay_data_as_context True \
    --target_processing_filename target_processing.json \
    --context_data_location ${CONTEXT_DATA} \