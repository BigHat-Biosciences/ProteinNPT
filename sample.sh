#!/bin/bash

export SM_CHANNEL_MODEL_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/model_configs"
export SM_CHANNEL_TARGET_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/target_configs"

# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_3ct_tm"
# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet_382_3ct_random_mut_hsa-display-ml_helper-tm-otf-0_0"

# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_joint_oas_3ct_tm"
# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/oas_and_capulet_382_3ct_tm_aho_aligned_merged"

export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_joint_oas_3ct_tm_imputed"
export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/oas_and_capulet_382_3ct_tm_tagg_imputed_aho_aligned"

SAVE_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/samples/vhh-capulet-001_0719_no-tag_5662/protnpt"

COND_METHODS="max min"
TARGET_CONFIG_NAME=fitness_tm_mean_std.json
METHOD="display"
TARGET_OASIS=1.0
NUM_CLOSEST_OASIS_TRAIN_SEQS=0
AHO_ALIGNED=True

# COND_METHODS="max"
# TARGET_CONFIG_NAME=fitness.json

NUM_AVG_MUTATIONS=36
NUM_RANDOM_TRAIN_SEQS=0
NUM_CLOSEST_FITNESS_TRAIN_SEQS=0
NUM_EVAL_TRAIN_SEQS=1000
N=1000

python run_sample.py \
    --test_fold_index 2 \
    --fold_variable_name train_test_split \
    --use_validation_set True \
    --use_wandb False \
    --assay_data_location datum.csv \
    --target_config_name ${TARGET_CONFIG_NAME} \
    --model_config_name PNPT_final.json \
    --aa_embeddings Linear_embedding \
    --eval_num_training_sequences_per_batch_per_gpu ${NUM_EVAL_TRAIN_SEQS} \
    --eval_num_sequences_to_score_per_batch_per_gpu 100 \
    --load_model_checkpoint True \
    --save_dir ${SAVE_DIR} \
    --n ${N} \
    --target_oasis_percentile ${TARGET_OASIS} \
    --eval_num_random_training_sequences ${NUM_RANDOM_TRAIN_SEQS} \
    --eval_num_closest_oasis_training_sequences ${NUM_CLOSEST_OASIS_TRAIN_SEQS} \
    --eval_num_closest_fitness_training_sequences ${NUM_CLOSEST_FITNESS_TRAIN_SEQS} \
    --num_avg_mutations ${NUM_AVG_MUTATIONS} \
    --run_name "${METHOD}_${COND_METHODS}_num_avg_mut_${NUM_AVG_MUTATIONS}_batch_${NUM_EVAL_TRAIN_SEQS}_by_${NUM_CLOSEST_OASIS_TRAIN_SEQS}_human_${NUM_CLOSEST_FITNESS_TRAIN_SEQS}_fitness" \
    --cond_methods ${COND_METHODS} \
    --aho_aligned ${AHO_ALIGNED} \
    # --metadata_cols oasis_percentile \