#!/bin/bash

export SM_CHANNEL_MODEL_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/model_configs"
export SM_CHANNEL_TARGET_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/target_configs"

# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_3ct_tm"
# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet_382_3ct_random_mut_hsa-display-ml_helper-tm-otf-0_0"

# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_joint_oas_3ct_tm"
# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/oas_and_capulet_382_3ct_tm_aho_aligned_merged"

export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/oas_and_disp_final"
export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/kdpe_tm_oasis_imputed_sc"

# COND_METHODS="max min mask mask mask"
TARGET_CONFIG_NAME=fitness_tm_tagg_kdpe_oasis.json
AHO_ALIGNED=True

# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_5662"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_6440"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_7275"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_6304"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_6327"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_5682"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_6279"

COND_VALUES="min P25 P50 P75 max seed"
SEED_CONSTRUCTS="vhh-capulet-001_0719_no-tag_5662 vhh-capulet-001_0719_no-tag_6440 vhh-capulet-001_0719_no-tag_7275 vhh-capulet-001_0719_no-tag_6304 vhh-capulet-001_0719_no-tag_6327 vhh-capulet-001_0719_no-tag_5682 vhh-capulet-001_0719_no-tag_6279"

TARGET_OASIS=1.0
NUM_CLOSEST_OASIS_TRAIN_SEQS=0

NUM_AVG_MUTATIONS=6
NUM_RANDOM_TRAIN_SEQS=0
NUM_CLOSEST_FITNESS_TRAIN_SEQS=1000
NUM_EVAL_TRAIN_SEQS=1000
N=1000


for SEED_CONSTRUCT in ${SEED_CONSTRUCTS}; do
    for COND_VALUE in ${COND_VALUES}; do

        METHOD="kdpe_tm_oasis_imputed_sc"
        SAVE_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/samples/${SEED_CONSTRUCT}/protnpt/${METHOD}"
        COND_METHODS="mask mask ${COND_VALUE} min mask"
        
        python run_sample.py \
            --use_validation_set True \
            --use_wandb False \
            --assay_data_location train.csv \
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
            --run_name "${COND_METHODS}_num_avg_mut_${NUM_AVG_MUTATIONS}_batch_${NUM_EVAL_TRAIN_SEQS}_human_${NUM_CLOSEST_FITNESS_TRAIN_SEQS}_fitness" \
            --cond_methods ${COND_METHODS} \
            --aho_aligned ${AHO_ALIGNED} \
            --seed_construct ${SEED_CONSTRUCT} \
            --seed 42

    done
done
