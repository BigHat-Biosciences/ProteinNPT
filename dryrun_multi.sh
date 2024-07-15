#!/bin/bash

export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/test_protnpt"
export SM_CHANNEL_MODEL_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/model_configs"
export SM_CHANNEL_TARGET_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/target_configs"
export SM_CHANNEL_MODEL_CHECKPOINT="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_joint_oas_3ct_tm_imputed"

export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/finetune_on_cfps_tm"

AHO_ALIGNED=True
RUN_NAME=finetune_protnpt_imputed_on_cfps
TARGET_CONFIG=fitness_tm_mean_std.json

COND_METHODS="max min"
NUM_AVG_MUTATIONS=6
NUM_EVAL_TRAIN_SEQS=1000
N=500

python run_train.py \
    --run_name ${RUN_NAME} \
    --target_processing_filename target_processing.json \
    --train_data_filename cfps_gt_train.csv \
    --eval_data_filenames cfps_gt_val.csv 3ct_imp.csv oas_imp.csv \
    --eval_save_on_name cfps_gt_val \
    --test_data_filename cfps_gt_test.csv \
    --use_wandb False \
    --use_validation_set True \
    --target_config_name fitness_tm_mean_std.json \
    --model_config_name PNPT_final.json \
    --aa_embeddings Linear_embedding \
    --fine_tune_model_embedding_parameters True \
    --eval_cg_from_seed True \
    --training_num_assay_sequences_per_batch_per_gpu 425 \
    --eval_num_training_sequences_per_batch_per_gpu ${NUM_EVAL_TRAIN_SEQS} \
    --eval_num_sequences_to_score_per_batch_per_gpu 100 \
    --num_avg_mutations ${NUM_AVG_MUTATIONS} \
    --cond_methods ${COND_METHODS} \
    --n ${N} \
    --num_eval_steps 100 \
    --num_total_training_steps 1000 \
    --num_saving_training_steps 250 \
    --aho_aligned ${AHO_ALIGNED} \
    --load_model_checkpoint True \
    --max_learning_rate 0.00005 \
    --min_learning_rate 0.00001 \
