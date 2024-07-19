#!/bin/bash

export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/test_protnpt"
export SM_CHANNEL_MODEL_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/model_configs"
export SM_CHANNEL_TARGET_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/target_configs"

export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/oas_and_disp_final"

AHO_ALIGNED=True
RUN_NAME=train_on_kdpe_tm_oasis
TARGET_CONFIG=fitness_tm_tagg_kdpe_oasis.json

COND_METHODS="max min max min max"
NUM_AVG_MUTATIONS=6
NUM_EVAL_TRAIN_SEQS=1000
N=500

python run_train.py \
    --run_name ${RUN_NAME} \
    --train_data_filename train.csv \
    --eval_data_filenames val_disp.csv val_oas.csv \
    --eval_save_on_name val_disp \
    --use_wandb False \
    --use_validation_set True \
    --target_config_name ${TARGET_CONFIG} \
    --model_config_name PNPT_final.json \
    --aa_embeddings Linear_embedding \
    --fine_tune_model_embedding_parameters True \
    --eval_cg_from_seed True \
    --training_num_assay_sequences_per_batch_per_gpu 225 \
    --eval_num_training_sequences_per_batch_per_gpu ${NUM_EVAL_TRAIN_SEQS} \
    --eval_num_sequences_to_score_per_batch_per_gpu 100 \
    --num_avg_mutations ${NUM_AVG_MUTATIONS} \
    --cond_methods ${COND_METHODS} \
    --n ${N} \
    --num_eval_steps 100 \
    --num_logging_training_steps 10 \
    --num_total_training_steps 1000 \
    --num_saving_training_steps 250 \
    --aho_aligned ${AHO_ALIGNED} \
    --use_cc_loss false \
    --use_directionality_loss false
