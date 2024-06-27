#!/bin/bash

export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/test_protnpt"
export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet-vhh-capulet_new-qc-thermostability-ml-sequences-0_21"
export SM_CHANNEL_MODEL_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/model_configs"
export SM_CHANNEL_TARGET_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/target_configs"

python run_train.py \
    --MSA_data_folder '' \
    --MSA_location aligned_sequences.a2m \
    --MSA_sequence_weights_filename aligned_sequences_hhfiltered_cov_75_maxid_90_minid_0.a2m \
    --MSA_weight_data_folder '' \
    --assay_data_location datum.csv \
    --embedding_model_location esm_msa1_t12_100M_UR50S \
    --fold_variable_name train_test_split \
    --model_config_name PNPT_final.json \
    --model_name_suffix testing \
    --num_total_training_steps 5 \
    --path_to_hhfilter hhfiltered \
    --sequence_embeddings_folder embeddings \
    --sequence_embeddings_location msat.h5 \
    --target_config_name fitness.json \
    --target_seq KVQLVESGGGVVQPGGSLRLSCAASGFSFRNFGMSWVRQAPGKGPEWVSAISGSGADTLYASPVKGRFIISRDNAKNTLYLQMNSLRPEDTAVYYCTIGGSLTRSSQGTLVTVSS \
    --test_fold_index 2 \
    --use_validation_set True \
    --use_wandb False