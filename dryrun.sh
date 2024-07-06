#!/bin/bash

export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/test_protnpt"
export SM_CHANNEL_MODEL_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/model_configs"
export SM_CHANNEL_TARGET_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/target_configs"

# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/examples"
export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/oas_and_capulet_382_3ct_tm_aho_aligned_merged"
SEED_SEQ=KVQLVES-GGGVVQPGGSLRLSCAASG-FSFRN-----FGMSWVRQAPGKGPEWVSAISGS---GADTLYASPVKGRFIISRDNAKNTLYLQMNSLRPEDTAVYYCTIGGS------------------------LTRSSQGTLVTVSS---

python -m pdb run_train.py \
    --test_fold_index 2 \
    --fold_variable_name train_test_split \
    --target_seq ${SEED_SEQ} \
    --use_validation_set True \
    --use_wandb False \
    --assay_data_location datum.csv \
    --target_config_name fitness.json \
    \
    --model_config_name PNPT_final.json \
    --model_name_suffix testing \
    --num_eval_steps 2 \
    --num_total_training_steps 100 \
    --aa_embeddings Linear_embedding \
    --fine_tune_model_embedding_parameters True \
    --metadata_cols sampling_weight \
    --indel_mode True \
    --eval_num_training_sequences_per_batch_per_gpu 410 \
    # --embedding_model_location esm_msa1_t12_100M_UR50S \
    # --sequence_embeddings_folder embeddings \
    # --sequence_embeddings_location msat.h5 \
    # --MSA_sequence_weights_filename aligned_sequences_hhfiltered_cov_75_maxid_90_minid_0.a2m \
    # --path_to_hhfilter hhfiltered \
