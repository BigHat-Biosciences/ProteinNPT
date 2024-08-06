#!/bin/bash

export SM_CHANNEL_MODEL_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/model_configs"
export SM_CHANNEL_TARGET_CONFIGS="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/ProteinNPT_data/target_configs"

# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_3ct_tm"
# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet_382_3ct_random_mut_hsa-display-ml_helper-tm-otf-0_0"

# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/protnpt_joint_oas_3ct_tm"
# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/oas_and_capulet_382_3ct_tm_aho_aligned_merged"

# export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/oas_and_disp_final"
# export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/protnpt/kdpe_tm_oasis_imputed_sc"

export SM_CHANNEL_TRAIN="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/cfps_ft"
export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/ft_kdpe_tm_oasis_imputed_sc_on_cfps_2nd/checkpoints/final"

COND_METHODS="1.1 min mask mask mask"
TARGET_CONFIG_NAME=fitness_tm_tagg_kdpe_oasis.json
AHO_ALIGNED=True

# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_5662"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_6440"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_7275"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_6304"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_6327"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_5682"
# SEED_CONSTRUCT="vhh-capulet-001_0719_no-tag_6279"

# COND_VALUES="min P25 P50 P75 max seed"
# SEED_CONSTRUCTS="vhh-capulet-001_0719_no-tag_5662 vhh-capulet-001_0719_no-tag_6440 vhh-capulet-001_0719_no-tag_7275 vhh-capulet-001_0719_no-tag_6304 vhh-capulet-001_0719_no-tag_6327 vhh-capulet-001_0719_no-tag_5682 vhh-capulet-001_0719_no-tag_6279"

TARGET_OASIS=1.0
NUM_CLOSEST_OASIS_TRAIN_SEQS=0

NUM_AVG_MUTATIONS=6
NUM_RANDOM_TRAIN_SEQS=0
NUM_CLOSEST_FITNESS_TRAIN_SEQS=1000
NUM_EVAL_TRAIN_SEQS=1000
N=1000

METHOD="ft_kdpe_tm_oasis_imputed_sc_on_cfps"
SAVE_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/samples/test_ood_kdpe"
# SAVE_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/samples_local/${SEED_CONSTRUCT}/protnpt/${METHOD}"
# COND_METHODS="mask mask mask mask ${COND_VALUE}"

# KDPE SEEDS at 1.078
KD_SEED1=KVQLLESGGGVVQPGNSLRLSCAASGFTFRSFGMSWVRQAPGKGPEWVSSISGSGMDTLYAKPVKGRFTISRDNAKTTLYLQMNSLRPEDTAVYYCTIGGSLTRSSQGTLVTVSS
KD_SEED2=RVQLVESGGGLVQPGGSLRLSCAASGFTFSSFGMSWVRQAPGKGMEWVSAISGSGYDTLYADKVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCTIGGSIRRSSQGTLVTVSS
KD_SEED3=SVQLLESGGGVVQPGNSLRLSCAASGFTFSSFGMSWVRQAPGKGMEWVSAISGSGYDTLYADKVKGRFTISRDNSKNTLYLQMNSLRPEDTAVYYCTIGGSLRRSSQGTLVTVSS
KD_SEED4=KVQLVESGGGVVQPGGSLRLSCAASGFSFRNFGMSWVRQAPGKGPEWVSAISGSGKDTLYASPVKGRFIISRDNAKNTLYLQMNSLRPEDTAVYYCTIGGSLTPSSQGTLVTVSS

# TM SEEDS at 64.985, 64.980, 64.980, 64.976
TM_SEED1=KVQLLESGGGVVQPGNSLRLSCAASGFTFRSFGMSWVRQAPGKGPEWVSSISGSGMDTLYAKPVKGRFTISRDNAKTTLYLQMNSLRPEDTAVYYCTIGGSLTRSSQGTLVTVSS
TM_SEED2=RVQLVESGGGLVQPGGSLRLSCAASGFTFSSFGMSWVRQAPGKGMEWVSAISGSGYDTLYADKVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCTIGGSIRRSSQGTLVTVSS
TM_SEED3=SVQLLESGGGVVQPGNSLRLSCAASGFTFSSFGMSWVRQAPGKGMEWVSAISGSGYDTLYADKVKGRFTISRDNSKNTLYLQMNSLRPEDTAVYYCTIGGSLRRSSQGTLVTVSS
TM_SEED4=KVQLVESGGGVVQPGGSLRLSCAASGFSFRNFGMSWVRQAPGKGPEWVSAISGSGKDTLYASPVKGRFIISRDNAKNTLYLQMNSLRPEDTAVYYCTIGGSLTPSSQGTLVTVSS

CURRENT_SEED=$KD_SEED1
CURRENT_SEED_NAME="KD_SEED1"

python run_sample.py \
    --use_validation_set True \
    --use_wandb False \
    --assay_data_location test.csv \
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
    --cond_methods ${COND_METHODS} \
    --aho_aligned ${AHO_ALIGNED} \
    --target_seq $CURRENT_SEED \
    --run_name "${COND_METHODS}_num_avg_mut_${NUM_AVG_MUTATIONS}_${CURRENT_SEED_NAME}" \
    --seed 42 \
    --target_processing_filename target_processing.json
    # --seed_construct ${SEED_CONSTRUCT} \
