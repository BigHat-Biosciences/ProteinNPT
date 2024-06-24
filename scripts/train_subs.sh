source ./config.sh
source activate protnpt

export model_config_location=$ProteinNPT_config_location # [ProteinNPT_config_location|Embeddings_MSAT_config_location|Embeddings_Tranception_config_location|Embeddings_ESM1v_config_location|OHE_config_location|OHE_TranceptEVE_config_location]
export sequence_embeddings_folder=$MSAT_embeddings_folder # [MSAT_embeddings_folder|Tranception_embeddings_folder|ESM1v_embeddings_folder]

export fold_variable_name='train_test_split' #[fold_random_5 | fold_contiguous_5 | fold_modulo_5]
export assay_index=1 #Replace with index of desired DMS assay in the ProteinGym reference file (`utils/proteingym`)
export model_name_suffix='All_singles_final' #Give a name to the model

data_dir=/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet_382_3ct_random_mut_hsa-display-ml_helper-tm-otf-0_0

export WANDB_PROJECT=protnpt
export assay_data_location=$data_dir/capulet_382_3ct.csv
export MSA_location=$data_dir/aligned_sequences.a2m
export target_seq=$(cat "$data_dir/seed_sequence.txt")

echo $assay_data_location
echo $MSA_location
echo $target_seq

python train.py \
    --data_location $proteinnpt_data_path \
    --assay_data_location $assay_data_location \
    --model_config_location ../proteinnpt/proteinnpt/model_configs/PNPT_final.json \
    --fold_variable_name $fold_variable_name \
    --target_config_location ../proteinnpt/utils/target_configs/fitness.json \
    --training_fp16 \
    --MSA_data_folder $data_dir \
    --MSA_location $data_dir/aligned_sequences.a2m \
    --MSA_weight_data_folder $data_dir \
    --MSA_sequence_weights_filename aligned_sequences_hhfiltered_cov_75_maxid_90_minid_0.a2m \
    --target_seq $target_seq \
    --test_fold_index 2 \
    --augmentation None \
    --model_name_suffix 3ct_tm \
    --use_validation_set \
    --num_total_training_steps 50000
    # --use_wandb \
    # --sequence_embeddings_folder $ESM1v_embeddings_folder \