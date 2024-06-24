# source config.sh  # To get $proteinnpt_data_path
# conda activate protnpt
# source $proteinnpt_data_path/proteinnpt_env/bin/activate # Uncomment if using python venv instead of conda env

# data_dir=/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet-vhh-capulet_new-qc-thermostability-ml-sequences-0_21
data_dir=/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet_382_3ct_random_mut_hsa-display-ml_helper-tm-otf-0_0

export WANDB_PROJECT=protnpt
export assay_data_location=$data_dir/capulet_382_3ct.csv
export MSA_location=$data_dir/aligned_sequences.a2m
export target_seq=$(cat "$data_dir/seed_sequence.txt")

echo $assay_data_location
echo $MSA_location
echo $target_seq

python -m pdb pipeline.py \
    --proteinnpt_data_location ${proteinnpt_data_path} \
    --assay_data_location ${assay_data_location} \
    --MSA_location ${MSA_location} \
    --target_seq ${target_seq} \
    --fold_variable_name train_test_split \
    --test_fold_index 2 \
    --run_local
    # --model_config_location ../proteinnpt/proteinnpt/model_configs/PNPT_final.json \
    # --model_name_suffix 3ct_tm \
