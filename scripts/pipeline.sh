# source config.sh  # To get $proteinnpt_data_path
# conda activate protnpt
#source $proteinnpt_data_path/proteinnpt_env/bin/activate # Uncomment if using python venv instead of conda env

data_dir=/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet-vhh-capulet_new-qc-thermostability-ml-sequences-0_21

export WANDB_PROJECT=protnpt
export assay_data_location=$data_dir/datum.csv
export MSA_location=$data_dir/aligned_sequences.a2m
export target_seq=$(cat "$data_dir/seed_sequence.txt")

echo $assay_data_location
echo $MSA_location
echo $target_seq

python pipeline.py \
    --proteinnpt_data_location ${proteinnpt_data_path} \
    --assay_data_location ${assay_data_location} \
    --MSA_location ${MSA_location} \
    --target_seq ${target_seq} \
    --fold_variable_name train_test_split \
    --run_local
