source config.sh  # To get $proteinnpt_data_path
# conda activate protnpt
#source $proteinnpt_data_path/proteinnpt_env/bin/activate # Uncomment if using python venv instead of conda env

export assay_data_location="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/ProteinNPT_data/data/fitness/substitutions_multiples/AMFR_HUMAN_Tsuboyama_2023_4G3O.csv"
export MSA_location="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/ProteinNPT_data/data/MSA/MSA_files/AMFR_HUMAN_2023-08-07_b04.a2m"
export target_seq="YFQGQLNAMAHQIQEMFPQVPYHLVLQDLQLTRSVEITTDNILEGRI"

echo $assay_data_location
echo $MSA_location
echo $target_seq

python -m pdb pipeline.py \
    --proteinnpt_data_location ${proteinnpt_data_path} \
    --assay_data_location ${assay_data_location} \
    --MSA_location ${MSA_location} \
    --target_seq ${target_seq}
