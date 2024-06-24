source ./config.sh
source activate protnpt

data_dir=/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/protnpt/capulet-vhh-capulet_new-qc-thermostability-ml-sequences-0_21

export model_location="$proteinnpt_data_path/checkpoint/base_pipeline/checkpoint-40000/checkpoint.t7"
export assay_data_location="$data_dir/datum.csv"
export embeddings_location="$proteinnpt_data_path/data/embeddings/MSA_Transformer/datum.h5"
export zero_shot_fitness_predictions_location="$proteinnpt_data_path/data/zero_shot_fitness_predictions/substitutions/datum.csv"
export fold_variable_name='train_test_split'
export test_fold_index=2
export output_scores_location="$proteinnpt_data_path/model_predictions/base_pipeline/test_split_40000_checkpoint_preds.csv"

python -m pdb eval.py \
    --model_location ${model_location} \
    --assay_data_location ${assay_data_location} \
    --embeddings_location ${embeddings_location} \
    --zero_shot_fitness_predictions_location ${zero_shot_fitness_predictions_location} \
    --fold_variable_name ${fold_variable_name} \
    --test_fold_index ${test_fold_index} \
    --output_scores_location ${output_scores_location}