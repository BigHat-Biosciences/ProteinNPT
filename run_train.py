import os,gc
import json
import argparse
import random
import numpy as np
import pandas as pd
import wandb
import torch
from pprint import pprint

from proteinnpt.proteinnpt.model import ProteinNPTModel
from proteinnpt.utils.esm.data import Alphabet
from proteinnpt.utils.model_utils import Trainer
from proteinnpt.utils.data_utils import get_dataset_from_csv_file, pnpt_spearmanr


def str2bool(v):
    if isinstance(v, list):
        return [str2bool(x) for x in v]
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def setup_config_and_paths(args):
    # Setup output paths inside the sagemaker container
    args.output_scores_location = args.output_dir + os.sep + 'model_predictions'
    args.model_location = args.output_dir + os.sep + 'checkpoints'
    os.makedirs(args.output_scores_location, exist_ok=True)
    os.makedirs(args.model_location, exist_ok=True)

    # Setup input paths inside the sagemaker container
    args.model_config_location = os.path.join(args.model_config_dir, args.model_config_name)
    args.target_config_location = os.path.join(args.target_config_dir, args.target_config_name)
    if args.model_checkpoint_dir is not None:
        args.model_checkpoint_location = os.path.join(args.model_checkpoint_dir, 'checkpoint.t7')
    
    # Model and Target Configs must exist to continue
    assert os.path.exists(args.model_config_location), f"Model config file not found at {args.model_config_location}"
    assert os.path.exists(args.target_config_location), f"Target config file not found at {args.target_config_location}"

    # Setup paths to training data
    args.assay_data_folder = args.input_dir
    args.target_processing_location = None
    if args.target_processing_filename is not None:
        args.target_processing_location = os.path.join(args.model_checkpoint_dir, args.target_processing_filename)
        
    args.train_data_location = os.path.join(args.input_dir, args.train_data_filename)
    args.eval_data_locations = [os.path.join(args.input_dir, f) for f in args.eval_data_filenames]
    args.test_data_location = os.path.join(args.input_dir, args.test_data_filename) if args.test_data_filename is not None else None

    assert os.path.exists(args.train_data_location), f"Training data file not found at {args.train_data_location}"
    assert all([os.path.exists(f) for f in args.eval_data_locations]), f"Eval data file not found at {args.eval_data_locations}"
    if args.test_data_location is not None:
        assert os.path.exists(args.test_data_location), f"Test data file not found at {args.test_data_location}"
    
    ############################# SETUP MODEL CONFIG #############################
    if args.model_config_location is not None:
        args.main_config=json.load(open(args.model_config_location))
        args_setup_from_config=set([])
        for key in args.main_config:
            if args.__dict__[key] is None:
                args.__dict__[key] = args.main_config[key]
                args_setup_from_config.add(key)
    ############################# SETUP MODEL CONFIG #############################
    
    ############################# SETUP TARGET CONFIG #############################
    args.target_config=json.load(open(f"{args.target_config_location}"))
    args.augmentation_short="none"

    # Check that all targets have a location with associated labelled data
    for _, target in enumerate(args.target_config):
        assert args.assay_data_folder is not None
        args.target_config[target]["location"] = args.assay_data_folder
        print("Location used for target {} is: {}".format(target, args.assay_data_folder))
    ############################# SETUP TARGET CONFIG #############################

    return args


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # target_names are the true targets we want to predict.
    # target_names_input also includes auxiliary labels (as used in ProteinNPT)
    target_names = [x for x in args.target_config.keys() if args.target_config[x]["in_NPT_loss"]]
    target_names_input = args.target_config.keys()
    num_targets = len(target_names)
    num_targets_input = len(target_names_input)
    
    print("We want to predict {} target(s): {}".format(num_targets, ', '.join(target_names)))
    assert num_targets == num_targets_input, "Number of targets in target_config and target_config_input do not match"

    assay_id = args.train_data_location.split(".csv")[0].split(os.sep)[-1]
    assay_file_name = args.train_data_filename
    print("Training model for assay: {}, where the test_fold index is: {}".format(assay_id, args.test_fold_index))
    
    args.save_model_checkpoint = not args.do_not_save_model_checkpoint
    args.frozen_embedding_parameters = not args.fine_tune_model_embedding_parameters
    
    effective_batch_size = args.gradient_accumulation * args.training_num_assay_sequences_per_batch_per_gpu
    print("Effective batch size is {}".format(effective_batch_size))     
    
    ############################# MODEL SETUP #############################
    alphabet = Alphabet.from_architecture("ESM-1b")
    model = ProteinNPTModel(args, alphabet)
    model_name = f"{args.run_name}_{assay_id}"
    ############################# MODEL SETUP #############################

    ############################# GET PATHS TO ASSAY DATA #############################
    assay_file_names={}
    for target in target_names_input:
        assay_file_names[target] = assay_file_name
    ############################# GET PATHS TO ASSAY DATA #############################

    ############################# GET TRAINING DATA #############################  
    if args.target_processing_location and os.path.exists(args.target_processing_location):
        target_processing = json.load(open(args.target_processing_location))
        train_data, _ = get_dataset_from_csv_file(args, args.train_data_location, args.metadata_cols, target_processing=target_processing)
    else:
        train_data, target_processing = get_dataset_from_csv_file(args, args.train_data_location, args.metadata_cols)
    
    print("############################################ TARGET PROCESSING ############################################")
    for target_name, target_config in args.target_config.items():
        assert target_name in target_processing, f"Target {target_name} not found in target processing"
        target_stats = target_processing[target_name]
        target_processing[target_name]["mask"] = np.inf
        print(f"Target: {target_name}")
        pprint(target_config)
        pprint(target_stats)
        print()
    print("############################################################################################################")
    
    # save the target processing dict to json file in output dir
    with open(args.output_dir + os.sep + 'target_processing.json', 'w') as f:
        json.dump(target_processing, f)
    
    val_datas = {}
    test_data = None
    
    for val_data_loc in args.eval_data_locations:
        filename = val_data_loc.split(".csv")[0].split(os.sep)[-1]
        val_dataset, _ = get_dataset_from_csv_file(args, val_data_loc, args.metadata_cols, target_processing=target_processing)
        val_datas[filename] = val_dataset

    if args.eval_save_on_name is not None:
        assert args.eval_save_on_name in val_datas.keys(), f"Eval save on name {args.eval_save_on_name} not found in eval data locations"
    else:
        args.eval_save_on_name = list(val_datas.keys())[0]
    
    if args.test_data_location is not None:
        test_data, _ = get_dataset_from_csv_file(args, args.test_data_location, args.metadata_cols, target_processing=target_processing)

    MSA_start_position = args.MSA_start = 1
    MSA_end_position = args.MSA_end = len(train_data[0]['mutant_mutated_seq_pairs'][1])
    args.seq_len = MSA_end_position
    args.MSA_seq_len = MSA_end_position - MSA_start_position + 1
    ############################# GET TRAINING DATA #############################

    ############################# SETUP SAMPLING EVAL FROM SEED #############################
    cg_oracle_fns = {}
    if args.eval_cg_from_seed:
        from conditional_plm.oracles import ThermoOracle, AffinityOracle
        from conditional_plm.data.capulet import get_capulet_reference_sequence

        print()
        print("SAMPLING PARAMETERS")
        print(f"Number of sampling attempts: {args.n}")
        print(f"Probabilistic AA mask: {args.proba_aa_mask}")
        print()
        
        for name, stats, cond_method in zip(target_processing.keys(), target_processing.values(), args.cond_methods):
            try:
                target_fitness_value = float(cond_method)
            except ValueError:
                assert cond_method in stats, f"Conditioning method {cond_method} not found in stats"
                target_fitness_value = stats[cond_method]
            if args.target_config[name]["standardize"]:
                normalized_cond_value = (target_fitness_value - stats['mean']) / stats['std']
            else:
                normalized_cond_value = target_fitness_value
            print(f"Target: {name}, conditioning on: {target_fitness_value} => {normalized_cond_value:.2f}")

        therm_oracle = ThermoOracle.load_default()
        aff_oracle = AffinityOracle.load_default()
        ref_seq = get_capulet_reference_sequence()
        cg_oracle_fns["tm"] = lambda samples: therm_oracle.forward(samples, ref_seq).cpu().detach().numpy()
        cg_oracle_fns["kdpe"] = lambda samples: aff_oracle.forward(samples, ref_seq).cpu().detach().numpy()
    ############################# VERIFY SAMPLING EVAL FROM SEED #############################
    
    if args.use_wandb:
        # wandb.login(key=os.getenv("WANDB_API_KEY"))   # No need for this as wandb.init() pulls the key from the environment
        combined_dict = {**vars(args), "parameter_count": sum(p.numel() for p in model.parameters()), "assay_id": assay_id }
        wandb.init(project="protnpt", config=combined_dict, name=model_name, dir=args.wandb_location, save_code=True)
    
    ############################# TRAINING #############################
    trainer = Trainer(
        model=model,
        args=args,
        train_data=train_data, 
        val_datas=val_datas,
        cg_oracle_fns=cg_oracle_fns,
        MSA_sequences=None, 
        MSA_weights=None,
        MSA_start_position=MSA_start_position,
        MSA_end_position=MSA_end_position,
        target_processing=target_processing,
        distributed_training=True if torch.cuda.device_count() > 1 else False
    )

    # Load model from checkpoint or train from scratch
    if args.load_model_checkpoint and os.path.exists(args.model_checkpoint_location):
        checkpoint = torch.load(args.model_checkpoint_location)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        trainer_final_status = {
            'total_training_steps': -1,
            'total_training_epochs': -1,
            'total_train_time': -1
        }
        model.cuda()
        model.set_device()
        print(f"Model loaded from checkpoint {args.model_checkpoint_location}")

    trainer_final_status = trainer.train()
    print('Final training step: {} | Num training epochs: {} | Total train time: {} hrs'.format(trainer_final_status['total_training_steps'], trainer_final_status['total_training_epochs'], str(trainer_final_status['total_train_time'] / 3600)))
    ############################# TRAINING #############################

    ############################# SAVE MODEL #############################
    model_save_path = args.model_location + os.sep + 'final' + os.sep + 'checkpoint.t7'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
        'final_training_step': trainer_final_status['total_training_steps'],
    }, model_save_path)
    ############################# SAVE MODEL #############################

    ############################# EVALUATION #############################
    if args.test_data_location is None:
        del model; del trainer
        gc.collect(); torch.cuda.empty_cache()
        if args.use_wandb: wandb.finish()
        return
    
    test_preds = trainer.predict(
        data=test_data,
        train_data=train_data,
    )
    test_logs = {}
    for target_name in target_names:
        preds = np.array(test_preds[f"predictions_{target_name}"])
        labels = np.array(test_preds[f"labels_{target_name}"])
        test_logs[f"pp_{target_name}_spearmanr"] = pnpt_spearmanr(preds, labels)

    if args.use_wandb:
        wandb.log(test_logs)
        wandb.finish()

    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df.to_csv(args.output_scores_location + os.sep + f"{model_name}_test_preds.csv", index=False)
    ############################# EVALUATION #############################

    # Freeing up GPU memory after training on a given fold split:
    del model; del trainer
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ProteinNPT or baseline model')
    
    ############################# SAGEMAKER PARAMETERS #############################
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.getenv("SM_MODEL_DIR"),
        help='Training output will be stored there (i.e., checkpoints and test set predictions).'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=os.getenv("SM_CHANNEL_TRAIN"),
        help='Training data will be stored there'
    )
    parser.add_argument(
        '--model_checkpoint_dir',
        type=str,
        default=os.getenv("SM_CHANNEL_MODEL_CHECKPOINT"),
        help='Directory where input model checkpoint files are stored if we want to resume training'
    )
    parser.add_argument(
        '--model_config_dir',
        type=str,
        default=os.getenv("SM_CHANNEL_MODEL_CONFIGS"),
        help='Directory where model config files are stored'
    )
    parser.add_argument(
        '--target_config_dir',
        type=str,
        default=os.getenv("SM_CHANNEL_TARGET_CONFIGS"),
        help='Directory where target config files are stored'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=os.getenv("SM_CHANNEL_RUN_NAME", "samples"),
        help='Name of the run'
    )

    parser.add_argument('--target_processing_filename', default=None, type=str, help='Name of the target processing file')
    parser.add_argument('--train_data_filename', default="datum.csv", type=str, help='Name of the training data file')
    parser.add_argument('--eval_data_filenames', default=[], type=str, nargs='+', help='Name of the evaluation data file(s)')
    parser.add_argument('--eval_save_on_name', default=None, type=str, help='Name of the evaluation data file to use for model saving')
    parser.add_argument('--test_data_filename', default=None, type=str, help='Name of the test data file (optional)')

    parser.add_argument('--metadata_cols', default=[], type=str, nargs='+', help='Columns to use as metadata')
    parser.add_argument('--model_config_name', default="model_config.json", type=str, help='Model configuration file name')
    parser.add_argument('--target_config_name', default="target_config.json", type=str, help='Target configuration file name')

    parser.add_argument('--sequence_embeddings_location', default=None, type=str, help='Actual location of sequence embeddings .h5 file')
    parser.add_argument('--sequence_embeddings_folder', default=None, required=False, type=str, help='Folder with embeddings')
    
    parser.add_argument('--embedding_model_location', default=None, type=str, help='Location of model used to embed protein sequences')
    parser.add_argument('--zero_shot_fitness_predictions_location', default=None, type=str, help='Path to zero-shot fitness predictions used as additional covariates (baselines) or auxiliary labels (ProteinNPT)')
    
    parser.add_argument('--aho_aligned', type=str2bool, nargs='?', const=True, default=False, help='Whether the Aho aligned sequences are used')
    parser.add_argument('--n', default=1000, type=int, help='Number of samples to generate')
    parser.add_argument('--cond_methods', default=["mean+1std"], type=str, nargs='+', help='Conditioning methods')
    parser.add_argument('--num_avg_mutations', default=6., type=float, help='Number of average mutations in the generated sequences')
    parser.add_argument('--target_oasis_percentile', default=None, type=float, help='Target OASIS percentile')
    parser.add_argument('--use_assay_data_as_context', type=str2bool, nargs='?', const=True, default=False, help='Whether to use assay data as context')
    parser.add_argument('--use_cc_loss', type=str2bool, nargs='?', const=True, default=True, help='Whether to use CC loss')
    parser.add_argument('--use_directionality_loss', type=str2bool, nargs='?', const=True, default=True, help='Whether to use directionality loss')

    parser.add_argument('--eval_cg_from_seed', type=str2bool, nargs='?', const=True, default=True, help='Whether to evaluate from seed')
    parser.add_argument('--eval_num_random_training_sequences', default=0, type=int, help='Number of random training sequences to be leveraged at inference time')
    parser.add_argument('--eval_num_closest_oasis_training_sequences', default=0, type=int, help='Number of most human like training sequences to be leveraged at inference time')
    parser.add_argument('--eval_num_closest_aligned_sequences', default=0, type=int, help='Number of closest aligned sequences to the target sequence to be leveraged at inference time')
    parser.add_argument('--eval_num_closest_fitness_training_sequences', default=0, type=int, help='Number of closest training sequences to the target sequence by fitness to be leveraged at inference time')
    parser.add_argument('--eval_num_training_sequences_per_batch_per_gpu', default=None, type=int, help='Number of sequences from training (with label) at inference time [ProteinNPT only]')

    parser.add_argument('--target_seq', default=None, type=str, help='WT sequence mutated in the assay')
    parser.add_argument('--target_seq_mutable_mask', default=None, type=str2bool, nargs='+', help='Mask of mutable positions in the target sequence')
    parser.add_argument('--target_seq_cdr_mask', default=None, type=str, nargs='+', help='Mask of CDR positions in the target sequence')
    ############################# SAGEMAKER PARAMETERS #############################
    
    # Data parameters
    parser.add_argument('--wandb_location', default="wandb", type=str, help='Wandb directory where metadata is stored')
    parser.add_argument('--augmentation', default=None, type=str, help='Type of augmentation used ["None","zero_shot_fitness_predictions_covariate" or "zero_shot_fitness_predictions_auxiliary_labels"]. Note that default value is set in each model config files')
    parser.add_argument('--fold_variable_name', default="train_test_split", type=str, help='Name of the fold variable in the processed assay files')
    parser.add_argument('--test_fold_index', default=-1, type=int, help='Index of fold to test performance on [If "-1" is provided, we will train on all seed splits sequentially]')
    parser.add_argument('--use_validation_set', type=str2bool, nargs='?', const=True, default=False, help='Whether to use a validation set during training [If yes, we will stop training based on CV loss and patience param. Train until the end otherwise]')
    parser.add_argument('--num_data_loaders_workers', default=0, type=int, help='Number of workers to use to fetch and load data in memory')
    
    # Model parameters
    parser.add_argument('--model_type', default=None, type=str, help='Model type')
    parser.add_argument('--model_name_suffix', default=None, type=str, help='Suffix to reference model')
    parser.add_argument('--aa_embeddings', default=None, type=str, help='Type of protein sequence embedding [MSA_Transformer|Tranception|ESM1v|ESM2|Linear_embedding]')
    parser.add_argument('--long_sequences_slicing_method', default='center', type=str, help='Method to slice long sequences [rolling, center, left]. We do not slice OHE input')
    parser.add_argument('--max_positions', default=None, type=int, help='Maximum context length')
    parser.add_argument('--embed_dim', default=None, type=int, help='Embedding dimension')
    parser.add_argument('--ffn_embed_dim', default=None, type=int, help='Feedforward embedding dimension')
    parser.add_argument('--attention_heads', default=None, type=int, help='Number of attention heads')
    parser.add_argument('--conv_kernel_size', default=None, type=int, help='Size of convolutional kernel')
    parser.add_argument('--weight_decay', default=None, type=float, help='Weight decay to apply to network weights during training')
    parser.add_argument('--dropout', default=None, type=float, help='Dropout')
    parser.add_argument('--attention_dropout', default=None, type=float, help='Attention dropout')
    parser.add_argument('--activation_dropout', default=None, type=float, help='Activation dropout')
    parser.add_argument('--num_protein_npt_layers', default=None, type=int, help='Number of ProteinNPT layers [ProteinNPT only]')
    parser.add_argument('--target_prediction_head', default=None, type=str, help='Target prediction head type [AA_embeddings_mean_pooled, One_hot_encoding]')
    parser.add_argument('--target_prediction_model', default=None, type=str, help='Target prediction head model type [linear | MLP | CNN]')
    
    # Training & Eval parameters
    parser.add_argument('--num_total_training_steps', default=None, type=int, help='Number of total training steps')
    parser.add_argument('--num_logging_training_steps', default=None, type=int, help='Number of steps between 2 consecutive training loss logging')
    parser.add_argument('--do_not_save_model_checkpoint', type=str2bool, nargs='?', const=True, default=False, help='Whether to save model checkpoint')
    parser.add_argument('--load_model_checkpoint', type=str2bool, nargs='?', const=True, default=False, help='Whether to load model checkpoint')
    parser.add_argument('--num_saving_training_steps', default=None, type=int, help='Number of steps between 2 consecutive model checkpoint saving')
    parser.add_argument('--num_eval_steps', default=None, type=int, help='Number of steps between 2 consecutive evaluations on validation set')
    parser.add_argument('--num_warmup_steps', default=None, type=int, help='Number of training steps for lr warmup')
    parser.add_argument('--gradient_accumulation', default=None, type=int, help='Number of gradient accumulation steps (ie., number of forward & bwd passes per gradient optim. step)')
    parser.add_argument('--training_num_assay_sequences_per_batch_per_gpu', default=None, type=int, help='Number of assay sequences (with labels) to be leveraged during training per device')
    parser.add_argument('--eval_num_sequences_to_score_per_batch_per_gpu', default=None, type=int, help='Number of sequences to score (no label) at inference time')
    parser.add_argument('--eval_training_sequences_sampling_method', default=None, type=str, help='How to sample training points (with label) at inference time [ProteinNPT only]')
    parser.add_argument('--indel_mode', type=str2bool, nargs='?', const=True, default=False, help='indel mode')
    parser.add_argument('--seed', default=None, type=int, help='Random seed used during training')
    parser.add_argument('--num_MSA_sequences_per_training_instance', default=None, type=int, help='Number of MSA sequences to be leveraged during training')
    parser.add_argument('--num_MSA_sequences_per_eval_instance', default=None, type=int, help='Number of MSA sequences to be leveraged at evaluation time')
    parser.add_argument('--max_tokens_per_msa', default=2**14, type=int, help='Used during inference to batch attention computations in a single forward pass. This allows increased input sizes with less memory.')
    parser.add_argument('--early_stopping_patience', default=None, type=int, help='Number of consecutive evals for which the loss has to not go below the min value to call early stopping (if None, no early stopping)')
    parser.add_argument('--max_learning_rate', default=3e-4, type=float, help='Max learning rate after warmup')
    parser.add_argument('--min_learning_rate', default=1e-5, type=float, help='Min learning rate post warmup and cosine decline')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='Beta1 value in AdamW optimizer')
    parser.add_argument('--adam_beta2', default=0.999, type=float, help='Beta1 value in AdamW optimizer')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Term added to the denominator to improve numerical stability in AdamW')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='Label smoothing parameter in the MLM loss')
    parser.add_argument('--grad_norm_clip', default=1.0, type=float, help='Maximum gradient value above which we do gradient clipping')
    parser.add_argument('--fine_tune_model_embedding_parameters', type=str2bool, nargs='?', const=True, default=False, help='Whether to fine tune the model providing protein sequence embeddings')
    parser.add_argument('--training_fp16', type=str2bool, nargs='?', const=True, default=False, help='Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.')
    parser.add_argument('--use_wandb', type=str2bool, nargs='?', const=True, default=False, help='Whether to log runs in wandb')
    
    # No reference file
    parser.add_argument('--MSA_start', default=None, type=int, help='Index of first AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Index of last AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    
    args = parser.parse_args()
    setup_config_and_paths(args)
    
    ############################# SETUP SEED SEQUENCE #############################
    os.environ["PARTNER"] = "capulet"
    os.environ["DEPLOYMENT_ENVIRONMENT"] = "prod"
    from bh.biocore.sequences.constructs import ConstructSvc
    from conditional_plm.oracles import ThermoOracle, AffinityOracle
    from conditional_plm.data.humanness import biophi_v_humanness, biophi_v_humannesses, DEFAULT_MIN_PERCENT_SUBJECTS
    from conditional_plm.data.capulet import (
        get_capulet_mutable_cdr_mask,
        get_capulet_cdr_mask,
        get_aho_aligned_sequence,
        get_aho_aligned_mutable_mask,
        get_aho_aligned_cdr_mask,
        get_capulet_reference_sequence
    )

    if args.target_seq is None:
        seed_seq = ConstructSvc.get_by_name(args.seed_construct).get_part_aa_sequence()
    else:
        seed_seq = args.target_seq.replace("-", "")
    
    ref_seq = get_capulet_reference_sequence()

    therm_oracle = ThermoOracle.load_default()
    aff_oracle = AffinityOracle.load_default()
    
    if args.aho_aligned:
        args.target_seq = get_aho_aligned_sequence(seed_seq, "-")
        args.target_seq_mutable_mask = get_aho_aligned_mutable_mask(args.target_seq)
        args.target_seq_cdr_mask = get_aho_aligned_cdr_mask(args.target_seq)
    else:
        args.target_seq = seed_seq
        args.target_seq_mutable_mask = get_capulet_mutable_cdr_mask(args.target_seq)
        args.target_seq_cdr_mask = get_capulet_cdr_mask(args.target_seq)
    
    clean_target_seq = args.target_seq.replace('-', '')
    args.proba_aa_mask = args.num_avg_mutations / len(clean_target_seq)
    assert len(args.target_seq) == len(args.target_seq_mutable_mask), \
        "Target sequence, mutable mask and CDR mask must have the same length"

    seed_tm = therm_oracle.forward([clean_target_seq], ref_seq).cpu().detach().item()
    seed_kdpe = aff_oracle.forward([clean_target_seq], ref_seq).cpu().detach().item()
    seed_oasis = biophi_v_humanness(clean_target_seq).get_oasis_percentile(DEFAULT_MIN_PERCENT_SUBJECTS / 100)

    args.MSA_start = 1
    args.MSA_end = len(args.target_seq)

    print(f"Seed Construct: {args.target_seq}")
    print(f"Seed TM: {seed_tm}")
    print(f"Seed KDPE: {seed_kdpe}")
    print(f"Seed OASIS percentile: {seed_oasis}")

    ############################# START TRAINING #############################
    main(args)