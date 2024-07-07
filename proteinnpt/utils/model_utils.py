import math
import os
import time
import wandb
import random
import tqdm
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data_utils import collate_fn_protein_npt


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer. 
    Adapted from Huggingface Transformers library.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def get_learning_rate(training_step, num_warmup_steps=1000, num_total_training_steps=20000, max_learning_rate=3e-4, min_learning_rate=3e-5):
    """
    Cosine learning rate scheduler with warmup.
    """
    if training_step <= num_warmup_steps:
        lr = (max_learning_rate * training_step) / num_warmup_steps
    elif training_step > num_total_training_steps:
        lr=min_learning_rate
    else:
        ratio_total_steps_post_warmup = (training_step - num_warmup_steps) / (num_total_training_steps - num_warmup_steps)
        cosine_scaler = 0.5 * (1.0 + math.cos(math.pi * ratio_total_steps_post_warmup))
        lr = min_learning_rate + cosine_scaler * (max_learning_rate - min_learning_rate)
    return lr

def learning_rate_scheduler(num_warmup_steps=1000, num_total_training_steps=20000, max_learning_rate=3e-4, min_learning_rate=3e-5):
    def get_lr(training_step):
        return get_learning_rate(training_step, num_warmup_steps, num_total_training_steps, max_learning_rate, min_learning_rate)
    return get_lr

def get_reconstruction_loss_coefficient(training_step, num_total_training_steps=20000, start_MLM_coefficient=0.5, end_MLM_coefficient=0.05):
    ratio_total_steps = training_step / num_total_training_steps
    cosine_scaler = 0.5 * (1.0 + math.cos(math.pi * ratio_total_steps))
    reconstruction_loss_coeff = end_MLM_coefficient + cosine_scaler * (start_MLM_coefficient - end_MLM_coefficient)
    return reconstruction_loss_coeff

def update_lr_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def collapse_triplets(s):
    triplets = s.split(":")
    positions = {}
    for triplet in triplets:
        pos = triplet[1:-1]
        aa1, aa2 = triplet[0], triplet[-1]
        if pos in positions:
            positions[pos] = positions[pos][:-1] + aa2
        else:
            positions[pos] = aa1 + pos + aa2
    s_new = ":".join([aa for aa in positions.values()])
    return s_new

def apply_cdr_mask(sequence, cdr_mask):
    shift_indices = np.where(np.diff(cdr_mask) != 0)[0]
    assert len(shift_indices) == 6, "Sequence does not have 3 CDRs and 4 FRs!"
    shift_indices += 1
    shift_indices = np.insert(shift_indices, 0, 0)
    shift_indices = np.append(shift_indices, len(sequence))
    frs = [sequence[shift_indices[i]:shift_indices[i+1]] for i in range(0, len(shift_indices), 2)]
    cdrs = [sequence[shift_indices[i]:shift_indices[i+1]] for i in range(1, len(shift_indices)-1, 2)]
    regions = {
        "fr1": frs[0], "cdr1": cdrs[0], "fr2": frs[1], "cdr2": cdrs[1],
        "fr3": frs[2], "cdr3": cdrs[2], "fr4": frs[3]
    }
    return regions


class Trainer():
    def __init__(self, 
        model,
        args,
        train_data, 
        val_data=None,
        val_seed_data=None,
        MSA_sequences=None, 
        MSA_weights=None,
        MSA_start_position=None,
        MSA_end_position=None,
        target_processing=None,
        distributed_training=False
        ):
        self.model = model
        self.args = args
        self.train_data = train_data
        self.val_data = val_data
        self.val_seed_data = val_seed_data
        self.MSA_sequences = MSA_sequences
        self.MSA_weights = MSA_weights
        self.MSA_start_position = MSA_start_position
        self.MSA_end_position = MSA_end_position
        self.target_processing = target_processing
        self.distributed_training = distributed_training
            
    def train(self):
        """
        Returns the last value of training_step (useful in case of early stopping for isntance)
        """
        import proteinnpt
        self.model.train()
        self.model.cuda()
        self.model.set_device()

        if self.distributed_training:
            # TODO: Implement weighted distributed sampler if we really want to run in distributed mode
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data)
        elif 'sampling_weight' in self.train_data.features:
            train_sampler = torch.utils.data.WeightedRandomSampler(
                self.train_data['sampling_weight'],
                len(self.train_data['sampling_weight'])
            )
        else:
            train_sampler = None
        
        #To ensure reproducibility with seed setting
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = torch.utils.data.DataLoader(
                            dataset=self.train_data, 
                            batch_size=self.args.training_num_assay_sequences_per_batch_per_gpu, 
                            shuffle=(train_sampler is None),
                            num_workers=self.args.num_data_loaders_workers, 
                            pin_memory=True, 
                            sampler=train_sampler,
                            collate_fn=collate_fn_protein_npt,
                            worker_init_fn=seed_worker,
                            generator=g,
                        )
        optimizer = self.model.create_optimizer()
        scheduler = learning_rate_scheduler(
            num_warmup_steps=self.args.num_warmup_steps, 
            num_total_training_steps=self.args.num_total_training_steps, 
            max_learning_rate=self.args.max_learning_rate, 
            min_learning_rate=self.args.min_learning_rate
        )
        
        train_iterator = iter(train_loader)
        num_epochs = 0
        prior_log_time = time.time()
        total_train_time = 0
        log_train_total_loss = 0
        if self.model.model_type=="ProteinNPT":
            log_train_reconstruction_loss = 0
            log_train_num_masked_tokens = 0
            log_train_num_target_masked_tokens_dict = defaultdict(int)
        else:
            log_num_sequences_predicted = 0
        log_train_target_prediction_loss_dict = defaultdict(int)
        all_spearmans_eval_during_training = []
        max_average_spearman_across_targets = - math.inf
        if self.args.training_fp16: scaler = torch.cuda.amp.GradScaler()

        for training_step in tqdm.tqdm(range(1, self.args.num_total_training_steps+1)):
            optimizer.zero_grad(set_to_none=True)
            lr = scheduler(training_step)
            update_lr_optimizer(optimizer, lr)
            reconstruction_loss_coeff = get_reconstruction_loss_coefficient(training_step, num_total_training_steps=self.args.num_total_training_steps) if (self.model.model_type=="ProteinNPT" and not self.model.PNPT_no_reconstruction_error) else 0
            for gradient_accum_step in range(self.args.gradient_accumulation):
                try:
                    batch = next(train_iterator)
                except:
                    num_epochs +=1
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                if self.model.model_type=="ProteinNPT":
                    processed_batch = proteinnpt.proteinnpt.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet,
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        target_processing = self.target_processing,
                        training_sequences = None,
                        proba_target_mask = 0.15,
                        proba_aa_mask = 0.15,
                        eval_mode = False,
                        device=self.model.device,
                        indel_mode=self.args.indel_mode
                    )                    
                else:
                    processed_batch = proteinnpt.baselines.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet, 
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        device=self.model.device,
                        eval_mode=False,
                        indel_mode=self.args.indel_mode
                    )

                if self.args.augmentation=="zero_shot_fitness_predictions_covariate":
                    zero_shot_fitness_predictions = processed_batch['target_labels']['zero_shot_fitness_predictions'].view(-1,1)
                    del processed_batch['target_labels']['zero_shot_fitness_predictions']
                else:
                    zero_shot_fitness_predictions = None
            
                if self.args.training_fp16:
                    with torch.cuda.amp.autocast():
                        if self.model.model_type=="ProteinNPT":
                            output = self.model(
                                tokens=processed_batch['masked_tokens'],
                                targets=processed_batch['masked_targets'],
                                zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                                sequence_embeddings=processed_batch['sequence_embeddings']
                            )
                            total_loss, reconstruction_loss, target_prediction_loss_dict = self.model.protein_npt_loss(
                                token_predictions_logits=output['logits_protein_sequence'], 
                                token_labels=processed_batch['token_labels'], 
                                target_predictions=output['target_predictions'], 
                                target_labels=processed_batch['target_labels'], 
                                MLM_reconstruction_loss_weight=reconstruction_loss_coeff, 
                                label_smoothing=self.args.label_smoothing
                            )
                        else:
                            output = self.model(
                                tokens=processed_batch['input_tokens'],
                                zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                                sequence_embeddings=processed_batch['sequence_embeddings']
                            )
                            total_loss, target_prediction_loss_dict = self.model.prediction_loss(
                                target_predictions=output["target_predictions"], 
                                target_labels=processed_batch['target_labels'],
                                label_smoothing=self.args.label_smoothing
                            )
                        scaler.scale(total_loss).backward()
                else:
                    if self.model.model_type=="ProteinNPT":
                        output = self.model(
                            tokens=processed_batch['masked_tokens'],
                            targets=processed_batch['masked_targets'],
                            zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                            sequence_embeddings=processed_batch['sequence_embeddings']
                        )
                        total_loss, reconstruction_loss, target_prediction_loss_dict = self.model.protein_npt_loss(
                            token_predictions_logits=output['logits_protein_sequence'], 
                            token_labels=processed_batch['token_labels'], 
                            target_predictions=output['target_predictions'], 
                            target_labels=processed_batch['target_labels'], 
                            MLM_reconstruction_loss_weight=reconstruction_loss_coeff, 
                            label_smoothing=self.args.label_smoothing
                        )
                        if total_loss.item() > 10.0 and training_step >= 100:
                            print("High training loss detected: {}".format(total_loss.item()))
                    else:
                        output = self.model(
                            tokens=processed_batch['input_tokens'],
                            zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                            sequence_embeddings=processed_batch['sequence_embeddings']
                        )
                        total_loss, target_prediction_loss_dict = self.model.prediction_loss(
                            target_predictions=output["target_predictions"], 
                            target_labels=processed_batch['target_labels'],
                            label_smoothing=self.args.label_smoothing
                        )
                    total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip)
            # Taking optimizer update out of the inner loop to support gradient accumulation
            if self.args.training_fp16:
                with torch.cuda.amp.autocast():
                    scaler.step(optimizer)
                    scaler.update()
            else:
                optimizer.step()

            log_train_total_loss += total_loss
            if self.model.model_type=="ProteinNPT": 
                num_masked_tokens_in_batch = (~processed_batch['token_labels'].eq(-100)).sum().item()
                log_train_num_masked_tokens += num_masked_tokens_in_batch
                log_train_reconstruction_loss += reconstruction_loss * num_masked_tokens_in_batch
                for target_name in self.model.target_names:
                    if self.args.target_config[target_name]["type"]=="continuous":
                        num_masked_tokens_target_in_batch = processed_batch['masked_targets'][target_name][:,-1].eq(1.0).sum().item() # Masked targets are encoded by 1.0. Mask column is the very last one
                    else:
                        num_masked_tokens_target_in_batch = processed_batch['masked_targets'][target_name].eq(self.args.target_config[target_name]["dim"]).sum().item() # Index of mask is exactly self.args.target_config[target_name]["dim"] (largest value possible)
                    log_train_num_target_masked_tokens_dict[target_name] += num_masked_tokens_target_in_batch
                    log_train_target_prediction_loss_dict[target_name] += target_prediction_loss_dict[target_name] * num_masked_tokens_target_in_batch
            else:
                log_num_sequences_predicted += len(batch['mutant_mutated_seq_pairs'])
                for target_name in self.model.target_names:
                    log_train_target_prediction_loss_dict[target_name] += target_prediction_loss_dict[target_name] * len(batch['mutant_mutated_seq_pairs'])
            if training_step % self.args.num_logging_training_steps == 0 and self.args.use_wandb:
                time_end_step = time.time()
                delta_time_since_last_log = time_end_step - prior_log_time
                total_train_time += delta_time_since_last_log
                prior_log_time = time_end_step
                train_logs = {
                    "training_step": training_step, 
                    "step_time": delta_time_since_last_log / (self.args.num_logging_training_steps)
                }
                if self.model.model_type=="ProteinNPT": 
                    train_logs["train_total_loss_per_step"]: log_train_total_loss / self.args.num_logging_training_steps
                    train_logs["train_reconstruction_loss_per_masked_token"] = log_train_reconstruction_loss.item() / log_train_num_masked_tokens
                    for target_name in self.model.target_names:
                        train_logs["train_prediction_"+str(target_name)+"_loss_per_masked_token"] = log_train_target_prediction_loss_dict[target_name].item() / log_train_num_target_masked_tokens_dict[target_name]
                else:
                    train_logs["train_total_loss_per_seq"]: log_train_total_loss / log_num_sequences_predicted
                    for target_name in self.model.target_names:
                        train_logs["train_prediction_"+str(target_name)+"_loss_per_seq"] = log_train_target_prediction_loss_dict[target_name] / log_num_sequences_predicted
                
                wandb.log(train_logs)
                log_train_total_loss = 0
                log_train_target_prediction_loss_dict = defaultdict(int)
                if self.model.model_type=="ProteinNPT":
                    log_train_reconstruction_loss = 0
                    log_train_num_masked_tokens = 0
                    log_train_num_target_masked_tokens_dict = defaultdict(int)
                else:
                    log_num_sequences_predicted = 0 
                
            if self.args.save_model_checkpoint and (training_step % self.args.num_saving_training_steps) == 0:
                if not os.path.exists(self.args.model_location): os.mkdir(self.args.model_location)
                if not os.path.exists(self.args.model_location + os.sep + 'checkpoint-'+str(training_step)): os.mkdir(self.args.model_location + os.sep + 'checkpoint-'+str(training_step))
                torch.save({
                    'training_step': training_step,
                    'args': self.args,
                    'state_dict': self.model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, 
                    self.args.model_location + os.sep + 'checkpoint-'+str(training_step) + os.sep + 'checkpoint.t7'
                )
            
            if training_step % self.args.num_eval_steps == 0 and self.args.use_validation_set:
                if self.model.model_type=="ProteinNPT":
                    seed_eval_results = self.eval_cg_from_seed(
                        test_data=self.val_seed_data,
                        train_data=self.train_data
                    )
                    eval_results = self.eval(
                        test_data=self.val_data,
                        train_data=self.train_data,
                        reconstruction_loss_weight=0.0,
                        output_all_predictions=True
                    )
                else:
                    seed_eval_results = self.eval_cg_from_seed(
                        test_data=self.val_seed_data,
                        train_data=self.train_data
                    )
                    eval_results = self.eval(
                        test_data=self.val_data, 
                        output_all_predictions=True
                    )

                # Parse logs for eval property prediction using validation set
                eval_logs = {"Training step": training_step}
                eval_logs['Eval total loss per seq.'] = eval_results['eval_total_loss']
                average_spearman_across_targets = 0 # If early stopping based on validation spearman and multiple targets, we check that avg spearman is not decreasing for a certain # of times in a row
                for target_name in self.model.target_names:
                    eval_logs['Eval loss '+str(target_name)+' per seq.'] = eval_results['eval_target_prediction_loss_dict'][target_name]
                    if self.args.target_config[target_name]["dim"]==1:
                        eval_logs['Eval spearman '+target_name] = spearmanr(eval_results['output_scores']['predictions_'+target_name], eval_results['output_scores']['labels_'+target_name])[0]
                    else:
                        # In the categorical setting, we predict the spearman between the logits of the category with highest index, and target value indices. This is meaningul in the binary setting. Use with care if 3 categories or more.
                        eval_logs['Eval spearman '+target_name] = spearmanr(eval_results['output_scores']['predictions_'+target_name][:,-1], eval_results['output_scores']['labels_'+target_name])[0]
                    average_spearman_across_targets += eval_logs['Eval spearman '+target_name]
                average_spearman_across_targets /= len(self.model.target_names)
                print(" | ".join([key + ": "+str(round(eval_logs[key],5)) for key in eval_logs.keys()]))

                # Parse logs for eval conditional generation from seed sequence                
                if self.args.use_wandb:
                    wandb.log(eval_logs)
                    wandb.log(seed_eval_results)
                
                # Early stopping and best model saving based on eval spearman
                all_spearmans_eval_during_training.append(average_spearman_across_targets)
                if average_spearman_across_targets > max_average_spearman_across_targets:
                    max_average_spearman_across_targets = average_spearman_across_targets
                    if not os.path.exists(self.args.model_location): os.mkdir(self.args.model_location)
                    if not os.path.exists(self.args.model_location + os.sep + 'checkpoint-best-spearman'):
                        os.mkdir(self.args.model_location + os.sep + 'checkpoint-best-spearman')

                    torch.save({
                            'training_step': training_step,
                            'args': self.args,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : optimizer.state_dict()
                        }, 
                        self.args.model_location + os.sep + 'checkpoint-best-spearman' + os.sep + 'checkpoint.t7'
                    )

                if (training_step >= 1000) and (self.args.early_stopping_patience is not None) and (np.array(all_spearmans_eval_during_training)[-self.args.early_stopping_patience:].max() < max_average_spearman_across_targets):
                    print("Early stopping. Training step: {}. Total eval loss: {}. Avg spearman: {}".format(training_step, eval_results['eval_total_loss'], average_spearman_across_targets))
                    break
                
                self.model.train() # Move back the model to train mode after eval loop
        
        trainer_final_status = {
            'total_training_steps': training_step,
            'total_train_time': total_train_time,
            'total_training_epochs': num_epochs
        }
        return trainer_final_status

    def eval(
        self,
        test_data,
        output_all_predictions=False,
        need_head_weights=False,
        train_data = None,
        reconstruction_loss_weight=0.0,
        selected_indices_seed=0
    ):
        """
        total_eval_target_prediction_loss is the sum of all target prediction losses across all targets
        total_eval_target_prediction_loss contains the breakdown by target
        num_predicted_targets has the number of predicted items
        output_scores is a dict with sequences, predictions and labels
        """
        import proteinnpt
        self.model.eval()
        self.model.cuda()
        self.model.set_device()
        test_data_size = len(test_data['mutant_mutated_seq_pairs'])
        train_data_size = len(train_data['mutant_mutated_seq_pairs'])
        with torch.no_grad():
            eval_loader = torch.utils.data.DataLoader(
                                dataset=test_data, 
                                batch_size=self.args.eval_num_sequences_to_score_per_batch_per_gpu, 
                                shuffle=False,
                                num_workers=self.args.num_data_loaders_workers,
                                pin_memory=True,
                                collate_fn=collate_fn_protein_npt
                            )
            eval_iterator = iter(eval_loader)
            
            num_eval_batches = 0
            eval_total_loss = 0
            if self.model.model_type=="ProteinNPT": 
                eval_reconstruction_loss = 0
                eval_num_masked_tokens = 0
                eval_num_masked_targets = defaultdict(int)
            else:
                num_predicted_targets = 0
            eval_target_prediction_loss_dict = defaultdict(int)
            output_scores = defaultdict(list) if output_all_predictions else None

            if need_head_weights:
                col_attentions=[]
                row_attentions=[]

            for batch in tqdm.tqdm(eval_iterator):
                if output_all_predictions:
                    output_scores['mutated_sequence'] += list(zip(*batch['mutant_mutated_seq_pairs']))[1]
                    output_scores['mutant'] += list(zip(*batch['mutant_mutated_seq_pairs']))[0]
                if self.model.model_type=="ProteinNPT":
                    processed_batch = proteinnpt.proteinnpt.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet, 
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        target_processing = self.target_processing,
                        training_sequences = train_data,
                        num_training_sequences=train_data_size,
                        proba_target_mask = 1.0, 
                        proba_aa_mask = 0.0,
                        eval_mode = True,
                        device=self.model.device,
                        selected_indices_seed=selected_indices_seed,
                        indel_mode=self.args.indel_mode
                    )
                else:
                    processed_batch = proteinnpt.baselines.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet, 
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        device=self.model.device,
                        eval_mode=True,
                        indel_mode=self.args.indel_mode
                    )
                if self.args.augmentation=="zero_shot_fitness_predictions_covariate":
                    zero_shot_fitness_predictions = processed_batch['target_labels']['zero_shot_fitness_predictions'].view(-1,1)
                    del processed_batch['target_labels']['zero_shot_fitness_predictions']
                else:
                    zero_shot_fitness_predictions = None
        
                if self.model.model_type=="ProteinNPT":
                    output = self.model(
                        tokens=processed_batch['masked_tokens'],
                        targets=processed_batch['masked_targets'],
                        zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                        sequence_embeddings=processed_batch['sequence_embeddings'],
                        need_head_weights=need_head_weights
                    )
                    batch_loss, batch_reconstruction_loss, batch_target_prediction_loss_dict = self.model.protein_npt_loss(
                        token_predictions_logits=output['logits_protein_sequence'], 
                        token_labels=processed_batch['token_labels'], 
                        target_predictions=output['target_predictions'], 
                        target_labels=processed_batch['target_labels'], 
                        MLM_reconstruction_loss_weight=reconstruction_loss_weight, 
                        label_smoothing=self.args.label_smoothing
                    )
                    if batch_loss.item() > 10.0:
                        print("High eval loss detected: {}".format(batch_loss.item()))
                else:
                    output = self.model(
                        tokens=processed_batch['input_tokens'],
                        zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                        sequence_embeddings=processed_batch['sequence_embeddings']
                    )
                    batch_loss, batch_target_prediction_loss_dict = self.model.prediction_loss(
                        target_predictions=output["target_predictions"], 
                        target_labels=processed_batch['target_labels'],
                        label_smoothing=self.args.label_smoothing
                    )
                
                num_eval_batches += 1
                eval_total_loss += batch_loss.item()
                if self.model.model_type=="ProteinNPT":
                    num_masked_tokens_in_batch = (processed_batch['masked_tokens'].eq(self.model.alphabet.mask_idx)).sum().item()
                    eval_num_masked_tokens += num_masked_tokens_in_batch
                    eval_reconstruction_loss += batch_reconstruction_loss.item() * num_masked_tokens_in_batch
                    for target_name in self.model.target_names:
                        if self.args.target_config[target_name]["type"]=="continuous":
                            num_masked_tokens_target_in_batch = processed_batch['masked_targets'][target_name][:,-1].eq(1.0).sum().item() # Masked targets are encoded by 1.0. Mask column is the very last one
                        else:
                            num_masked_tokens_target_in_batch = processed_batch['masked_targets'][target_name].eq(self.args.target_config[target_name]["dim"]).sum().item() # Index of mask is exactly self.args.target_config[target_name]["dim"] (largest value possible)
                        eval_num_masked_targets[target_name] += num_masked_tokens_target_in_batch
                        eval_target_prediction_loss_dict[target_name] += batch_target_prediction_loss_dict[target_name].item() * num_masked_tokens_target_in_batch
                else:
                    num_predicted_targets += len(batch['mutant_mutated_seq_pairs'])
                    for target_name in self.model.target_names:
                        eval_target_prediction_loss_dict[target_name] += batch_target_prediction_loss_dict[target_name].item() * len(batch['mutant_mutated_seq_pairs'])
                if output_all_predictions:
                    num_of_mutated_seqs_to_score = processed_batch['num_of_mutated_seqs_to_score'] if self.model.model_type=="ProteinNPT" else len(processed_batch['mutant_mutated_seq_pairs'])
                    for target_name in self.model.target_names:
                        output_scores['predictions_'+target_name] += list(output["target_predictions"][target_name][:num_of_mutated_seqs_to_score].cpu().numpy())
                        output_scores['labels_'+target_name] += list(processed_batch['target_labels'][target_name][:num_of_mutated_seqs_to_score].cpu().numpy())
                if need_head_weights:
                    col_attentions.append(output["col_attentions"])
                    row_attentions.append(output["row_attentions"])

            output_scores = pd.DataFrame.from_dict(output_scores)
            output_scores_numeric_cols = [col_name for col_name in output_scores.columns if col_name not in ['mutated_sequence']]
            output_scores = output_scores[output_scores_numeric_cols]
            assert len(output_scores)==output_scores['mutant'].nunique()
            mutated_seqs_dict = {}
            mutant_mutated_seqs = list(zip(*test_data['mutant_mutated_seq_pairs']))
            mutated_seqs_dict['mutant'] = mutant_mutated_seqs[0]
            mutated_seqs_dict['mutated_sequence'] = mutant_mutated_seqs[1]
            mutated_seqs_df = pd.DataFrame.from_dict(mutated_seqs_dict)
            output_scores = pd.merge(output_scores, mutated_seqs_df, on='mutant', how='left')

        # Normalization
        for target_name in self.model.target_names:
            if self.model.model_type=="ProteinNPT":
                eval_target_prediction_loss_dict[target_name] /= eval_num_masked_targets[target_name] # We track exactly how many targets were masked across batches to account for potential discrepancies across batches (eg., last abtch may not have the same number of labels)
            else:
                eval_target_prediction_loss_dict[target_name] /= num_predicted_targets
        eval_results = {
            'eval_total_loss':eval_total_loss / num_eval_batches,
            'eval_target_prediction_loss_dict': eval_target_prediction_loss_dict,
            'output_scores': output_scores
        }
        if need_head_weights:
            print("dimension of first attention column {}".format(col_attentions[0].shape))
            eval_results['col_attentions'] = torch.stack(col_attentions, dim=0).cpu().numpy()
            eval_results['row_attentions'] = torch.stack(row_attentions, dim=0).cpu().numpy()
        
        if self.model.model_type=="ProteinNPT":
            if eval_num_masked_tokens > 0:
                eval_results['eval_reconstruction_loss'] = eval_reconstruction_loss / eval_num_masked_tokens
                eval_results['eval_num_masked_tokens'] = eval_num_masked_tokens
            eval_results['eval_num_masked_targets'] = eval_num_masked_targets
        else:
            eval_results['eval_num_predicted_targets'] = num_predicted_targets
        return eval_results


    def eval_cg_from_seed(
        self,
        test_data,
        train_data = None,
        proba_aa_mask = 0.05217391304347826,
        selected_indices_seed=0,
    ):
        """
        Generates variants from the seed sequence conditioned on custom properties, then 
        computes a self-consistency score based on the predicted properties of the generated variants.
        Ideally, we want the predicted properties of the generated variants to be close to the
        target properties.
        TODO: Predicted properties are not yet implemented
        TODO: Implement a more general version of this function that can handle multiple properties
        TODO: Implement other sampling methods
        """
        import proteinnpt
        self.model.eval()
        self.model.cuda()
        self.model.set_device()
        test_data_size = len(test_data['mutant_mutated_seq_pairs'])
        train_data_size = len(train_data['mutant_mutated_seq_pairs'])

        lead_seq = self.args.target_seq
        cdr_mask = self.args.target_seq_cdr_mask
        assert lead_seq is not None, "Lead sequence is required for conditional generation from seed"
        assert cdr_mask is not None, "CDR mask is required for conditional generation from seed"

        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_token_mask = sorted([self.model.alphabet.tok_to_idx[aa] for aa in amino_acids])
        aa_token_map = sorted(amino_acids, key=lambda x: self.model.alphabet.tok_to_idx[x])
        sample_mask = self.args.target_seq_mutable_mask if self.args.target_seq_mutable_mask is not None\
            else [True]*len(lead_seq)
        full_sample_mask = np.concatenate([[False], sample_mask, [False]], axis=0).squeeze()    # Pad the mask for <bos> and <eos> tokens
        temperature = 1.0
        new_sequences = []
        regions = ['fr1', 'cdr1', 'fr2', 'cdr2', 'fr3', 'cdr3', 'fr4']
        prop = 'tm' # Only support single property for now

        with torch.no_grad():
            eval_loader = torch.utils.data.DataLoader(
                                dataset=test_data, 
                                batch_size=self.args.eval_num_sequences_to_score_per_batch_per_gpu, 
                                shuffle=False,
                                num_workers=self.args.num_data_loaders_workers,
                                pin_memory=True,
                                collate_fn=collate_fn_protein_npt
                            )
            eval_iterator = iter(eval_loader)
            
            num_eval_batches = 0
            eval_total_loss = 0
            if self.model.model_type=="ProteinNPT": 
                eval_reconstruction_loss = 0
                eval_num_masked_tokens = 0
                eval_num_masked_targets = defaultdict(int)
            else:
                num_predicted_targets = 0
            eval_target_prediction_loss_dict = defaultdict(int)
            output_scores = defaultdict(list)

            for batch in tqdm.tqdm(eval_iterator):
                output_scores['mutated_sequence'] += list(zip(*batch['mutant_mutated_seq_pairs']))[1]
                output_scores['mutant'] += list(zip(*batch['mutant_mutated_seq_pairs']))[0]
                
                if self.model.model_type=="ProteinNPT":
                    processed_batch = proteinnpt.proteinnpt.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet, 
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        target_processing = self.target_processing,
                        training_sequences = train_data,
                        num_training_sequences=train_data_size,
                        proba_target_mask = 0.0,                    # Never mask the target properties during CG
                        proba_aa_mask = proba_aa_mask,              # Mask amino acids with this probability
                        aa_can_mask=full_sample_mask,               # Only mask amino acids at these positions
                        eval_mode = True,
                        mask_training_aa = False,                   # Do not mask training amino acids during eval
                        device=self.model.device,
                        selected_indices_seed=selected_indices_seed,
                        indel_mode=self.args.indel_mode
                    )
                else:
                    processed_batch = proteinnpt.baselines.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet, 
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        device=self.model.device,
                        eval_mode=True,
                        indel_mode=self.args.indel_mode
                    )

                if self.args.augmentation=="zero_shot_fitness_predictions_covariate":
                    zero_shot_fitness_predictions = processed_batch['target_labels']['zero_shot_fitness_predictions'].view(-1,1)
                    del processed_batch['target_labels']['zero_shot_fitness_predictions']
                else:
                    zero_shot_fitness_predictions = None
        
                if self.model.model_type=="ProteinNPT":
                    output = self.model(
                        tokens=processed_batch['masked_tokens'],
                        targets=processed_batch['masked_targets'],
                        zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                        sequence_embeddings=processed_batch['sequence_embeddings'],
                        need_head_weights=False
                    )
                    batch_loss, batch_reconstruction_loss, batch_target_prediction_loss_dict = self.model.protein_npt_loss(
                        token_predictions_logits=output['logits_protein_sequence'], 
                        token_labels=processed_batch['token_labels'], 
                        target_predictions=output['target_predictions'], 
                        target_labels=processed_batch['target_labels'], 
                        MLM_reconstruction_loss_weight=1.0,
                        label_smoothing=self.args.label_smoothing
                    )
                    if batch_loss.item() > 10.0:
                        print("High eval loss detected: {}".format(batch_loss.item()))
                else:
                    output = self.model(
                        tokens=processed_batch['input_tokens'],
                        zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                        sequence_embeddings=processed_batch['sequence_embeddings']
                    )
                    batch_loss, batch_target_prediction_loss_dict = self.model.prediction_loss(
                        target_predictions=output["target_predictions"], 
                        target_labels=processed_batch['target_labels'],
                        label_smoothing=self.args.label_smoothing
                    )
                
                num_eval_batches += 1
                eval_total_loss += batch_loss.item()

                # Sample generated sequences using the logits of the output
                logits = output['logits_protein_sequence'].detach().squeeze().cpu().numpy()
                sample_mask = processed_batch['masked_tokens'].eq(self.model.alphabet.mask_idx).squeeze().cpu().numpy()

                for i in range(self.args.eval_num_sequences_to_score_per_batch_per_gpu):
                    ith_logits = logits[i]
                    sample_mask_idx = np.where(sample_mask[i])[0]
                    if len(sample_mask_idx) == 0:
                        continue
                    ith_logits = torch.tensor(ith_logits[sample_mask_idx][:,aa_token_mask])

                    # Sample from the logits to obtain new amino acid token indices
                    aa_categorical = torch.distributions.Categorical(logits=ith_logits/temperature)
                    new_aa_token_indices = aa_categorical.sample()
                    new_sequence = list(lead_seq)
                    
                    for ind, aa_token in zip(sample_mask_idx, new_aa_token_indices):
                        assert sample_mask[i][ind], "Sample mask is not set correctly"
                        assert full_sample_mask[ind], "Full sample mask is not set correctly"
                        new_sequence[ind-1] = aa_token_map[aa_token.item()]   # -1 to account for <bos> token

                    new_sequence = [aa for aa in new_sequence if aa != '-']
                    new_sequences.append("".join(new_sequence))
                
                if self.model.model_type=="ProteinNPT":
                    num_masked_tokens_in_batch = (processed_batch['masked_tokens'].eq(self.model.alphabet.mask_idx)).sum().item()
                    eval_num_masked_tokens += num_masked_tokens_in_batch
                    eval_reconstruction_loss += batch_reconstruction_loss.item() * num_masked_tokens_in_batch
                else:
                    num_predicted_targets += len(batch['mutant_mutated_seq_pairs'])

        unique_new_sequences = list(set(new_sequences))
        assert all([len(s) == len(cdr_mask) for s in unique_new_sequences]),\
            "All generated sequences must have the same length as the seed sequence"

        new_samples_regions = [apply_cdr_mask(s, cdr_mask) for s in unique_new_sequences]
        new_samples_regions = {reg: [s[reg] for s in new_samples_regions] for reg in regions}
        num_unique_regions = {
            f"seed_eval_num_{reg.capitalize()}_{prop[1:-1]}": len(set(new_samples_regions[reg]))
            for reg in new_samples_regions.keys()
        }
        
        eval_results = {
            'seed_eval_total_loss': eval_total_loss / num_eval_batches,
            'seed_eval_reconstruction_loss': eval_reconstruction_loss / eval_num_masked_tokens,
            'seed_eval_avg_num_masked_tokens': eval_num_masked_tokens / num_eval_batches,
            'seed_eval_num_unique_seqs': len(unique_new_sequences),
            **num_unique_regions
        }

        print(eval_results, sep='\n')
        
        return eval_results
    