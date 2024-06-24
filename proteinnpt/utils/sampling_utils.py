import h5py
import sys
import os
import torch
import numpy as np
from typing import List, Dict
from torchtyping import TensorType as TT

from .data_utils import mask_targets, slice_sequences, get_indices_retrieved_embeddings
from proteinnpt.proteinnpt.data_processing import PNPT_sample_training_points_inference

class SamplingModel:
    """
    Similar class to Trainer that allows us to sample new sequences from the model conditioned on desired target values
    and zero-shot fitness predictions. The model only masks the amino acids in the test sequence and outputs the posterior
    distribution of the masked amino acids at these positions.
    """
    def __init__(self, model, args, train_data, target_processing):
        self.model = model
        self.args = args
        self.train_data = train_data
        self.target_processing = target_processing

        self.model.eval()
        self.model.cuda()
        self.model.set_device()

    def prepare_batch(self, x: List[str], masks: List[List[bool]], target_values: Dict[str, TT["B"]], zero_shot_fitness_predictions: TT["B"] = None, selected_indices_seed=0):
        """
        Prepares a batch for the model to score where `x` test sequences will have some of their amino acids masked and 
        the remaining `args.eval_num_training_sequences_per_batch_per_gpu` will be unmasked sequence, label pairs from the
        provided `self.train_data` set. 
        """
        batch = {
            # For now, x must be the seed sequence
            'mutant_mutated_seq_pairs': [['nan', x] for x in x],
            **target_values,
            'zero_shot_fitness_predictions': zero_shot_fitness_predictions
        }

        target_names = self.args.target_config.keys()
        raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1])
        number_of_mutated_seqs_to_score = len(batch['mutant_mutated_seq_pairs']) 
        
        batch_masked_targets = {}
        batch_target_labels = {}

        for target_name in target_names:
            assert target_name in target_values or target_name == "zero_shot_fitness_predictions"
            masked_targets, target_labels = mask_targets(
                inputs = batch[target_name],
                input_target_type = self.args.target_config[target_name]["type"], 
                target_processing = self.target_processing[target_name], 
                proba_target_mask = 0.0,
                proba_random_mutation = 0.0,
                proba_unchanged = 1.0,
            )
            batch_masked_targets[target_name] = masked_targets
            batch_target_labels[target_name] = target_labels

        num_sequences_training_data = len(self.train_data['mutant_mutated_seq_pairs'])
        if self.model.training_sample_sequences_indices is None:
            selected_indices_dict = {}
            num_ensemble_seeds = self.model.PNPT_ensemble_test_num_seeds if self.model.PNPT_ensemble_test_num_seeds > 0 else 1
            for ensemble_seed in range(num_ensemble_seeds):
                selected_indices_dict[ensemble_seed] = PNPT_sample_training_points_inference(
                    training_sequences=self.train_data,
                    sequences_sampling_method=self.args.eval_training_sequences_sampling_method, 
                    num_sampled_points=self.args.eval_num_training_sequences_per_batch_per_gpu
                )
            self.model.training_sample_sequences_indices = selected_indices_dict
        selected_indices = self.model.training_sample_sequences_indices[selected_indices_seed]
        num_selected_training_sequences = len(np.array(self.train_data['mutant_mutated_seq_pairs'])[selected_indices])
        batch['mutant_mutated_seq_pairs'] += list(np.array(self.train_data['mutant_mutated_seq_pairs'])[selected_indices])

        for target_name in target_names:
            # training_sequences[target_name] expected of size (len_training_seq,2). No entry is actually masked here since we want to use all available information to predict as accurately as possible
            masked_training_targets, training_target_labels = mask_targets(
                inputs = torch.tensor(np.array(self.train_data[target_name])[selected_indices]),
                input_target_type = self.args.target_config[target_name]["type"], 
                target_processing = self.target_processing[target_name], 
                proba_target_mask = 0.0,
                proba_random_mutation = 0.0,
                proba_unchanged = 1.0
            )
            batch_masked_targets[target_name] = torch.cat( [batch_masked_targets[target_name], masked_training_targets], dim=0).float().to(self.model.device)
            num_all_mutated_sequences_input = number_of_mutated_seqs_to_score + num_selected_training_sequences
            assert batch_masked_targets[target_name].shape[0] == num_all_mutated_sequences_input, "Error adding training data to seqs to score: {} Vs {}".format(batch_masked_targets[target_name].shape[0], num_all_mutated_sequences_input)
            batch_target_labels[target_name] = torch.cat( [batch_target_labels[target_name], training_target_labels]).float().to(self.model.device) 
            assert batch_masked_targets[target_name].shape[0] == batch_target_labels[target_name].shape[0], "Lengths of masked targets and target labels do not match: {} Vs {}".format(batch_masked_targets[target_name].shape[0], batch_target_labels[target_name].shape[0])

        # Embedding loading needs to happen here to ensure we also load training sequences at eval time
        # TODO: Compute sequence embeddings on the fly for sampling sequences since we can't guarantee that we will always start from a known sequence
        if self.args.sequence_embeddings_location is not None:
            assert os.path.exists(self.args.sequence_embeddings_location), f"Sequence embeddings location doesn't exist: {self.args.sequence_embeddings_location}"
            try:
                indices_retrieved_embeddings = get_indices_retrieved_embeddings(batch, self.args.sequence_embeddings_location)
                assert len(indices_retrieved_embeddings)==len(batch['mutant_mutated_seq_pairs']) , "At least one embedding was missing"
                with h5py.File(self.args.sequence_embeddings_location, 'r') as h5f:
                    sequence_embeddings = torch.tensor(np.array([h5f['embeddings'][i] for i in indices_retrieved_embeddings])).float()
            except Exception as e:
                print("Error loading main sequence embeddings:", e)
                sys.exit(0)
        else:
            sequence_embeddings = None
        
        # Slice sequences around mutation if sequence longer than context length
        if self.args.max_positions is not None and raw_sequence_length + 1 > self.args.max_positions: # Adding one for the BOS token
            if self.args.long_sequences_slicing_method=="center" and self.args.aa_embeddings=="MSA_Transformer":
                print("Center slicing method not adapted to MSA Transformer embedding as sequences would not be aligned in the same system anymore. Defaulting to 'left' mode.")
                self.args.long_sequences_slicing_method="left"
            batch['mutant_mutated_seq_pairs'], batch_target_labels, batch_masked_targets, batch_scoring_optimal_window = slice_sequences(
                list_mutant_mutated_seq_pairs = batch['mutant_mutated_seq_pairs'], 
                max_positions=self.args.max_positions,
                method=self.args.long_sequences_slicing_method,
                rolling_overlap=self.args.max_positions//4,
                eval_mode=True,
                batch_target_labels=batch_target_labels,
                batch_masked_targets=batch_masked_targets,
                start_idx=1,
                target_names=target_names,
                num_extra_tokens=2 if self.args.aa_embeddings=="Tranception" else 1
            )
        else:
            batch_scoring_optimal_window = None

        # Tokenize protein sequences
        if self.args.aa_embeddings == "MSA_Transformer" and num_all_mutated_sequences_input > 1 and self.args.sequence_embeddings_location is None: 
            #Re-organize list of sequences to have training_num_assay_sequences_per_batch_per_gpu MSA batches, where in each the sequence to score is the first and the rest are the sampled MSA sequences.
            num_sequences = num_all_mutated_sequences_input + self.args.num_MSA_sequences_per_training_instance
            assert len(batch['mutant_mutated_seq_pairs']) == num_sequences, "Unexpected number of sequences"
            all_mutated_sequences_input = batch['mutant_mutated_seq_pairs'][:num_all_mutated_sequences_input]
            MSA_sequences = batch['mutant_mutated_seq_pairs'][num_all_mutated_sequences_input:]
            batch['mutant_mutated_seq_pairs'] = [ [sequence] + MSA_sequences for sequence in all_mutated_sequences_input]

        if self.args.aa_embeddings in ["MSA_Transformer", "Linear_embedding"] or self.args.aa_embeddings.startswith("ESM"):
            token_batch_converter = self.model.alphabet.get_batch_converter()
            batch_sequence_names, batch_AA_sequences, batch_token_sequences = token_batch_converter(batch['mutant_mutated_seq_pairs'])
            if self.args.aa_embeddings=="MSA_Transformer" and self.args.sequence_embeddings_location is not None: #If loading MSAT embeddings from disk, we drop the MSA dimension (done already if not MSAT via the different tokenizer)
                num_MSAs_in_batch, num_sequences_in_alignments, seqlen = batch_token_sequences.size()
                batch_token_sequences = batch_token_sequences.view(num_sequences_in_alignments, seqlen) #drop the dummy batch dimension from the tokenizer when using ESM1v / LinearEmbedding
        elif self.args.aa_embeddings == "Tranception":
            _, sequence = zip(*batch['mutant_mutated_seq_pairs'])
            batch_token_sequences = torch.tensor(self.model.alphabet(sequence, add_special_tokens=True, truncation=True, padding=True, max_length=self.model.aa_embedding.config.n_ctx)['input_ids'])
            
        # Mask protein sequences using the provided masks
        masks = torch.concatenate([torch.zeros((len(masks),1)).bool(), torch.tensor(np.array(masks))], dim=-1)
        batch_token_sequences[:number_of_mutated_seqs_to_score][masks] = self.model.alphabet.mask_idx

        masked_indices = torch.zeros_like(batch_token_sequences)
        masked_indices[batch_token_sequences == self.model.alphabet.mask_idx] = 1
        masked_indices = masked_indices.bool()
        batch_token_labels = batch_token_sequences.clone()
        batch_token_labels[~masked_indices] = -100
        batch_masked_tokens = batch_token_sequences

        if self.args.sequence_embeddings_location is not None:
            if sequence_embeddings.shape[1] > masked_indices.shape[1]: # When dealing with sequences of different sizes, and sequences in batch happen to be all smaller than longest sequence in assay for which we computed embeddings
                extra_padding_in_embeddings = (sequence_embeddings.shape[1] - masked_indices.shape[1])
                sequence_embeddings = sequence_embeddings[:,:-extra_padding_in_embeddings]
            sequence_embeddings[masked_indices] = 0.0

        batch_masked_tokens = batch_masked_tokens.to(self.model.device)
        batch_token_labels = batch_token_labels.to(self.model.device)
        processed_batch = {
            'masked_tokens': batch_masked_tokens,
            'token_labels': batch_token_labels,
            'masked_targets': batch_masked_targets,
            'target_labels': batch_target_labels,
            'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs'],
            'num_all_mutated_sequences_input': num_all_mutated_sequences_input,
            'num_of_mutated_seqs_to_score': number_of_mutated_seqs_to_score,
            'num_selected_training_sequences': num_selected_training_sequences,
            'sequence_embeddings': sequence_embeddings
        }
        return processed_batch

    def forward(self, x: str, target_values: dict, zero_shot_fitness_predictions: dict):
        processed_batch = self.prepare_batch([x], target_values, zero_shot_fitness_predictions)
        breakpoint()
        output = self.model(
            tokens=processed_batch['masked_tokens'],
            targets=processed_batch['masked_targets'],
            zero_shot_fitness_predictions=zero_shot_fitness_predictions,
            sequence_embeddings=processed_batch['sequence_embeddings'],
            need_head_weights=False
        )
        breakpoint()
