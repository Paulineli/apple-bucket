'''
Shell command to run the script:
  Single GPU:  python das.py --train --hf-cache-dir /vision/u/puyinli/Multi_Variable_Causal_Abstraction/.hf_cache --batch-size 8
  Multi GPU:   python das.py --train --hf-cache-dir /vision/u/puyinli/Multi_Variable_Causal_Abstraction/.hf_cache --batch-size 32 --num-gpus 8
  
  Boundless DAS (auto-selects feature dimensions):
  Single GPU:  python das.py --train --intervention-type boundless --hf-cache-dir /vision/u/puyinli/Multi_Variable_Causal_Abstraction/.hf_cache --batch-size 8
  Multi GPU:   python das.py --train --intervention-type boundless --hf-cache-dir /vision/u/puyinli/Multi_Variable_Causal_Abstraction/.hf_cache --batch-size 32 --num-gpus 8
'''
import os
import sys

# Set HuggingFace cache directory BEFORE importing transformers
# This must happen before any HuggingFace library is imported
for i, arg in enumerate(sys.argv):
    if arg == "--hf-cache-dir" and i + 1 < len(sys.argv):
        os.environ["HF_HOME"] = sys.argv[i + 1]
        print(f"Using HuggingFace cache directory: {sys.argv[i + 1]}")
        break

import util_model 
import util_data 
import torch
import random

# ========== Register Qwen3 support for pyvene ==========
# Qwen3 has the same architecture as Qwen2, so we reuse the same mappings
def register_qwen3_for_pyvene():
    """Register Qwen3 model types with pyvene's type mappings."""
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
        from pyvene.models.intervenable_modelcard import type_to_module_mapping, type_to_dimension_mapping
        from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
            qwen2_type_to_module_mapping,
            qwen2_type_to_dimension_mapping,
            qwen2_lm_type_to_module_mapping,
            qwen2_lm_type_to_dimension_mapping,
            qwen2_classifier_type_to_module_mapping,
            qwen2_classifier_type_to_dimension_mapping,
        )
        
        # Register Qwen3 models using Qwen2 mappings (same architecture)
        if hasattr(qwen3_modeling, 'Qwen3Model'):
            type_to_module_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_dimension_mapping
        
        if hasattr(qwen3_modeling, 'Qwen3ForCausalLM'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_dimension_mapping
        
        if hasattr(qwen3_modeling, 'Qwen3ForSequenceClassification'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForSequenceClassification] = qwen2_classifier_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForSequenceClassification] = qwen2_classifier_type_to_dimension_mapping
        
        print("Successfully registered Qwen3 support for pyvene")
    except ImportError as e:
        print(f"Warning: Could not register Qwen3 for pyvene: {e}")
    except Exception as e:
        print(f"Warning: Error registering Qwen3 for pyvene: {e}")

register_qwen3_for_pyvene()
# ========================================================
from tqdm.auto import tqdm
from tqdm import trange
from torch.utils.data import DataLoader
from pyvene import (
    IntervenableModel,
    VanillaIntervention,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
import json 
import argparse
import numpy as np

def get_model_input_device(model):
    """Get the device where model inputs should be sent.
    
    For models with device_map (multi-GPU), this returns the device where
    the input embeddings are located. For single-device models, returns
    that device.
    """
    # Try to get device from input embeddings (most reliable for multi-GPU)
    try:
        embed = model.get_input_embeddings()
        if embed is not None:
            return next(embed.parameters()).device
    except (StopIteration, AttributeError):
        pass
    
    # Fallback: check hf_device_map for the embedding layer
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Find the embedding layer device
        for key, device in model.hf_device_map.items():
            if 'embed' in key.lower():
                return torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        # If no embed found, use the first layer's device
        first_device = next(iter(model.hf_device_map.values()))
        return torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    
    # Fallback: use first parameter's device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_metrics(eval_preds, eval_labels):
    ''' This function is used to compute the accuracy of the predictions. '''
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        total_count += 1
        correct_count += eval_pred == eval_label
    accuracy = float(correct_count) / float(total_count)
    return {"accuracy": accuracy}

def compute_loss(outputs, labels):
    ''' This function is used to compute the loss of the predictions. We will use cross entropy loss. '''
    CE = torch.nn.CrossEntropyLoss()
    return CE(outputs, labels)

def batched_random_sampler(data, batch_size):
    batch_indices = [_ for _ in range(int(len(data) / batch_size))]
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i * batch_size, (b_i + 1) * batch_size):
            yield i

def set_random_seed(seed: int):
    """Set random seed for python, numpy and torch (CPU and CUDA).

    This helps reproducibility for dataset shuffling, model init and training.
    """
    import os
    # Python
    random.seed(seed)
    # OS-level hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    # NumPy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_positional_indices(sample_input_id, tokenizer, sub_token = 'logic_function(', return_token_mapping=False):
    ''' This function is used to find the positional indices of the first token in the input_ids after sub_token.
    This returns the position where {t0} appears in the formatted prompt.
    Input:
        sample_input_id: the input tensor of token ids, tensor with shape (1, k)
        tokenizer: the tokenizer used to tokenize the input, transformers.PreTrainedTokenizer
        sub_token: the sub_token to be found, str (default: 'check(')
        return_token_mapping: if True, also returns a dict mapping position -> token string
    Output:
        pos_indices: the positional indices of the first token after the sub_token (i.e., position of {t0}). 
                     If there are multiple occurrences, return the last one. int and the position of '<|im_end|>'
        token_mapping (optional): dict mapping position (int) -> token (str), only returned if return_token_mapping=True'''
    
    # Handle shape (1, k) by squeezing or indexing
    if len(sample_input_id.shape) == 2 and sample_input_id.shape[0] == 1:
        input_ids = sample_input_id.squeeze(0)
    elif len(sample_input_id.shape) == 1:
        input_ids = sample_input_id
    else:
        raise ValueError(f"Unexpected shape for sample_input_id: {sample_input_id.shape}")
    
    # Build token mapping if requested
    token_mapping = {}
    if return_token_mapping:
        for i, token_id in enumerate(input_ids):
            token_mapping[i] = tokenizer.decode([token_id])
    
    # Decode the full sequence to find the text position
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    
    # Find the last occurrence of the sub_token in the decoded text
    text_pos = full_text.rfind(sub_token)
    
    # length of sequence
    input_length = len(input_ids)  # Default to length if not found
    
    
    if text_pos == -1:
        print(f"Warning: '{sub_token}' not found in decoded text")
        print(f"Decoded text: {full_text}")
        if return_token_mapping:
            return -1, input_length, token_mapping
        return -1, input_length
    
    # Now find which token position corresponds to the character position after sub_token
    target_char_pos = text_pos + len(sub_token)
    
    # Decode token by token to find the first token that starts AFTER target_char_pos
    # We want the token position where the previous tokens' decoded text ends at or before target_char_pos
    last_occurrence_pos = -1
    
    for i in range(len(input_ids)):
        # Get the text decoded up to (but not including) token i
        prev_text = tokenizer.decode(input_ids[:i], skip_special_tokens=False) if i > 0 else ""
        # If previous text already covers target position, token i is the first token after sub_token
        if len(prev_text) >= target_char_pos:
            last_occurrence_pos = i
            break
    
    # If we never found it (edge case), fall back to last token
    if last_occurrence_pos == -1:
        last_occurrence_pos = len(input_ids) - 1
    
    # print(f"Position of first argument (t0): {last_occurrence_pos}")
    if return_token_mapping:
        return last_occurrence_pos, input_length, token_mapping
    return last_occurrence_pos, input_length

def DAS_training(intervenable, train_dataset, optimizer, pos, device, tokenizer=None, epochs = 10, batch_size = 64, gradient_accumulation_steps = 1):
    '''Main code for training the model with DAS intervention.
    Input:
        intervenable: the model with the intervention, pyvene.IntervenableModel
        train_dataset: the training dataset, contain input_ids, source_input_ids, labels
        optimizer: the optimizer to be used, torch.optim.Adam
        pos: the position of the intervention, int
        device: the device id to be used, int
        epochs: the number of epochs to train, int
        batch_size: the batch size to be used, int
        gradient_accumulation_steps: the number of steps to accumulate gradients, int
    Output:
        None, the model will be trained in-place.
    This function will train the model with the intervention, and compute the loss and accuracy.
    '''
    
    intervenable.model.train()  # set the module to train mode, which enables drop-off but no grads
    print("intervention trainable parameters: ", intervenable.count_parameters()) # count the number of trainable parameters in the intervention

    train_iterator = trange(0, int(epochs), desc="Epoch")  # create a progress bar for the epochs
    total_step = 0
    for epoch in train_iterator:
        epoch_correct = 0
        epoch_total = 0
        epoch_loss_sum = 0.0
        
        epoch_iterator = tqdm( # create a progress bar for the batches
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(train_dataset, batch_size),
            ),
            desc=f"Epoch: {epoch}",
            position=0,
            leave=False,
            dynamic_ncols=True,
        )

        for batch in epoch_iterator:
            # Jiyuan: Need to verify the shape of input_ids and source_input_ids. The code should be correct, but I don't remember the exact shape.
            batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
            #print(batch["input_ids"].shape, batch["source_input_ids"].shape)
            current_batch_size = batch["input_ids"].shape[0]
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Interchange intervention: Please pay attention to the shape. It can be tricky.
            _, counterfactual_outputs = intervenable(
                {"input_ids": batch["input_ids"]},
                [
                    {"input_ids": batch["source_input_ids"]},
                ],
                {
                    "sources->base": (
                        [[[pos]] * current_batch_size], [[[pos]] * current_batch_size],
                    )
                },
                subspaces=[
                    [[0]] * current_batch_size,
                ],
            )

            # compute metrics
            preds = counterfactual_outputs.logits[:,-1,:].argmax(dim=-1)
            labels = batch["labels"].squeeze()
            

            # loss and backprop
            loss = compute_loss(
                counterfactual_outputs.logits[:,-1,:], labels
            )
            
            # Accumulate epoch stats
            epoch_correct += (preds == labels).sum().item()
            epoch_total += current_batch_size
            epoch_loss_sum += loss.item() * current_batch_size


            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            total_step += 1

            # Step optimizer after accumulating enough gradients
            if total_step % gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()

        # Handle any remaining accumulated gradients at end of epoch
        if total_step % gradient_accumulation_steps != 0:
            optimizer.step()
            intervenable.set_zero_grad()

        # Show loss and accuracy for the epoch
        train_iterator.set_postfix(
            loss=epoch_loss_sum / epoch_total if epoch_total > 0 else 0,
            accuracy=epoch_correct / epoch_total if epoch_total > 0 else 0,
        )
        epoch_iterator.close()  # Close inner progress bar after each epoch

def das_test(intervenable, pos, test_dataset, device, batch_size = 64, intervention_type = 'das', return_details = False):
    ''' This function is used to test the model with the intervention.
    Input:
        intervenable: the model with the intervention, pyvene.IntervenableModel
        pos: the position of the intervention, int
        test_dataset: the testing dataset, contain input_ids, source_input_ids, labels
        device: the device id to be used, int
        batch_size: the batch size to be used, int
        return_details: if True, also return per-sample correctness (for creating analysis datasets)
    Output:
        acc: the accuracy of the model, float
        (optional) details: dict with 'eval_labels', 'eval_preds', 'correct', 'indices' if return_details=True
    This function will test the model with the intervention, and compute the accuracy.'''
    eval_labels = []
    eval_preds = []
    sample_indices = []  # Track which samples were processed (in order)
    
    with torch.no_grad():
        # When return_details is True, use sequential sampling to preserve order
        # Otherwise, use batched_random_sampler for standard testing
        if return_details:
            # Sequential sampling - process samples in order
            sampler = range(0, (len(test_dataset) // batch_size) * batch_size, 1)
            data_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,  # Keep order for feature alignment
            )
        else:
            data_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(test_dataset, batch_size),
            )
        
        epoch_iterator = tqdm(
            data_loader,
            desc=f"Testing",
            position=0,
            leave=False,
        )

        batch_idx = 0
        for batch in epoch_iterator:
            batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
            #print(batch["input_ids"].shape, batch["source_input_ids"].shape)
            current_batch_size = batch["input_ids"].shape[0]
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            if intervention_type == 'das':
                _, counterfactual_outputs = intervenable(
                    {"input_ids": batch["input_ids"]},
                    [
                        {"input_ids": batch["source_input_ids"]},
                    ],
                    {
                        "sources->base": (
                            [[[pos]] * current_batch_size], [[[pos]] * current_batch_size],
                        )
                    },
                    subspaces=[
                        [[0]] * current_batch_size,
                    ],
                )
            elif intervention_type == 'vanilla':
                _, counterfactual_outputs = intervenable(
                    {"input_ids": batch["input_ids"]},
                    [
                        {"input_ids": batch["source_input_ids"]},
                    ],
                    {
                        "sources->base": (
                            [[[pos]] * current_batch_size], [[[pos]] * current_batch_size],
                        )
                    }
                )
            else:
                raise ValueError("intervention_type must be 'das' or 'vanilla'")
            eval_labels += [batch["labels"].squeeze()]
            eval_preds += [counterfactual_outputs.logits[:,-1,:].argmax(dim=-1)]
            
            if return_details:
                # Track sample indices for this batch
                # If the batch contains 'idx' field, use it; otherwise fall back to sequential
                if "idx" in batch:
                    sample_indices.extend(batch["idx"].tolist())
                else:
                    start_idx = batch_idx * batch_size
                    sample_indices.extend(range(start_idx, start_idx + current_batch_size))
            batch_idx += 1
   
    eval_labels = torch.cat(eval_labels)
    eval_preds = torch.cat(eval_preds)
    acc = sum(eval_preds == eval_labels).item() / len(eval_labels)
    
    if return_details:
        # Compute per-sample correctness (binary: 1 if correct, 0 otherwise)
        correct = (eval_preds == eval_labels).int()
        details = {
            'eval_labels': eval_labels.cpu().tolist(),
            'eval_preds': eval_preds.cpu().tolist(),
            'correct': correct.cpu().tolist(),
            'indices': sample_indices  # Indices into the original dataset
        }
        return acc, details
    return acc

def config_das(model, layer, device, weight=None, subspace_dimension=1):
    '''The function is used to set up the configuration for DAS intervention and wrap the model as an IntervenableModel.
    Input: 
        model: the model to be used
        layer: the layer to be used for intervention
        device: the device to be used
        weight: the weight of the intervention, optional
        weight: the weight of the intervention, optional
    Output:
        intervenable: the model with the intervention
    This function will create an IntervenableModel with the given configuration.'''
    config = IntervenableConfig(
            model_type = type(model),
            representations=[
                RepresentationConfig(
                    layer,              # layer
                    "block_output",          # component
                    "pos",              # intervention unit
                    1,                  # max number of unit
                    low_rank_dimension = subspace_dimension, # low rank dimension
                    subspace_partition = [[0, subspace_dimension]],
                ),
            ],
            intervention_types=LowRankRotatedSpaceIntervention,
        )
    intervenable = IntervenableModel(config, model)
    if weight is not None:
        # Set the weight of the intervention (single layer, always index #0)
        # Use load_state_dict if weight is a state_dict (OrderedDict)
        intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#0"].rotate_layer.load_state_dict(weight)
    
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable

def config_vanilla(model, layer, device):
    '''The function is used to set up the configuration for vanilla intervention and wrap the model as an IntervenableModel.
    Input: 
        model: the model to be used
        layer: the layer to be used for intervention
        device: the device to be used
    Output:
        intervenable: the model with the intervention
    This function will create an IntervenableModel with the given configuration.'''
    config = IntervenableConfig(
            model_type = type(model),
            representations=[
                {
                    "layer": layer,              # layer
                   "component": "block_output",          # component
                    "unit": "pos",              # intervention unit
                    "max_number_of_units": 1,
                },
            ],
            intervention_types=VanillaIntervention,
        )
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable

def save_weight(weights, name: str, path: str):
    # Save the model state dict
    if not os.path.exists(path):
        os.mkdir(path)
    name = os.path.join(path, name)
    # Check that all values are dicts of state_dicts (not modules)
    for key, subdict in weights.items():
        if not isinstance(subdict, dict):
            raise RuntimeError(f"weights['{key}'] is not a dict (got {type(subdict)})")
        for subkey, value in subdict.items():
            if not isinstance(value, dict):
                raise RuntimeError(f"weights['{key}']['{subkey}'] is not a state_dict (got {type(value)})")
    # Save state dict of interventions
    torch.save(weights, name)
    print(f"Model saved to {name}")

def load_weight(name):
    # Load the model state dict
    if os.path.exists(name):
        weights = torch.load(name)
        print("Weights loaded successfully!")
    else:
        print(f"Did not find existing model from {name}")
    return weights

def find_candidate_alignments(
    model,
    dataset,
    poss,
    layers,
    batch_size,
    device,
    n_candidates = 10,
    subspace_dimension = 1,
    intervention_name = None,
    tokenizer = None,
):
    ''' This function is used to find the candidate alignments for the intervention.
    Input: 
        model: the model with the intervention
        dataset: the dataset
        poss: the positions of the intervention
        device: the device to be used
        batch_size: the batch size
        n_candidates: the number of candidates to be found
    Output:
        candidates: the candidates for the intervention
        weights: the weights of the candidates
    '''
    candidates = {}
    weights = {}
    # split dataset into training and testing
    train_dataset = dataset[:int(len(dataset) * 0.6)]
    test_dataset = dataset[int(len(dataset) * 0.6):]

    # count proprtion of differnt intervenved labels and based labels
    true_count = sum(1 for dp in test_dataset if dp["labels"] == dp["base_labels"])
    print(f"Overall Proportion of True labels in the dataset: {true_count}/{len(test_dataset)} = {true_count/len(test_dataset):.2f}")

    # Create directory for partial results if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    total_iterations = len(layers) * len(list(poss))
    current_iteration = 0

    for layer in layers:
        intervenable = config_das(model, layer, device, subspace_dimension=subspace_dimension)
        for pos in poss:
            current_iteration += 1
            print(f"\n[{current_iteration}/{total_iterations}] Processing Layer {layer}, Position {pos}")
            
            # create optimizer
            optimizer_params = []
            for k, v in intervenable.interventions.items():
                optimizer_params += [{"params": v.rotate_layer.parameters()}]
            optimizer = torch.optim.Adam(optimizer_params, lr=0.001)  # Increased from 0.001 to 0.01
            # train the model
            DAS_training(intervenable, train_dataset, optimizer, pos=pos, device=device, epochs=5, batch_size=batch_size, tokenizer=tokenizer)  # Increased from 5 to 10 epochs
            # test the model
            intervenable.disable_model_gradients()
            acc = das_test(intervenable, pos, test_dataset, device=device, batch_size=batch_size)
            candidates[(layer, pos)] = acc
            print(f"Layer {layer}, Position {pos}: Test Accuracy = {acc:.4f}")
            
            # Take a safe snapshot of the rotate_layer state_dict so later in-place
            # changes to the intervenable don't mutate previously stored weights.
            sd = intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#0"].rotate_layer.state_dict()
            weights[(layer, pos)] = {k: v.clone().detach().cpu() for k, v in sd.items()}
            
            # Save partial results after each iteration
            partial_candidates = {f"L{k[0]}_P{k[1]}": v for k, v in candidates.items()}
            partial_weights = {f"L{k[0]}_P{k[1]}": v for k, v in weights.items()}
            
            # Use intervention_name in filename to avoid overwriting results from different ops
            suffix = f"_{intervention_name}" if intervention_name else ""
            with open(f"results/candidates_partial{suffix}_{subspace_dimension}_pretrain.json", "w") as f:
                json.dump(partial_candidates, f, indent=4)
            
            torch.save(partial_weights, f"results/weights_partial{suffix}_{subspace_dimension}_pretrain.pt")
            print(f"Partial results saved ({current_iteration}/{total_iterations} completed)")

    # sort the candidates by accuracy
    candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    # keep only the top n_candidates
    candidates = candidates[:n_candidates]
    # convert to dict
    candidates = {f"L{k[0]}_P{k[1]}": v for k, v in candidates}

    # keep the corresponding weights of the candidates
    weights = {f"L{k[0]}_P{k[1]}": v for k, v in weights.items() if f"L{k[0]}_P{k[1]}" in candidates.keys()}

    return candidates, weights

def extract_layer_pos(string):
    layer, pos = string.split('_')  # Split on underscore -> ["L5", "P78"]
    layer_num = int(layer[1:])      # Remove "L" and convert to int -> 5
    pos_num = int(pos[1:])          # Remove "P" and convert to int -> 78
    return layer_num, pos_num

def select_candidates(node, candidates, causal_model,dataset_generator, weights):
    ''' This function is used to select the candidates for the intervention.
    Input: 
        candidates: the candidates for the intervention
        weights: the weights of the candidates
        dataset_generator: the dataset generator input: intervention
    Output:
        selected_candidates: the selected candidates for the intervention
    More work needs to be done here.
    '''
    children = causal_model.paraents[node]
    if len(children) == 0:
        candidates = candidates[node]
        # return the candidiate with highest accuracy
        selected_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        selected_candidates = selected_candidates[0]
        return selected_candidates
    
    dataset = dataset_generator(children)

    rec_score = {}

    for candidate in candidates[node].keys():
        # extract layer and pos from candidate
        layer, pos = extract_layer_pos(candidate)

        # TBD 

def convert_to_binary_features(features_list, variable_names=['t0', 't1', 't2', 't3']):
    """
    Convert true/false string features (t0-t3) to binary features.
    
    Args:
        features_list: List of dicts with true/false string values for t0-t3
        variable_names: List of variable names to convert
        
    Returns:
        List of dicts with binary features, and None for mapping (not needed for true/false)
    """
    binary_features_list = []
    for features in features_list:
        binary_features = {}
        for var in variable_names:
            if var in features:
                value = str(features[var]).lower()
                # Convert true/false strings to binary: 1 for true, 0 for false
                binary_features[var] = 1 if value == 'true' else 0
        binary_features_list.append(binary_features)
    
    return binary_features_list, None

def extract_first_layer_activations(model, dataset, device, tokenizer, batch_size=64):
    """
    Extract activations from the first layer (layer 0) at the last token position.
    Extracts activations from both base input (input_ids) and source input (source_input_ids).
    
    Args:
        model: The language model
        dataset: Dataset containing input_ids and source_input_ids
        device: Device to run on
        tokenizer: Tokenizer to get pad_token_id
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (base_activations_list, source_activations_list), each containing
        activation tensors (one per sample), each of shape (hidden_size,)
    """
    base_activations_list = []
    source_activations_list = []
    model.eval()
    
    with torch.no_grad():
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        for batch in tqdm(data_loader, desc="Extracting activations", leave=False):
            batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
            current_batch_size = batch["input_ids"].shape[0]
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            source_input_ids = batch["source_input_ids"].to(device)
            
            # Get the last token position for each sample in the batch
            # Find the last non-padding token position
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            if pad_token_id is None:
                # If no pad token, use sequence length - 1
                seq_lengths = torch.tensor([input_ids.shape[1] - 1] * current_batch_size)
                source_seq_lengths = torch.tensor([source_input_ids.shape[1] - 1] * current_batch_size)
            else:
                # Find last non-padding position
                mask = (input_ids != pad_token_id).long()
                seq_lengths = mask.sum(dim=1) - 1  # -1 for 0-indexed
                source_mask = (source_input_ids != pad_token_id).long()
                source_seq_lengths = source_mask.sum(dim=1) - 1  # -1 for 0-indexed
            last_positions = seq_lengths.tolist()
            source_last_positions = source_seq_lengths.tolist()
            
            # Run model forward pass to get hidden states for base input
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
            
            # Extract activations from layer 0 (first layer) at last token position for base input
            layer_0_hidden = hidden_states[0]  # Shape: (batch_size, seq_len, hidden_size)
            
            for i, last_pos in enumerate(last_positions):
                activation = layer_0_hidden[i, last_pos, :].cpu()  # Shape: (hidden_size,)
                base_activations_list.append(activation)
            
            # Run model forward pass to get hidden states for source input
            source_outputs = model(source_input_ids, output_hidden_states=True)
            source_hidden_states = source_outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
            
            # Extract activations from layer 0 (first layer) at last token position for source input
            source_layer_0_hidden = source_hidden_states[0]  # Shape: (batch_size, seq_len, hidden_size)
            
            for i, source_last_pos in enumerate(source_last_positions):
                source_activation = source_layer_0_hidden[i, source_last_pos, :].cpu()  # Shape: (hidden_size,)
                source_activations_list.append(source_activation)
    
    return base_activations_list, source_activations_list

def test_with_weights(model, layer, device, pos, test_dataset, batch_size=64, intervention_type = 'das', weight = None, subspace_dimension=1, return_details=False):
    ''' This function is used to test the model with pre-trained intervention weights.
    Input:
        model: the model to be used
        layer: the layer to be used for intervention
        device: the device to be used
        pos: the position of the intervention, int
        test_dataset: the testing dataset, contain input_ids, source_input_ids, labels
        weight: the pre-trained weight (state_dict) of the intervention
        batch_size: the batch size to be used, int
        return_details: if True, also return per-sample correctness (for creating analysis datasets)
    Output:
        acc: the accuracy of the model, float
        (optional) details: dict with 'eval_labels', 'eval_preds', 'correct' if return_details=True
    This function will create an intervenable model with pre-trained weights and test its accuracy.'''
    
    # Create intervenable model with the pre-trained weight
    if intervention_type == 'das':
        intervenable = config_das(model, layer, device, weight, subspace_dimension=subspace_dimension)
    elif intervention_type == 'vanilla':
        intervenable = config_vanilla(model, layer, device)
    else:
        raise ValueError("intervention_type must be 'das' or 'vanilla'")
    
    # Test the model using the existing das_test function
    result = das_test(intervenable, pos, test_dataset, device=device, batch_size=batch_size, intervention_type=intervention_type, return_details=return_details)
    
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DAS training or testing with selectable causal model")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Run training to find candidate alignments")
    mode_group.add_argument("--test", action="store_true", help="Run testing using precomputed weights and candidates")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-14B", help="HuggingFace model ID to use")
    parser.add_argument("--causal-model", choices=["1", "2"], default="1", help="Which causal model to use: 1 (default) or 2")
    parser.add_argument("--intervention-type", type=str, choices=["das", "vanilla", "boundless"], default='das', help="Type of intervention: 'das' (fixed subspace), 'vanilla' (full vector), or 'boundless' (auto feature selection)")
    parser.add_argument("--weights-path", type=str, default=None, help="Path to das weights (.pt) for test mode")
    parser.add_argument("--candidates-path", type=str, default=None, help="Path to candidates JSON for test mode")
    parser.add_argument("--data-size", type=int, default=1024, help="Number of examples to generate per dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataset creation and evaluation")
    parser.add_argument("--subspace-dimension", type=int, default=1, help="Dimension of the subspace for intervention")
    parser.add_argument("--device", type=int, default=0, help="Device to use (0 refers to cuda:0, -2 refer auto, -1 refers cpu). If not set, auto-detects.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--local-model-dir", type=str, default=None, help="Directory to load/save model snapshot (speeds up subsequent runs)")
    parser.add_argument("--hf-cache-dir", type=str, default=None, help="HuggingFace cache directory (sets HF_HOME env var, must be set before imports)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use for model parallelism (use -1 for auto, 0 for CPU)")
    parser.add_argument("--sparsity-coef", type=float, default=0.01, help="Sparsity coefficient for boundless DAS L1 regularization")
    parser.add_argument("--temperature-start", type=float, default=1.0, help="Starting temperature for boundless DAS mask annealing")
    parser.add_argument("--temperature-end", type=float, default=0.01, help="Ending temperature for boundless DAS mask annealing")
    parser.add_argument("--layer-start", type=int, default=None, help="Starting layer index (inclusive) for parallel layer search")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (exclusive) for parallel layer search")
    parser.add_argument("--pos-start", type=int, default=None, help="Starting position index (inclusive) for intervention search (overrides auto-detection)")
    parser.add_argument("--pos-end", type=int, default=None, help="Ending position index (exclusive) for intervention search (overrides auto-detection)")
    parser.add_argument("--test-layer", type=int, default=None, help="Specific layer to test (for test mode only, overrides all candidates)")
    parser.add_argument("--test-pos", type=int, default=None, help="Specific position to test (for test mode only, overrides all candidates)")
    args = parser.parse_args()

    # Set random seed early for reproducibility
    set_random_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # construct causal model based on selection
    if args.causal_model == "1":
        or_causal_model = util_data.build_causal_model()
    elif args.causal_model == "2":        
        or_causal_model = util_data.build_causal_model2()
    else:
        raise RuntimeError(f"Unsupported causal model selection: {args.causal_model}")
    
    # load trained model with multi-GPU support
    model, tokenizer = util_model.get_model_and_tokenizer(
        args.model_id, 
        hf_token=os.environ.get("HF_TOKEN"), 
        local_dir=args.local_model_dir,
        num_gpus=args.num_gpus,
        gpu_id=args.device
    )
    
    # Determine device for tensors - get from model's input embeddings for multi-GPU compatibility
    if args.num_gpus == -1 or args.num_gpus > 1:
        # Model is distributed across GPUs via device_map="auto"
        # Get the device where inputs should be sent (where embeddings are)
        device = get_model_input_device(model)
        print(f"Model distributed across GPUs (num_gpus={args.num_gpus})")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Input device (from model embeddings): {device}")
    elif args.num_gpus == 1:
        device = get_model_input_device(model)
        print(f"Using device: {device}")
    else:  # num_gpus == 0
        device = torch.device("cpu")
        print("Using CPU")

    # create dataset params
    data_size = args.data_size
    batch_size = args.batch_size

    weights = {}

    # Handle different model configs: Qwen uses num_hidden_layers, GPT-2 uses n_layer
    num_layers = getattr(model.config, 'num_hidden_layers', getattr(model.config, 'n_layer', None))
    if num_layers is None:
        raise ValueError("Could not determine number of layers from model config")
    
    # Apply layer filtering if specified
    layer_start = args.layer_start if args.layer_start is not None else 0
    layer_end = args.layer_end if args.layer_end is not None else num_layers
    layers = range(layer_start, layer_end)
    
    # Build layer suffix for output file naming (for parallel runs)
    if args.layer_start is not None or args.layer_end is not None:
        layer_suffix = f"_L{layer_start}-{layer_end}"
        print(f"Processing layers {layer_start} to {layer_end-1} (total: {len(layers)} layers)")
    else:
        layer_suffix = None
        print(f"Processing all {num_layers} layers")

    if args.causal_model == "1":
        op_list = ["op2"] #["op1", "op2", "op3", "op4", "op5"]
        data_generator = "all"
        out_op = "op5"
    elif args.causal_model == "2":
        op_list = ["op4a", "op5a"]
        data_generator = "all2"
        out_op = "op6a"
    causal_model_tag = f"or_model_{args.causal_model}"
    intervention_type = args.intervention_type
    subspace_dimension=args.subspace_dimension

    if args.train:
        print("Starting training (finding candidate alignments)")
        candidates_total = {}
        das_weights = {}
        feature_counts_total = {}  # For boundless DAS

        # Import boundless DAS if needed
        if intervention_type == 'boundless':
            from boundless_das import (
                find_candidate_alignments_boundless,
            )
            print("Using Boundless DAS (auto feature selection)")

        for intervention in op_list:
            candidates_total[intervention] = {}
            das_weights[intervention] = {}
            dataset = util_data.make_counterfactual_dataset(
                data_generator,
                intervention,
                out_op,
                or_causal_model,
                model,
                tokenizer,
                data_size,
                device,
                batch_size=batch_size,
            )
            pos_after_sub_token, input_length, token_mapping = find_positional_indices(
                sample_input_id=dataset[0]["input_ids"],
                tokenizer=tokenizer,
                sub_token='logic_function(',
                return_token_mapping=True
            )
            pos_after_sub_token2, input_length2 = find_positional_indices(
                sample_input_id=dataset[1]["input_ids"],
                tokenizer=tokenizer,
                sub_token='logic_function('
            )
            if pos_after_sub_token != pos_after_sub_token2:
                raise RuntimeError("The position after sub_token is not consistent across samples.")
            
            # Use user-specified position range if provided, otherwise auto-detect
            if args.pos_start is not None and args.pos_end is not None:
                poss = range(args.pos_start, args.pos_end)
                print(f"Using user-specified position range: {args.pos_start} to {args.pos_end-1}")
            else:
                poss = range(pos_after_sub_token+7, input_length)
                print(f"Auto-detected position range: {pos_after_sub_token+7} to {input_length-1}")
            
            # Save the position-to-token mapping to a separate file
            os.makedirs("training_results", exist_ok=True)
            token_mapping_file = f"training_results/position_token_mapping_{intervention}_{causal_model_tag}.json"
            with open(token_mapping_file, "w") as f:
                json.dump(token_mapping, f, indent=4)
            print(f"Position-to-token mapping saved to {token_mapping_file}")
            print(f"Searching positions from {poss.start} to {poss.stop-1} for intervention {intervention}")

            print(f"Dataset created for {intervention}")
            print(f"Finding candidates for {intervention}")
            
            if intervention_type == 'boundless':
                # Use boundless DAS (auto feature selection)
                candidate, weight, feature_counts = find_candidate_alignments_boundless(
                    model,
                    dataset,
                    poss,
                    layers,
                    batch_size,
                    device,
                    n_candidates=len(layers)*len(poss),
                    intervention_name=intervention,
                    tokenizer=tokenizer,
                    sparsity_coef=args.sparsity_coef,
                    epochs=5,
                    layer_suffix=layer_suffix,
                )
                candidates_total[intervention].update(candidate)
                das_weights[intervention].update(weight)
                feature_counts_total[intervention] = feature_counts
            elif intervention_type == 'vanilla':
                # Vanilla intervention doesn't need training - skip and go directly to test mode
                print(f"Skipping training for vanilla intervention '{intervention}' - use --test mode directly")
                # Generate placeholder candidates (all layer/pos combinations) for test mode
                for layer in layers:
                    for pos in poss:
                        candidates_total[intervention][f"L{layer}_P{pos}"] = 0.0  # Placeholder accuracy
                # No weights for vanilla intervention
            else:
                # Use standard DAS (fixed subspace dimension)
                candidate, weight = find_candidate_alignments(
                    model,
                    dataset,
                    poss,
                    layers,
                    batch_size,
                    device,
                    n_candidates=len(layers)*len(poss),
                    subspace_dimension=args.subspace_dimension,
                    intervention_name=intervention,
                    tokenizer=tokenizer
                )
                candidates_total[intervention].update(candidate)
                das_weights[intervention].update(weight)

        # persist results
        os.makedirs("training_results", exist_ok=True)
        
        if intervention_type == 'boundless':
            # Save boundless DAS results
            with open(f"training_results/candidates_boundless_{causal_model_tag}_pretrain.json", "w") as f:
                json.dump(candidates_total, f, indent=4)
            print(f"Candidate alignments saved to training_results/candidates_boundless_{causal_model_tag}_pretrain.json")

            with open(f"training_results/das_weights_boundless_{causal_model_tag}_pretrain.pt", "wb") as f:
                torch.save(das_weights, f)
            print(f"Boundless DAS weights saved to training_results/das_weights_boundless_{causal_model_tag}_pretrain.pt")
            
            with open(f"training_results/feature_counts_boundless_{causal_model_tag}_pretrain.json", "w") as f:
                json.dump(feature_counts_total, f, indent=4)
            print(f"Feature counts saved to training_results/feature_counts_boundless_{causal_model_tag}_pretrain.json")
        elif intervention_type == 'vanilla':
            # Save vanilla results (no weights needed)
            with open(f"training_results/candidates_vanilla_{causal_model_tag}_pretrain.json", "w") as f:
                json.dump(candidates_total, f, indent=4)
            print(f"Candidate alignments saved to training_results/candidates_vanilla_{causal_model_tag}_pretrain.json")
            # No weights to save for vanilla intervention
        else:
            # Save standard DAS results
            with open(f"training_results/candidates_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}_pretrain.json", "w") as f:
                json.dump(candidates_total, f, indent=4)
            print(f"Candidate alignments saved to training_results/candidates_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}_pretrain.json")

            with open(f"training_results/das_weights_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}_pretrain.pt", "wb") as f:
                torch.save(das_weights, f)
            print(f"DAS weights saved to training_results/das_weights_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}_pretrain.pt")

    elif args.test:
        print("Starting testing using provided weights and candidates")
        
        # Import boundless DAS if needed
        if intervention_type == 'boundless':
            from boundless_das import test_with_boundless_weights
            print("Using Boundless DAS for testing")
        
        if args.intervention_type == 'das':
            if not args.weights_path:
                args.weights_path = f"training_results/das_weights_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}_pretrain.pt"
            if not args.candidates_path:
                args.candidates_path = f"training_results/candidates_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}_pretrain.json"

            # load provided artifacts
            das_weights = load_weight(args.weights_path)
            with open(args.candidates_path, "r") as f:
                candidates_total = json.load(f)
        
        elif args.intervention_type == 'boundless':
            if not args.weights_path:
                args.weights_path = f"result_op2/L20_P368_extracted.pt"
            if not args.candidates_path:
                args.candidates_path = f"result_op2/candidates_high.json"

            # load provided artifacts
            das_weights = load_weight(args.weights_path)
            with open(args.candidates_path, "r") as f:
                candidates_total = json.load(f)
        
        elif args.intervention_type == 'vanilla':
            # Vanilla intervention doesn't need weights or pre-trained candidates
            das_weights = {}  # Empty dict, vanilla doesn't use weights
            
            if args.candidates_path:
                # If user provides a candidates file, use it
                with open(args.candidates_path, "r") as f:
                    candidates_total = json.load(f)
                print(f"Loaded candidates from {args.candidates_path}")
            else:
                # Auto-generate all layer/pos combinations
                # We'll populate candidates_total per intervention after creating the dataset
                candidates_total = None  # Will be generated per intervention
                print("Vanilla intervention: will auto-generate all layer/pos combinations")
            print("Using Vanilla intervention for testing (no weights needed)")

        test_results = {}
        # Dictionary to store analysis datasets for each (pos, layer) combination
        # Structure: {intervention: {source_base_key: {candidate: {features: [...], labels: [...]}}}}
        analysis_datasets = {}

        if args.causal_model == "1":
            data_generator = "all"

        elif args.causal_model == "2":
            data_generator = "exhaustive2"

        for intervention in op_list:
            # types = util_data.corresponding_intervention(intervention)
            test_results[intervention] = {}
            analysis_datasets[intervention] = {}
            # for source_code, base_code in types:
            print(f"Creating dataset for {intervention}")
            # Get both tokenized and raw dataset (with t0-t3 features)
            dataset, raw_dataset = util_data.make_counterfactual_dataset(
                data_generator,
                intervention,
                out_op,
                or_causal_model,
                model,
                tokenizer,
                data_size,
                device,
                batch_size=batch_size,
                # source_code=source_code,
                # base_code=base_code,
                return_raw=True,
            )
            print(f"Dataset created for {intervention}\r")

            # Add idx field to each sample in dataset for proper tracking
            for i, sample in enumerate(dataset):
                sample["idx"] = i

            # Extract t0-t3 features from raw dataset
            # For causal_model 1: t0, t1, t2, t3 are strings
            # For causal_model 2: t0-t5 are from vocab (strings)
            features_list = []
            for dp in raw_dataset:
                input_ids = dp["input_ids"]
                source_input_ids = dp["source_input_ids"][0]
                if args.causal_model == "1":
                    # For causal model 1, t0-t3 are string values
                    features = {
                        "p": int(input_ids["t0"] == input_ids["t1"]),
                        "q": int(input_ids["t2"] == input_ids["t3"]),
                        "r": int(input_ids["t0"] == input_ids["t3"]),
                        "ps": int(source_input_ids["t0"] == source_input_ids["t1"]),
                        "qs": int(source_input_ids["t2"] == source_input_ids["t3"]),
                        "rs": int(source_input_ids["t0"] == source_input_ids["t3"]),
                    }
                else:
                    raise ValueError(f"Unsupported causal model for now: {args.causal_model}")
                features_list.append(features)

            # Convert t0-t3 to binary features
            # binary_features_list, value_mapping = convert_to_binary_features(features_list)
            binary_features_list = features_list
            # Extract activations from first layer on last token for both base and source inputs
            print(f"Extracting activations from first layer for {intervention}")
            base_activation_list, source_activation_list = extract_first_layer_activations(model, dataset, device, tokenizer, batch_size=batch_size)
            
            # Verify that activation lists match binary features list
            if len(base_activation_list) != len(binary_features_list):
                raise ValueError(f"Mismatch: {len(base_activation_list)} base activations but {len(binary_features_list)} feature samples")
            if len(source_activation_list) != len(binary_features_list):
                 raise ValueError(f"Mismatch: {len(source_activation_list)} source activations but {len(binary_features_list)} feature samples")
            
            # Add activation features to binary features (both base and source)
            for i, (binary_features, base_activation, source_activation) in enumerate(zip(binary_features_list, base_activation_list, source_activation_list)):
                # Convert activation tensors to lists for JSON serialization
                binary_features["activation_base"] = base_activation.tolist()
                binary_features["activation_source"] = source_activation.tolist()

            # get the candidates for this intervention
            if candidates_total is None and intervention_type == 'vanilla':
                # Auto-generate all layer/pos combinations for vanilla
                # First, get pos range from the dataset (same logic as training mode)
                pos_after_sub_token, input_length = find_positional_indices(
                    sample_input_id=dataset[0]["input_ids"],
                    tokenizer=tokenizer,
                    sub_token='logic_function('
                )
                if args.pos_start is not None and args.pos_end is not None:
                    poss = range(args.pos_start, args.pos_end)
                else:
                    poss = range(pos_after_sub_token+7, input_length)
                
                # Generate all layer/pos combinations
                candidates = {f"L{layer}_P{pos}": 0.0 for layer in layers for pos in poss}
                print(f"Auto-generated {len(candidates)} layer/pos combinations for vanilla testing")
            else:
                candidates = candidates_total.get(intervention, {})
            weights_for_intervention = das_weights.get(intervention, {}) if isinstance(das_weights, dict) else das_weights

            # Filter candidates by layer and position if specified
            if args.test_layer is not None or args.test_pos is not None:
                filtered_candidates = {}
                for candidate in candidates.keys():
                    layer, pos = extract_layer_pos(candidate)
                    # Check if this candidate matches the specified layer and/or position
                    layer_match = (args.test_layer is None) or (layer == args.test_layer)
                    pos_match = (args.test_pos is None) or (pos == args.test_pos)
                    if layer_match and pos_match:
                        filtered_candidates[candidate] = candidates[candidate]
                candidates = filtered_candidates
                if len(candidates) == 0:
                    print(f"Warning: No candidates found matching layer={args.test_layer}, pos={args.test_pos}")
                    continue
                else:
                    print(f"Filtered to {len(candidates)} candidate(s) matching layer={args.test_layer}, pos={args.test_pos}")

            results = {}
            
            for candidate in candidates.keys():
                layer, pos = extract_layer_pos(candidate)
                weight = weights_for_intervention.get(candidate)
                # Skip weight check for vanilla intervention (it doesn't use weights)
                if weight is None and intervention_type not in ['vanilla']:
                    print(f"Warning: weight for candidate {candidate} not found; skipping")
                    continue
                
                # Get accuracy and detailed per-sample results
                if intervention_type == 'boundless':
                    acc, details = test_with_boundless_weights(
                        model,
                        layer,
                        device,
                        pos,
                        dataset,
                        batch_size=batch_size,
                        weight=weight,
                        return_details=True
                    )
                elif intervention_type == 'vanilla':
                    acc, details = test_with_weights(
                        model,
                        layer,
                        device,
                        pos,
                        dataset,
                        batch_size=batch_size,
                        intervention_type='vanilla',
                        weight=None,  # Vanilla doesn't use weights
                        return_details=True
                    )
                else:
                    acc, details = test_with_weights(
                        model,
                        layer,
                        device,
                        pos,
                        dataset,
                        batch_size=batch_size,
                        intervention_type=intervention_type,
                        weight=weight,
                        subspace_dimension=args.subspace_dimension,
                        return_details=True
                    )
                print(f"Candidate: {candidate}, Accuracy: {acc:.4f} \r")
                results[candidate] = acc
                
                # Create analysis dataset for this (pos, layer) combination
                # Features: binary features for t0-t3 + activations from first layer (both base and source inputs)
                # Label: binary (1 if eval_labels == pred_label, 0 otherwise)
                # Use indices from details to align features with results
                aligned_binary_features = [binary_features_list[i] for i in details["indices"]]
                analysis_datasets[intervention][candidate] = {
                    "features": aligned_binary_features,
                    "labels": details["correct"],  # Binary: 1 if correct, 0 otherwise
                    "layer": layer,
                    "pos": pos,
                    "accuracy": acc
                }

            test_results[intervention] = results

            # Save partial results after each evaluation
            os.makedirs("test_results", exist_ok=True)
            with open(f"test_results/test_results_partial_{intervention_type}_{causal_model_tag}.json", "w") as f:
                json.dump(test_results, f, indent=4)
            print(f"Partial test results saved after {intervention}")

        # Save final test results
        if intervention_type == 'boundless':
            result_suffix = f"boundless_{causal_model_tag}"
        else:
            result_suffix = f"{intervention_type}_{causal_model_tag}_dim{subspace_dimension}"
        
        with open(f"test_results/test_results_{result_suffix}.json", "w") as f:
            json.dump(test_results, f, indent=4)
        print(f"Final test results saved to test_results/test_results_{result_suffix}.json")
        
        # Save analysis datasets to a single file
        # Convert to a format suitable for JSON serialization
        with open(f"test_results/analysis_datasets_{result_suffix}.json", "w") as f:
            json.dump(analysis_datasets, f, indent=4)
        print(f"Analysis datasets saved to test_results/analysis_datasets_{result_suffix}.json")

    # Release the GPU memory
    # For multi-GPU models with device_map, we can't simply call model.cpu()
    # Just clear the cache instead
    del model
    torch.cuda.empty_cache()
    print("GPU memory cleared")



"""
Code for parallel intervention (not used currently)
# The end
# Parallel intervention code that may be useful later:
# def config_das_parallel(model, layers, device, weights=None):
#     ''' This function is used to set up the configuration for parallel interchange intervention.
#     Input: 
#         model: the model to be used
#         locs: list of layer, the locations of the intervention
#         device: the device to be used
#         weights: list the weights of the intervention, in the same order as layers
#     Output:
#         intervenable: the model with the intervention
#     '''
#     representations = []
#     for layer in layers:
#         representations.append(
#             RepresentationConfig(
#                 layer,              # layer
#                 "block_output",          # component
#                 "pos",              # intervention unit
#                 1,                  # max number of unit
#                 low_rank_dimension = 1, # low rank dimension
#                 subspace_partition = [[0, 1]],
#                 intervention_link_key=0
#             )
#         )
#     config = IntervenableConfig(
#             model_type = type(model),
#             representations=representations,
#             intervention_types=LowRankRotatedSpaceIntervention,
#         )
    
#     intervenable = IntervenableModel(config, model)
#     if weights is not None:
#         # Set the weights of the intervention
#         rec_layer = {}
#         for i, layer in enumerate(layers):
#             if layer not in rec_layer:
#                 rec_layer[layer] = 0
#             else:
#                 rec_layer[layer] += 1
#             # Use load_state_dict if weights[i] is a state_dict (OrderedDict)
#             intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#{rec_layer[layer]}"].rotate_layer.load_state_dict(weights[i])
#             # Use load_state_dict if weights[i] is a state_dict (OrderedDict)
#             intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#{rec_layer[layer]}"].rotate_layer.load_state_dict(weights[i])
        
#     intervenable.set_device(device)
#     intervenable.disable_model_gradients()
#     return intervenable

# def parallel_intervention(intervenable, poss, test_dataset, batch_size):
#     ''' This function is used to set up the parallel intervention.
#     Input: 
#         intervenable: the model with the intervention
#         pos: the position of the intervention
#         batch_size: the batch size
#     Output:
#         acc: the accuracy of the model
#     '''

#     eval_labels = []
#     eval_preds = []
#     n_blocks = len(poss)
#     with torch.no_grad():
#         epoch_iterator = tqdm(
#             DataLoader(
#                 test_dataset,
#                 batch_size=batch_size,
#                 sampler=batched_random_sampler(test_dataset, batch_size),
#             ),
#             desc=f"Testing",
#             position=0,
#             leave=False,
#         )

#         for batch in epoch_iterator:
#             batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
#             batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
#             #print(batch["input_ids"].shape, batch["source_input_ids"].shape)
#             batch_size = batch["input_ids"].shape[0]
#             for k, v in batch.items():
#                 if v is not None and isinstance(v, torch.Tensor):
#                     batch[k] = v.to("cuda")

#             _, counterfactual_outputs = intervenable(
#                 {"input_ids": batch["input_ids"]},
#                 [
#                     {"input_ids": batch["source_input_ids"]},
#                 ] * n_blocks,
#                 {
#                     "sources->base": tuple(
#                        [[[[pos]] * batch_size] * 2 for pos in poss]
#                     )
#                 },
#                 subspaces=[
#                     [[0]] * batch_size,
#                 ] * n_blocks,
#             )
#             eval_labels += [batch["labels"].squeeze()]
#             eval_preds += [counterfactual_outputs.logits[:,-1,:].argmax(dim=-1)]
#     eval_labels = torch.cat(eval_labels)
#     eval_preds = torch.cat(eval_preds)
#     acc = compute_metrics(eval_preds, eval_labels)["accuracy"]
#     return acc
"""