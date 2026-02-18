import util_model as util_model
import util_data as util_data
import torch
import os
import random
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

def DAS_training(intervenable, train_dataset, optimizer, pos, epochs = 5, batch_size = 64, gradient_accumulation_steps = 1):
    '''Main code for training the model with DAS intervention.
    Input:
        intervenable: the model with the intervention, pyvene.IntervenableModel
        train_dataset: the training dataset, contain input_ids, source_input_ids, labels
        optimizer: the optimizer to be used, torch.optim.Adam
        pos: the position of the intervention, int
        epochs: the number of epochs to train, int
        batch_size: the batch size to be used, int
        gradient_accumulation_steps: the number of steps to accumulate gradients, int
    Output:
        None, the model will be trained in-place.
    This function will train the model with the intervention, and compute the loss and accuracy.
    '''
    
    intervenable.model.train()  # set the module to train mode, which enables drop-off but no grads
    print("intervention trainable parameters: ", intervenable.count_parameters()) # count the number of trainable parameters in the intervention
    device = next(intervenable.model.parameters()).device

    train_iterator = trange(0, int(epochs), desc="Epoch")  # create a progress bar for the epochs
    total_step = 0
    for epoch in train_iterator:
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
            batch_size = batch["input_ids"].shape[0]
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
                        [[[pos]] * batch_size], [[[pos]] * batch_size],
                    )
                },
                subspaces=[
                    [[0]] * batch_size,
                ],
            )
            # compute metrics
            eval_metrics = compute_metrics(
            counterfactual_outputs.logits[:,-1,:].argmax(dim=-1), batch["labels"].squeeze()
            )

            # loss and backprop
            loss = compute_loss(
                counterfactual_outputs.logits[:,-1,:], batch["labels"].squeeze()
            )

            epoch_iterator.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{eval_metrics['accuracy']:.4f}"},
                refresh=True
            )
            train_iterator.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{eval_metrics['accuracy']:.4f}"},
                refresh=True
            )

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            if total_step % gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()
            total_step += 1

        epoch_iterator.close()  # Close inner progress bar after each epoch

def _pad_and_stack(sequences, pad_token_id, left_pad=True):
    """Pad a list of 1D or 2D tensors to the same length and stack. left_pad=True for causal LM (last pos = real token)."""
    if not sequences:
        return None
    tensors = [s.squeeze() for s in sequences]
    if tensors[0].dim() == 0:
        return torch.stack(tensors)
    max_len = max(t.size(-1) for t in tensors)
    padded = []
    for t in tensors:
        if t.size(-1) < max_len:
            pad_len = max_len - t.size(-1)
            pad = torch.full((pad_len,) + t.shape[:-1], pad_token_id, dtype=t.dtype, device=t.device)
            if left_pad:
                t = torch.cat([pad, t], dim=-1)
            else:
                t = torch.cat([t, pad], dim=-1)
        padded.append(t)
    return torch.stack(padded)


def das_test_batch_correctness(intervenable, pos, batch_list, device, intervention_type='das', pad_token_id=None):
    """Run DAS on a list of tokenized examples and return per-example correctness.
    Each element of batch_list is a dict with keys: input_ids, source_input_ids, labels (tensors).
    Processes in a single batched forward when possible. Returns list of bool, same length as batch_list."""
    if not batch_list:
        return []
    if pad_token_id is None and hasattr(intervenable, 'model') and hasattr(intervenable.model, 'config'):
        pad_token_id = getattr(intervenable.model.config, 'pad_token_id', 0)
    if pad_token_id is None:
        pad_token_id = 0

    with torch.no_grad():
        # Batch: pad and stack
        input_ids_list = [ex["input_ids"].to(device) for ex in batch_list]
        source_list = [ex["source_input_ids"].to(device) for ex in batch_list]
        labels_list = [ex["labels"].to(device) for ex in batch_list]

        batched_input_ids = _pad_and_stack(input_ids_list, pad_token_id, left_pad=True)
        batched_source = _pad_and_stack(source_list, pad_token_id, left_pad=True)
        batch_size = batched_input_ids.shape[0]

        # Target label: last token of each label sequence (next-token prediction)
        target_labels = []
        for lab in labels_list:
            l = lab.squeeze()
            if l.numel() == 1:
                target_labels.append(l.item())
            else:
                target_labels.append(l.flatten()[-1].item())
        target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

        for k, v in [("input_ids", batched_input_ids), ("source", batched_source)]:
            if v is not None and isinstance(v, torch.Tensor):
                pass  # already on device
        batched_input_ids = batched_input_ids.to(device)
        batched_source = batched_source.to(device)

        if intervention_type == 'das':
            _, counterfactual_outputs = intervenable(
                {"input_ids": batched_input_ids},
                [{"input_ids": batched_source}],
                {"sources->base": ([[[pos]] * batch_size], [[[pos]] * batch_size])},
                subspaces=[[[0]] * batch_size],
            )
        else:
            _, counterfactual_outputs = intervenable(
                {"input_ids": batched_input_ids},
                [{"input_ids": batched_source}],
                {"sources->base": ([[[pos]] * batch_size], [[[pos]] * batch_size])},
            )
        preds = counterfactual_outputs.logits[:, -1, :].argmax(dim=-1)
        correct = (preds == target_labels).cpu().tolist()
    return correct


def das_test(intervenable, pos, test_dataset, batch_size = 64, intervention_type = 'das'):
    ''' This function is used to test the model with the intervention.
    Input:
        intervenable: the model with the intervention, pyvene.IntervenableModel
        pos: the position of the intervention, int
        test_dataset: the testing dataset, contain input_ids, source_input_ids, labels
        batch_size: the batch size to be used, int
    Output:
        acc: the accuracy of the model, float
    This function will test the model with the intervention, and compute the accuracy.'''
    eval_labels = []
    eval_preds = []
    device = next(intervenable.model.parameters()).device
    with torch.no_grad():
        epoch_iterator = tqdm(
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(test_dataset,batch_size),
            ),
            desc=f"Testing",
            position=0,
            leave=False,
        )

        for batch in epoch_iterator:
            batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
            #print(batch["input_ids"].shape, batch["source_input_ids"].shape)
            batch_size = batch["input_ids"].shape[0]
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
                            [[[pos]] * batch_size], [[[pos]] * batch_size],
                        )
                    },
                    subspaces=[
                        [[0]] * batch_size,
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
                            [[[pos]] * batch_size], [[[pos]] * batch_size],
                        )
                    }
                )
            else:
                raise ValueError("intervention_type must be 'das' or 'vanilla'")
            eval_labels += [batch["labels"].squeeze()]
            eval_preds += [counterfactual_outputs.logits[:,-1,:].argmax(dim=-1)]
    eval_labels = torch.cat(eval_labels)
    eval_preds = torch.cat(eval_preds)
    acc = compute_metrics(eval_preds, eval_labels)["accuracy"]
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

def config_das_parallel(model, layers, device, weights=None):
    ''' This function is used to set up the configuration for parallel interchange intervention.
    Input: 
        model: the model to be used
        locs: list of layer, the locations of the intervention
        device: the device to be used
        weights: list the weights of the intervention, in the same order as layers
    Output:
        intervenable: the model with the intervention
    '''
    representations = []
    for layer in layers:
        representations.append(
            RepresentationConfig(
                layer,              # layer
                "block_output",          # component
                "pos",              # intervention unit
                1,                  # max number of unit
                low_rank_dimension = 1, # low rank dimension
                subspace_partition = [[0, 1]],
                intervention_link_key=0
            )
        )
    config = IntervenableConfig(
            model_type = type(model),
            representations=representations,
            intervention_types=LowRankRotatedSpaceIntervention,
        )
    
    intervenable = IntervenableModel(config, model)
    if weights is not None:
        # Set the weights of the intervention
        rec_layer = {}
        for i, layer in enumerate(layers):
            if layer not in rec_layer:
                rec_layer[layer] = 0
            else:
                rec_layer[layer] += 1
            # Use load_state_dict if weights[i] is a state_dict (OrderedDict)
            intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#{rec_layer[layer]}"].rotate_layer.load_state_dict(weights[i])
            # Use load_state_dict if weights[i] is a state_dict (OrderedDict)
            intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#{rec_layer[layer]}"].rotate_layer.load_state_dict(weights[i])
        
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable

def parallel_intervention(intervenable, poss, test_dataset, batch_size):
    ''' This function is used to set up the parallel intervention.
    Input: 
        intervenable: the model with the intervention
        pos: the position of the intervention
        batch_size: the batch size
    Output:
        acc: the accuracy of the model
    '''

    eval_labels = []
    eval_preds = []
    n_blocks = len(poss)
    device = next(intervenable.model.parameters()).device
    with torch.no_grad():
        epoch_iterator = tqdm(
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(test_dataset, batch_size),
            ),
            desc=f"Testing",
            position=0,
            leave=False,
        )

        for batch in epoch_iterator:
            batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
            #print(batch["input_ids"].shape, batch["source_input_ids"].shape)
            batch_size = batch["input_ids"].shape[0]
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            _, counterfactual_outputs = intervenable(
                {"input_ids": batch["input_ids"]},
                [
                    {"input_ids": batch["source_input_ids"]},
                ] * n_blocks,
                {
                    "sources->base": tuple(
                       [[[[pos]] * batch_size] * 2 for pos in poss]
                    )
                },
                subspaces=[
                    [[0]] * batch_size,
                ] * n_blocks,
            )
            eval_labels += [batch["labels"].squeeze()]
            eval_preds += [counterfactual_outputs.logits[:,-1,:].argmax(dim=-1)]
    eval_labels = torch.cat(eval_labels)
    eval_preds = torch.cat(eval_preds)
    acc = compute_metrics(eval_preds, eval_labels)["accuracy"]
    return acc

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
    subspace_dimension = 1
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

    for layer in layers:
        intervenable = config_das(model, layer, device, subspace_dimension=subspace_dimension)
        for pos in poss:
            # create optimizer
            optimizer_params = []
            for k, v in intervenable.interventions.items():
                optimizer_params += [{"params": v.rotate_layer.parameters()}]
            optimizer = torch.optim.Adam(optimizer_params, lr=0.001)
            # train the model
            DAS_training(intervenable, train_dataset, optimizer, pos=pos, epochs=5, batch_size=batch_size)
            # test the model
            intervenable.disable_model_gradients()
            acc = das_test(intervenable, pos, test_dataset, batch_size)
            candidates[(layer, pos)] = acc
            # Take a safe snapshot of the rotate_layer state_dict so later in-place
            # changes to the intervenable don't mutate previously stored weights.
            sd = intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#0"].rotate_layer.state_dict()
            weights[(layer, pos)] = {k: v.clone().detach().cpu() for k, v in sd.items()}

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

def test_with_weights(model, layer, device, pos, test_dataset, batch_size=64, intervention_type = 'das', weight = None, subspace_dimension=1):
    ''' This function is used to test the model with pre-trained intervention weights.
    Input:
        model: the model to be used
        layer: the layer to be used for intervention
        device: the device to be used
        pos: the position of the intervention, int
        test_dataset: the testing dataset, contain input_ids, source_input_ids, labels
        weight: the pre-trained weight (state_dict) of the intervention
        batch_size: the batch size to be used, int
    Output:
        acc: the accuracy of the model, float
    This function will create an intervenable model with pre-trained weights and test its accuracy.'''
    
    # Create intervenable model with the pre-trained weight
    if intervention_type == 'das':
        intervenable = config_das(model, layer, device, weight, subspace_dimension=subspace_dimension)
    elif intervention_type == 'vanilla':
        intervenable = config_vanilla(model, layer, device)
    else:
        raise ValueError("intervention_type must be 'das' or 'vanilla'")
    
    # Test the model using the existing das_test function
    acc = das_test(intervenable, pos, test_dataset, batch_size, intervention_type)
    
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DAS training or testing with selectable causal model")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Run training to find candidate alignments")
    mode_group.add_argument("--test", action="store_true", help="Run testing using precomputed weights and candidates")
    parser.add_argument("--causal-model", choices=["1", "2"], default="1", help="Which causal model to use: 1 (default) or 2")
    parser.add_argument("--intervention-type", type=str, default='das', help="Type of intervention to use (e.g., 'das')")
    parser.add_argument("--weights-path", type=str, default=None, help="Path to das weights (.pt) for test mode")
    parser.add_argument("--candidates-path", type=str, default=None, help="Path to candidates JSON for test mode")
    parser.add_argument("--data-size", type=int, default=1024, help="Number of examples to generate per dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataset creation and evaluation")
    parser.add_argument("--subspace-dimension", type=int, default=1, help="Dimension of the subspace for intervention")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cpu' or 'cuda'). If not set, auto-detects.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (e.g., 0, 1). Overridden by --device if both are set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--layer", type=int, default=None, help="Layer index to use (default: all layers)")
    parser.add_argument("--pos-num", type=int, default=None, help="Position index for intervention (default: range 76-81)")
    parser.add_argument("--op-list", type=str, nargs="*", default=None,
                        help="Override op_list for interventions (e.g. --op-list op5 or --op-list op4a op5a). If not set, uses default per causal model.")
    args = parser.parse_args()

    # Set random seed early for reproducibility
    set_random_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # load data
    vocab, texts, labels = util_data.get_vocab_and_data()

    # construct causal model based on selection
    if args.causal_model == "1":
        or_causal_model = util_data.build_causal_model(vocab)
    elif args.causal_model == "2":        
        or_causal_model = util_data.build_causal_model2(vocab)
    else:
        raise RuntimeError(f"Unsupported causal model selection: {args.causal_model}")
    
    # load trained model
    model, tokenizer = util_model.load_model()
    if args.device is not None:
        device = args.device
    elif args.gpu is not None:
        device = f"cuda:{args.gpu}"
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")

    # create dataset params
    data_size = args.data_size
    batch_size = args.batch_size

    weights = {}

    # Train/test common params
    poss = [args.pos_num] if args.pos_num is not None else range(76, 82)
    layers = [args.layer] if args.layer is not None else range(model.config.n_layer)

    if args.causal_model == "1":
        default_op_list = ["op5"]
        data_generator = "all"
        out_op = "op5"
    elif args.causal_model == "2":
        default_op_list = ["op4a", "op5a"]
        data_generator = "all2"
        out_op = "op6a"
    op_list = args.op_list if args.op_list is not None and len(args.op_list) > 0 else default_op_list
    print(f"Using op_list: {op_list}")
    causal_model_tag = f"or_model_{args.causal_model}"
    intervention_type = args.intervention_type
    subspace_dimension=args.subspace_dimension

    if args.train:
        print("Starting training (finding candidate alignments)")
        # Load existing results so we append instead of overwriting
        candidates_path = f"training_results/candidates_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.json"
        weights_path = f"training_results/das_weights_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.pt"
        candidates_total = {}
        das_weights = {}
        if os.path.exists(candidates_path):
            with open(candidates_path, "r") as f:
                candidates_total = json.load(f)
            print(f"Loaded existing candidates from {candidates_path}")
        if os.path.exists(weights_path):
            das_weights = torch.load(weights_path)
            print(f"Loaded existing weights from {weights_path}")

        for intervention in op_list:
            if intervention not in candidates_total:
                candidates_total[intervention] = {}
            if intervention not in das_weights:
                das_weights[intervention] = {}
            dataset = util_data.make_counterfactual_dataset(
                data_generator,
                intervention,
                vocab,
                texts,
                labels,
                out_op,
                or_causal_model,
                model,
                tokenizer,
                data_size,
                device,
                batch_size=batch_size,
            )
            print(f"Dataset created for {intervention}")
            print(f"Finding candidates for {intervention}")
            candidate, weight = find_candidate_alignments(
                model,
                dataset,
                poss,
                layers,
                batch_size,
                device,
                n_candidates=72,
                subspace_dimension=args.subspace_dimension
            )
            candidates_total[intervention].update(candidate)
            das_weights[intervention].update(weight)

        # persist results (append: existing results were loaded above and merged with new run)
        os.makedirs("training_results", exist_ok=True)
        intervention_type = 'das'
        with open(candidates_path, "w") as f:
            json.dump(candidates_total, f, indent=4)
        print(f"Candidate alignments saved to {candidates_path}")

        torch.save(das_weights, weights_path)
        print(f"DAS weights saved to {weights_path}")

    elif args.test:
        print("Starting testing using provided weights and candidates")
        if args.intervention_type == 'das':
            if not args.weights_path:
                args.weights_path = f"training_results/das_weights_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.pt"
            if not args.candidates_path:
                args.candidates_path = f"training_results/candidates_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.json"

            # load provided artifacts
            das_weights = load_weight(args.weights_path)
            with open(args.candidates_path, "r") as f:
                candidates_total = json.load(f)

        test_results = {}

        if args.causal_model == "1":
            data_generator = "all"

        elif args.causal_model == "2":
            data_generator = "exhaustive2"

        for intervention in op_list:
            types = util_data.corresponding_intervention(intervention)
            test_results[intervention] = {}
            for source_code, base_code in types:
                print(f"Creating dataset for {intervention}, source: {source_code}, base: {base_code}")
                dataset = util_data.make_counterfactual_dataset(
                    data_generator,
                    intervention,
                    vocab,
                    texts,
                    labels,
                    out_op,
                    or_causal_model,
                    model,
                    tokenizer,
                    data_size,
                    device,
                    batch_size=batch_size,
                    source_code=source_code,
                    base_code=base_code,
                )
                print(f"Dataset created for {intervention}, source: {source_code}, base: {base_code}\r")

                # get the candidates for this intervention
                candidates = candidates_total.get(intervention, {})
                weights_for_intervention = das_weights.get(intervention, {}) if isinstance(das_weights, dict) else das_weights

                # Optionally restrict to selected layer/pos_num
                candidate_keys = list(candidates.keys())
                if args.layer is not None and args.pos_num is not None:
                    selected_key = f"L{args.layer}_P{args.pos_num}"
                    if selected_key in candidate_keys:
                        candidate_keys = [selected_key]
                    else:
                        print(f"Warning: candidate {selected_key} not in candidates; evaluating all")

                results = {}
                for candidate in candidate_keys:
                    layer, pos = extract_layer_pos(candidate)
                    weight = weights_for_intervention.get(candidate)
                    if weight is None:
                        print(f"Warning: weight for candidate {candidate} not found; skipping")
                        continue
                    acc = test_with_weights(
                        model,
                        layer,
                        device,
                        pos,
                        dataset,
                        batch_size=batch_size,
                        intervention_type=intervention_type,
                        weight=weight,
                        subspace_dimension=args.subspace_dimension
                    )
                    print(f"Source: {source_code}, Base: {base_code}, Candidate: {candidate}, Accuracy: {acc:.4f} \r")
                    results[candidate] = acc

                test_results[intervention]["s" + source_code + "_b" + base_code] = results

                # Save partial results after each evaluation
                os.makedirs("test_results", exist_ok=True)
                with open(f"test_results/test_results_partial_{intervention_type}_{causal_model_tag}.json", "w") as f:
                    json.dump(test_results, f, indent=4)
                print(f"Partial test results saved after {intervention}")

        # Save final test results
        with open(f"test_results/test_results_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.json", "w") as f:
            json.dump(test_results, f, indent=4)
        print(f"Final test results saved to test_results/test_results_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.json")

    # Release the GPU memory
    model.cpu()
    torch.cuda.empty_cache()
    print("GPU memory cleared")