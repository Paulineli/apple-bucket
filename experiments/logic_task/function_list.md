# Serial Interchange Intervention and Discussion

## List of Functions

### In `util_data.py`

* `get_vocab_and_data()`: Load tvocabulary and data from the specified files. Returns lists `vocab`, `texts`, `labels`.
* `build_causal_model(vocab, model_type = 'or_model')`: Build a causal model based on the specified model type. The function returns a CausalModel object. 
* `build_causal_mode2(vocab, model_type = 'or_model')`: Same as above. Build the alternaive model.  
* `format_input(raw_input, context_texts, context_labels)`: Format the input for the model in the form of "context1=label1\n context2=label2\n ...input=t0,t1,t2,t3,t4,t5=". This is good for in-context learning. 
* `data_filter(causal_model, model, tokenizer, dataset, device, batch_size = 16)`:This function filters the dataset based on the model's predictions. It checks if the model's predictions for both base and source inputs match the labels in the dataset. If they do, the data point is kept; otherwise, it is discarded. The function returns a new filtered dataset.
* `make_counterfactual_dataset(...)`: This function takes in `dataset_type`, `interv`, `vocab`, `texts`, `labels`, `output_op`, `equality_model`, `model`, `tokenizer`, `data_size`, `device`, `batch_size = 32`. It generates a counterfactual tokenized dataset. The output dataset is already filtered and tokenized and is in the ICL format.
* `influenced_ops(source_code: str, base_code: str)`: Generates ops that will flip the output given the source and the base input types. 
* `corresponding_intervention`: Generated source and base input types corresponding to an operation. 


**Below are four contefactual dataset types.** 【TO CHECK】

They take in the high-level causal model, vocabulary, the name of the intervened variable, and the sample size, and generate dataset with source inputs and (intervened) base inputs. 

* `make_counterfactual_dataset_ft(causal_model, vocab, intervention:str, samplesize:int)`
* `make_counterfactual_dataset_fixed(causal_model, vocab, intervention:str, samplesize:int)`
* `make_counterfactual_dataset_average(causal_model, vocab, intervention:str, samplesize:int)`
* `make_counterfactual_dataset_all(causal_model, vocab, interventions:list, samplesize:int)`
* `make_counterfactual_dataset_all2(causal_model, vocab, interventions:list, samplesize:int)`

### In `util_model.py`
* `def load_model(model_path_ft = "./ft_model/fine_tuned_gpt2_or")`: Load the transformer model (GPT-2) from the specified path or from Hugging Face if not found.

### In `step1_das.py`
* `compute_metrics(eval_preds, eval_labels)`: This function is used to compute the accuracy of the predictions. It returns the accuracy. 
* `compute_loss(outputs, labels)`: This function returns the entropy loss. 
* `batched_random_sampler(data, batch_size)`: samples data batches of a fixed sample size. 
* `config_das(model, layer, device)`: The function is used to set up the configuration for DAS intervention and returns an IntervenableModel (the model with the intervention). 
* `config_das_parallel(model, layers, device, weights=None)`: This function sets up the configuration for parallel interchange intervention.
* `DAS_training(intervenable, train_dataset, optimizer, pos, epochs = 10, batch_size = 64, gradient_accumulation_steps = 1)`: function that trains the model with DAS intervantion. 
* `das_test(intervenable, pos, test_dataset, batch_size = 64)`: Tests the model with interchange intervention. Returns interchange intervention accuracy (float). 
* `save_weight(weights, name: str, path: str)`: Saves the model state dict and the state dict of interventions. 
* `load_weight(name)`: Loads the model state dict and returns the weights.
* `find_candidate_alignments(model, dataset, poss, layers, batch_size, device, n_candidates = 10)`: Returns candidates and weights of the candidates. 
* `extract_layer_pos(string)`: Takes a position indicator of the form "L5_P78". Returns `layer_num`, `pos_num`. 
* `select_candidates()`: TBD