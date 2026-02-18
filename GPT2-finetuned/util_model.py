from transformers import GPT2LMHeadModel, AutoModelForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
import os
import random
import pandas as pd
import os


def load_model(model_path_ft = "./fine_tuned_gpt2_or"):
    '''Load the model from the specified path or from Hugging Face if not found.'''
    if model_path_ft is None:
        model_path_ft = "./fine_tuned_gpt2_or"
    if os.path.exists(model_path_ft):
        print(f"Loading existing model from {model_path_ft}")
        model = GPT2LMHeadModel.from_pretrained(model_path_ft)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path_ft)
        print("Model loaded successfully!")
    elif os.path.exists("./pretrained_model/pretrained_tuned_gpt2"):
        print(f"Loading existing model from ./pretrained_model/pretrained_tuned_gpt2")
        model = GPT2LMHeadModel.from_pretrained("./pretrained_model/pretrained_tuned_gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("./pretrained_model/pretrained_tuned_gpt2")
        print("Model loaded successfully!")
    else:
        print(f"Did not find existing model from {model_path_ft}")
        print("Loading a new model from hugging face")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print("Model loaded successfully!")
        # Save the model
        os.mkdir("./pretrained_model/")
        model.save_pretrained("./pretrained_model/pretrained_tuned_gpt2")
        tokenizer.save_pretrained("./pretrained_model/pretrained_tuned_gpt2")
    return model, tokenizer