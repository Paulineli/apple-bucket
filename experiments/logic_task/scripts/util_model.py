from transformers import GPT2LMHeadModel, AutoModelForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
import os
import random
import pandas as pd
import os


from paths import ARTIFACTS, REPO_ROOT


def load_model(model_path_ft = None):
    '''Load the model from the specified path or from Hugging Face if not found.'''
    _default_ft = ARTIFACTS / "models" / "fine_tuned_gpt2_or"
    _archived_pretrained = REPO_ROOT / "archived" / "pretrained_model" / "pretrained_tuned_gpt2"

    if model_path_ft is None:
        model_path_ft = str(_default_ft)
    if model_path_ft and os.path.exists(model_path_ft):
        print(f"Loading existing model from {model_path_ft}")
        model = GPT2LMHeadModel.from_pretrained(model_path_ft)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path_ft)
        print("Model loaded successfully!")
    elif _archived_pretrained.is_dir():
        ap = str(_archived_pretrained)
        print(f"Loading existing model from {ap}")
        model = GPT2LMHeadModel.from_pretrained(ap)
        tokenizer = GPT2Tokenizer.from_pretrained(ap)
        print("Model loaded successfully!")
    else:
        print(f"Did not find existing model from {model_path_ft}")
        print("Loading a new model from hugging face")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print("Model loaded successfully!")
        # Save next to this experiment (authors may move under archived/ if unused)
        _fallback_dir = str(ARTIFACTS / "pretrained_model_fallback")
        os.makedirs(_fallback_dir, exist_ok=True)
        model.save_pretrained(_fallback_dir)
        tokenizer.save_pretrained(_fallback_dir)
    return model, tokenizer