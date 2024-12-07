import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM
from matplotlib import pyplot as plt
from scipy.stats import zscore
import gc
import os
import torch

model_names = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen1.5-14B",
]
model_sizes = [
    "0.5B",
    "3B",
    "7B",
    "14B",
]

for name, filename in zip(model_names, model_sizes):
    os.makedirs(filename, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(name)
    for layer_name, layer in model.named_parameters():
        save_loc = f'{filename}/{layer_name}.pt'
        torch.save(layer, save_loc)
        print("Saved layer: ", save_loc)
    del model
    gc.collect()
