import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM
from matplotlib import pyplot as plt
from scipy.stats import zscore
import gc

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
    print(name)
    model_layer_names = [name for name, param in model.named_parameters()]
    self_attn_layer_indicies = [index for index, name in enumerate(model_layer_names) if 'self_attn' in name]
    outliers = {
        name: np.mean(np.abs(zscore(layer)) > 3) for name, layer in 
        [(name, param.data.flatten().numpy()) for name, param in model.named_parameters()]
    }
    prop_outliers[name] = outliers
    del model
    gc.collect()
    print("Complete")


# In[4]:


prop_outliers
