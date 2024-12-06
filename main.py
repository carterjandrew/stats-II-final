#!/usr/bin/env python
# coding: utf-8

# # Statistical analysis of quen llm model outliers
# **By:** *Carter Andrew*  
# `11-23-2024`

# # Intro
# Here we have some introduction to the goal of this notebook

# ## Inspiration
# TODO: There is some paper talking about how larger language models have a larger number of outliers with extreme values in their weights. I want to do my own investigation to confirm if this is true

# # Setup
# Here we import libraries, set up notebook behaviour, download models, ect...

# In[1]:


import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib import pyplot as plt
from scipy.stats import zscore


# ## Loading models
# We can load our models for each size from `HuggingFace` by using their `transformers` library

# In[5]:


model_names = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen1.5-14B",
    #"Qwen/Qwen2.5-32B"
]


# # Extraing weight averages
# We can do the following to get out our numbers to work with
# 1. For each model:
# 2. Aggregate layers
# 3. Compute the number of outliers per layer
# 4. Add our values into a database

# In[6]:

import gc

prop_outliers = {}
for name in model_names:
    model = AutoModelForCausalLM.from_pretrained(name)
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

import json
with open("output.json", 'w') as fp:
    json.dump(prop_outliers, fp)
