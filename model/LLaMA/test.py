import torch
import numpy as np
import os 
import time

from model import LLaMA

batch_size = 16
seq_len = 512
context_len = 512
d_model = 256
n_heads = 8
n_blocks = 4
vocab_size = 10000
pos_emb_dropout_p = 0. 
pos_emb_type = "pe"
learned = False
ntk_rope_scaling = False
dyn_scaling = False
attn_type = "mva" # mhsa, mqa, gqa, mva, tksva (slow and memory intensive, as I haven't written a kernel for it yet)
top_k_sparsev = 128
p_threshold = 0.1
p_threshold_steps_fraction = 0.6
n_groups = 4
supress_warnings = True
_inference = True

model = LLaMA(
    context_len = context_len,
    d_model=d_model,
    n_heads = n_heads,
    n_blocks = n_blocks,
    vocab_size = vocab_size,
    pos_emb_dropout_p = pos_emb_dropout_p,
    pos_emb_type = pos_emb_type,
    learned = learned,
    ntk_rope_scaling = ntk_rope_scaling,
    dyn_scaling = dyn_scaling,
    attn_type = attn_type,
    n_groups = n_groups,
    top_k_sparsev = top_k_sparsev,
    p_threshold = p_threshold,
    p_threshold_steps_fraction = p_threshold_steps_fraction,
    supress_warnings = supress_warnings
    )

x = torch.randint(low = 0, high = vocab_size, size = (batch_size, seq_len))

print('compiling')

model = torch.compile(model)
    
# Forward Pass
x = torch.randint(low = 0, high = vocab_size, size = (batch_size, seq_len))
print(f'FORWARD ----------')
start = time.time()
print(model(x).shape, '\n')
print(time.time() - start)

# Inference
x1 = torch.randint(low = 0, high = vocab_size, size = (batch_size, seq_len))
x2 = torch.randint(low = 0, high = vocab_size, size = (batch_size, 1))

print(f'INFERENCE ----------')
start = time.time()