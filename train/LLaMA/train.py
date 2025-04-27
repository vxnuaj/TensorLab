import torch
import torch.optim as opt
import torch.nn as nn
import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from model import LLaMA
from train_utils import Trainer, Config, get_scheduler
from dataloader import get_data, get_dataloader

root_path = 'main/configs'
data_root_path = 'data/train'

# GET CONFIGS FROM configs/{}.json

loss_config = Config.get_config(
    root_path = root_path,
    config_type = 'loss'
)

lr_config = Config.get_config(
    root_path = root_path,
    config_type = 'lr'
)

opt_config = Config.get_config(
    root_path = root_path,
    config_type = 'opt'
)

train_config = Config.get_config(
    root_path = root_path,
    config_type = 'train'
)

wandb_config = Config.get_config(
    root_path = root_path,
    config_type = 'run'
)

dataloader_config = Config.get_config(
    root_path = root_path,
    config_type = 'dataloader'
)

model_config = Config.get_config(
    root_path = root_path,
    config_type = 'model'
)

# DATA & DATALOADER

X, y = get_data(dataloader_config['data_root_path'])
dataloader = get_dataloader(X, y, **dataloader_config)
model = torch.compile(LLaMA(**model_config))
trainer_config = Config(**train_config)
optimizer = opt.AdamW(**opt_config)
scheduler = get_scheduler(optimizer, **lr_config)
criterion = nn.CrossEntropy(**loss_config)

# TRAINER & BEGIN TRAINING

trainer = Trainer(
    model = model,
    criterion = criterion,
    dataloader = dataloader,
    scheduler = scheduler,
    config = trainer_config
)

trainer.train()