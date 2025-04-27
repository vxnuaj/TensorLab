import torch
import torch.nn as nn
import torch.distributed as dist

import math
import time
import warnings
import json
import os
import gc
import sys
import wandb

from torch.distributed import ReduceOp
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from dataloader import get_dataloader

from dataclasses import dataclass, field
from tqdm import tqdm

@dataclass 
class Config:
    vocab_size:int
    batch_size:int
    context_length:int
    epochs:int
    checkpoint_steps:int
    save_checkpoint_path:str
    val_steps:int
    mixed_precision:bool
    max_grad_norm:float
    track_grad_norm:bool
    parallel_type:str
    val_batch_size:int
    val_num_workers:int
    val_shuffle:bool
    val_pin_memory:bool
    val_mixed_precision:bool
    X_val_path:str
    y_val_path:str
    wandb_:bool
    run_config:dict
    _compile:bool
    _compile_warmup:int
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        fields = Config.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs
        
    @staticmethod
    def get_config(root_path:str, config_type:str):
        assert config_type in ['loss', 'lr', 'opt', 'train', 'run', 'model'], ValueError("config_type must be in 'loss', 'lr', 'opt' or 'train'")
        if config_type == 'loss':
            with open(os.path.join(root_path, 'loss_config.json'), 'r') as f:
                return json.load(f)
        elif config_type == 'lr':
            with open(os.path.join(root_path, 'lr_config.json'), 'r') as f:
                return json.load(f)           
        elif config_type == 'opt':
            with open(os.path.join(root_path, 'opt_config.json'), 'r') as f:
                return json.load(f)
        elif config_type == 'train':
            with open(os.path.join(root_path, 'train_config.json'), 'r') as f:
                return json.load(f)           
        elif config_type == 'run':
            with open(os.path.join(root_path, 'run_config.json'), 'r') as f:
                return json.load(f)           
        elif config_type == 'model':
            with open(os.path.join(root_path, 'model_config.json'), 'r') as f:
                return json.load(f)                  
        elif config_type == 'dataloader':
            with open(os.path.join(root_path, 'dataloader_config.json', 'r')) as f:
                return json.load(f)

class Trainer:
    def __init__(
        self,
        model,
        criterion,
        dataloader,
        optimizer,
        scheduler,
        config: Config
        ):
       
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.config = config
     
        self.vocab_size = config.vocab_size 
        self.batch_size = config.batch_size 
        self.context_length = config.context_length 
        self.epochs = config.epochs
        self.checkpoint_steps = config.checkpoint_steps
        self.save_checkpoint_path = config.save_checkpoint_path
        self.val_steps = config.val_steps
        self.mixed_precision = config.mixed_precision
        self.max_grad_norm = config.max_grad_norm
        self.track_grad_norm = config.track_grad_norm
        self.parallel_type = config.parallel_type
        self.val_batch_size = config.val_batch_size
        self.val_num_workers = config.val_num_workers
        self.val_shuffle = config.val_shuffle
        self.val_pin_memory = config.val_pin_memory
        self.val_mixed_precision = config.val_mixed_precision
        self.X_val_path = config.X_val_path
        self.y_val_path = config.y_val_path
        self.wandb_ = config.wandb_ # bool, if True, then wandb tracking is enabled
        self.run_config = config.run_config
        self._compile = config._compile
        self._compile_warmup_steps = config._compile_wamrup_steps
       
        self.scaler = GradScaler() if self.mixed_precision else None
        self._check_dataloader_sampler()
        self._setup_parallel()
        self.device = self._get_device()
        self.model = self._get_model(model)
        
    def train(self):
    
        self._compile_wamrup() 
        self._init_wandb() 
        self._check_device_warn()
        global_steps = 0 
        
        for epoch in range(self.epochs):
            progress_bar = tqdm(enumerate(self.dataloader), desc="Training", total=len(self.dataloader), 
                                disable = (dist.get_rank()!=0 and self.parallel_type in ['fsdp', 'ddp']))
            for i, (X, y) in progress_bar:
                X, y = X.to(self.device), y.to(self.device)
                if self.mixed_precision:
                    with autocast(device_type = 'cuda', dtype = torch.float16):
                        if dist.get_rank() == 0:
                            start_time = time.time()
                        logits = self.model(X)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                        pplx = torch.exp(loss)
                    loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss, pplx) 
                    self.scaler.scale(loss).backward() 
                    if dist.get_rank() == 0:
                        end_time = time.time()
                    self.scaler.unscale_(self.optimizer)
                    if self.track_grad_norm:
                        grad_norm_dict = self._clip_grad_norm()
                    self._get_grad_norm()
                    self.scaler.step(self.optimizer) 
                    self.scaler.update()
                    self.scheduler.step()
                else:
                    logits = self.model(X) 
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    pplx = torch.exp(loss)
                    loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss, pplx) 
                    loss.backward()
                    self._clip_grad_norm()
                    if self.track_grad_norm:
                        grad_norm_dict = self._get_grad_norm()
                    self.optimizer.step()
                    self.scheduler.step()
                
                global_steps += 1 
               
                if dist.get_rank() == 0: 
                    progress_bar.set_description(desc = f"Epoch: {epoch + 1} | Local step: {i} | \
                                                 Global step: {global_steps} | Loss: {loss_avg.item()} | pplx: {pplx_avg.item()} | Time: {end_time - start_time}") 
               
                if self.wandb_: 
                    wandb_dict = {
                        "loss": loss_avg.item(),
                        "perplexity": pplx_avg.item(),
                    }
                    if self.track_grad_norm:
                        wandb_dict.update(grad_norm_dict)
                    wandb.log(wandb_dict)  
                    
                if global_steps % self.checkpoint_steps == 0:
                    dist.barrier() 
                    self._clr_mem(gc_ = True, cuda_clr_cache = True, X = X, y = y, logits = logits)
                    
                    model_state_dict = self._get_model_state_dict()
                    optim_state_dict = self._get_optim_state_dict()  
                    scheduler_state_dict = self._get_scheduler_state_dict()
                  
                    self._save_checkpoint(
                        path = self.save_checkpoint_path,
                        model_state_dict = model_state_dict,
                        optim_state_dict = optim_state_dict,
                        scheduler_state_dict = scheduler_state_dict,
                        epoch = epoch,
                        global_steps = global_steps
                        ) 
                   
                if self.val_steps and global_steps % self.val_steps == 0:
                    self.model.eval() 
                    if global_steps % self.checkpoint_steps != 0:
                        dist.barrier()
                        self._clr_mem(gc_ = True, cuda_clr_cache = True, X = X, y = y, logits = logits)
                    
                    val_dataloader = self._get_val_dataloader(self.X_val_path, self.y_val_path)
                   
                    val_progress_bar = tqdm(enumerate(val_dataloader), desc = "Evaluating", total = len(val_dataloader),
                                        disable = (dist.get_rank()!=0 and self.parallel_type in ['fsdp', 'ddp']))
                   
                    val_steps = 0 
                    loss_accum = 0
                    pplx_accum = 0 
                    
                    for i, (X_val, y_val) in val_progress_bar:
                        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                        if self.val_mixed_precision:
                            with autocast(device_type = 'cuda', dtype = torch.float16):
                                logits = self.model(X_val)
                                loss = self.criterion(logits.view(-1, logits.size(-1)), y_val.view(-1))
                                pplx = torch.exp(loss)
                            loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss, pplx)
                        else:
                            logits = self.model(X_val)
                            loss = self.criterion(logits.view(-1, logits.size(-1)), y_val.view(-1))
                            pplx = torch.exp(loss)
                            loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss, pplx)

                        loss_accum += loss_avg
                        pplx_accum += pplx_avg
                        val_steps += 1

                    val_loss = loss_accum / val_steps
                    val_pplx = pplx_accum / val_steps

                    print(f"Validation Loss: {val_loss.item()} | Validation PPLX: {val_pplx.item()}")
                   
                    if self.wandb_:
                        wandb.log({
                            "val loss": val_loss.item(),
                            "val perplexity": val_pplx.item()
                            }
                        )
                   
                    self._clr_mem(gc_ = True, cuda_clr_cache = True, X_val = X_val, y_val = y_val, logits = logits) 
                    
                    self.model.train()

        self._cleanup()
   
    def _clip_grad_norm(self):
        if self.max_grad_norm: 
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
     
    def _get_grad_norm(self):
        if self.track_grad_norm:
            assert self.wandb_, ValueError('wandb_ must be set to True if you want to track the gradient norm') 
            grad_norm_dict = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    clean_name = name.replace('_fsdp_wrapped_module.', '').replace('._flat_param', '')
                    param_norm = p.grad.norm(2)
                    grad_norm_dict[clean_name] = param_norm
            return grad_norm_dict 
            
    def _check_device_warn(self):
        if self.device.type == 'cpu':
            warnings.warn('Training on CPU')
            cont = input('Continue [y/n]?')
            if cont.lower() == 'n':
                sys.exit(0)
                
    def _setup_parallel(self):
        dist.init_process_group(backend = 'nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
    def _cleanup(self):
        dist.destroy_process_group()
        
    def _get_device(self):
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}") if torch.cuda.is_available() else 'cpu'
        return device
        
    def _check_dataloader_sampler(self):
        
        if self.parallel_type in ['fsdp', 'ddp']:
            if not isinstance(self.dataloader.sampler, DistributedSampler):
                raise ValueError('if parallel_type is fsdp or ddp, then the sampler of the dataloader must DistributedSampler')
            
    def _get_model(self, model):
        local_rank = int(os.environ['LOCAL_RANK'])
        if self.parallel_type in ['ddp']:
            return DDP(model.to(self.device), device_ids = [local_rank])
        elif self.parallel_type in ['fsdp']:
            return FSDP(model.to(self.device), device_id = [local_rank])
        else:
            return model.to(self.device)            
            
    def _get_avg_rank_loss_pplx(self, loss, pplx):
        if self.parallel_type == 'ddp':
            loss_tensor = loss.detach().clone()
            dist.all_reduce(loss_tensor, op = ReduceOp.SUM)
            dist.all_reduce(pplx, op = ReduceOp.SUM)
            loss_avg = loss_tensor / dist.get_world_size() 
            pplx_avg = pplx / dist.get_world_size()
            return loss_avg, pplx_avg
        elif self.parallel_type == 'fsdp':
            dist.all_reduce(loss, op=ReduceOp.SUM)
            dist.all_reduce(pplx, op=ReduceOp.SUM)
            loss_avg = loss / dist.get_world_size()
            pplx_avg = pplx / dist.get_world_size()
            return loss_avg, pplx_avg
        
    def _get_model_state_dict(self):
        if self.parallel_type == 'ddp':
            return self.model.module.state_dict()
        elif self.parallel_type == 'fsdp':
            return FSDP.full_state_dict(self.model, rank0_only=True)["model"]
        else:
            if dist.get_rank() == 0:
                return self.model.state_dict()
        
    def _get_optim_state_dict(self):
        if self.parallel_type == 'fsdp':
            return FSDP.optim_state_dict(self.model, self.optimizer, rank0_only=True)
        else:
            return self.optimizer.state_dict()

    def _get_scheduler_state_dict(self):
        if self.parallel_type in ['fsdp', 'ddp'] and dist.get_rank() == 0:
            return self.scheduler.state_dict()
    
    def _save_checkpoint(self, root_path, model_state_dict, optim_state_dict, scheduler_state_dict, epoch, global_steps):
        if dist.get_rank() == 0:
            
            root_path = os.path.join(root_path, self.run_config['name']) 
            os.makedirs(root_path, exist_ok = True) 
            
            torch.save(
                {'epoch': epoch, 'global_steps': global_steps, 'model': model_state_dict,
                 'optim': optim_state_dict, 'scheduler_state_dict': scheduler_state_dict},
                f = os.path.join(root_path, f'checkpoint_{epoch}_global_steps_{global_steps}.pt')
            ) 
            
            print(f"Saved checkpoint at epoch {epoch} and global steps {global_steps}.")
            
    def _clr_mem(self, gc_ = False, cuda_clr_cache = True, *args, **kwargs):
        if gc_:
            gc.collect() 
        if cuda_clr_cache:
            torch.cuda.empty_cache()
        for i in args:
            del i  
        for key in kwargs:
            del kwargs[key] 
            
    def _get_val_dataloader(self, X_val_path, y_val_path):
       
        X_val = torch.load(X_val_path)
        y_val = torch.load(y_val_path)
        
        val_dataloader = get_dataloader(
            X = X_val,
            y = y_val,
            batch_size = self.val_batch_size,
            num_workers = self.val_num_workers,
            shuffle = self.val_shuffle,
            pin_memory = self.val_pin_memory,
            parallelism_type = self.parallel_type,
            rank = dist.get_rank()
        ) 
        
        return val_dataloader
    
    def _init_wandb(self):
        if self.wandb_:
            assert isinstance(self.run_config, dict), ValueError('run_config must be type dict')
            wandb.init(
                **self.run_config
            ) 

    def _compile_wamrup(self):
        if self._compile:
            print('Running compile wamrup')
            x = torch.randint(low = 0, high = self.vocab_size, size = (self.batch_size, self.context_length))
            for _ in tqdm(range(self._compile_warmup_steps), desc = 'Compile warmup.', total = self._compile_warmup_steps):
                self.model(x)
            self._clr_mem(gc_= True, cuda_clr_cache=True, x = x) 
            print(f'Finished running compile wamrup, beginning training...') 
  

def get_scheduler(optimizer, warmup_steps, constant_steps, decay_steps, max_lr, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            lr = min_lr + (max_lr - min_lr) * (step / warmup_steps)
        elif step < warmup_steps + constant_steps:
            lr = max_lr
        else:
            decay_step = step - (warmup_steps + constant_steps)
            cosine_progress = min(1.0, decay_step / decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(cosine_progress * math.pi))
            lr = min_lr + (max_lr - min_lr) * cosine_decay
        return lr / optimizer.param_groups[0]['initial_lr']
    return LambdaLR(optimizer, lr_lambda)