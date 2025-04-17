import torch
import torch.nn as nn
import torch.distributed as dist
import math
import torch.nn.functional as F

import sys
import wandb
import json
import time
import warnings
import functools
import os
import gc

from pathlib import Path
from typing import List
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import RandomSampler, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.amp import autocast, GradScaler
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, project_root)

from model.SLM.blocks import TransformerBlock

def train(
    model, optim, scheduler, criterion, dataloader, device,
    val_steps: int, val_size: int,
    autocast_dtype: torch.dtype=None, mixed_precision=False,
    total_epochs=None, clip_grad_norm=False, max_grad_norm: int=5,
    track_grad_norm=False, rank=None, parallelism_type: str=None,
    checkpoint_epoch: int=0, checkpoint_batch_idx: int=0,
    global_steps: int=0, val_data_path: str=None,
    save_checkpoint_steps: int=None, save_checkpoint_path: str=None,
    _model_key='model_state_dict', _optim_key='optim_state_dict',
    _scheduler_key='scheduler_state_dict', wandb_: bool=False,
    pad_token_id=None, **kwargs
):
    """Train a model with support for distributed training, mixed precision, and checkpointing.

    Args:
        model (nn.Module): Model to train.
        optim (torch.optim.Optimizer): Optimizer for training.
        scheduler: Learning rate scheduler.
        criterion (nn.Module): Loss function.
        dataloader (DataLoader): DataLoader for training data.
        device (torch.device): Device to run training on.
        val_steps (int): Number of steps between validation.
        val_size (int): Size of validation data.
        autocast_dtype (torch.dtype, optional): Data type for mixed precision, or None.
        mixed_precision (bool): Whether to use mixed precision training. Defaults to False.
        total_epochs (int, optional): Total number of epochs to train.
        clip_grad_norm (bool): Whether to clip gradient norms. Defaults to False.
        max_grad_norm (int): Maximum gradient norm for clipping. Defaults to 5.
        track_grad_norm (bool): Whether to track gradient norms. Defaults to False.
        rank (int, optional): Process rank for distributed training.
        parallelism_type (str, optional): Type of parallelism ('fsdp', 'ddp', 'dp', or None).
        checkpoint_epoch (int): Starting epoch for resuming training. Defaults to 0.
        checkpoint_batch_idx (int): Starting batch index for resuming. Defaults to 0.
        global_steps (int): Starting global step count. Defaults to 0.
        val_data_path (str, optional): Path to validation data.
        save_checkpoint_steps (int, optional): Steps between checkpoints.
        save_checkpoint_path (str, optional): Directory to save checkpoints.
        _model_key (str): Key for model state in checkpoint. Defaults to 'model_state_dict'.
        _optim_key (str): Key for optimizer state in checkpoint. Defaults to 'optim_state_dict'.
        _scheduler_key (str): Key for scheduler state in checkpoint. Defaults to 'scheduler_state_dict'.
        wandb_ (bool): Whether to log to Weights & Biases. Defaults to False.
        pad_token_id (int, optional): ID of padding token, or None.
        **kwargs: Additional keyword arguments.

    Raises:
        ValueError: If DistributedSampler is not used for distributed training or max_grad_norm is not set when needed.
        AssertionError: If save_checkpoint_steps or save_checkpoint_path is not set.

    Returns:
        None
    """
    # Configuration checks
    _assert_config(save_checkpoint_steps, save_checkpoint_path, clip_grad_norm, max_grad_norm)
    if parallelism_type in ['fsdp', 'ddp'] and not isinstance(dataloader.sampler, DistributedSampler):
        raise ValueError("For distributed training, dataloader must use DistributedSampler")
    if (checkpoint_epoch or checkpoint_batch_idx) and isinstance(dataloader.sampler, RandomSampler):
        warnings.warn(f'Resuming at epoch {checkpoint_epoch}, step {checkpoint_batch_idx} with RandomSampler.')
        time.sleep(3)

    # Mixed-precision setup
    scaler = GradScaler() if mixed_precision else None
    dtype = autocast_dtype or torch.float16 if mixed_precision else None

    steps = global_steps
    model.train()

    if rank == 0 or parallelism_type not in ['fsdp', 'ddp']:
        print(f"Starting training at epoch {checkpoint_epoch}, step {checkpoint_batch_idx}")

    try:
        for epoch in range(checkpoint_epoch, total_epochs):
            if isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)

            with tqdm(total=len(dataloader), desc=f"Epoch {epoch}/{total_epochs}",
                      disable=(rank != 0 and parallelism_type in ['fsdp', 'ddp'])) as pbar:
                try:
                    for i, (X, y) in enumerate(dataloader):
                        if epoch == checkpoint_epoch and i < checkpoint_batch_idx:
                            continue

                        X, y = X.to(device), y.to(device)
                        optim.zero_grad()

                        # Forward
                        with autocast(device_type='cuda', dtype=dtype) if mixed_precision else torch.no_grad():
                            logits = model(X)
                        logits = logits.view(-1, logits.size(-1))
                        targets = y.view(-1)
                        per_token_loss = (criterion(logits, targets)
                                          if not mixed_precision else criterion(logits, targets))

                        # Loss reduction
                        loss = _reduce_loss(per_token_loss, targets, pad_token_id, parallelism_type)

                        # Backward and optimizer step
                        if mixed_precision:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optim)
                        else:
                            loss.backward()

                        if clip_grad_norm:
                            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        if track_grad_norm:
                            grad_norm_dict = get_grad_norm(model)

                        if mixed_precision:
                            scaler.step(optim)
                            scaler.update()
                        else:
                            optim.step()

                        # Cleanup and scheduler
                        _clear_cache(clr_cuda_cache=True, gc_=True, X=X, y=y)
                        scheduler.step()
                        steps += 1

                        # Logging
                        lr = scheduler.get_last_lr()[0]
                        pplx = math.exp(loss.item()) if loss.item() < 20 else None
                        pbar.set_description(
                            f"Epochs: {epoch+1}/{total_epochs} | Steps: {steps+1} |"
                            f" Loss: {loss.item():.4f} | PPLX: {pplx} | LR: {lr}"
                        )
                        pbar.update(1)

                        if wandb_ and (rank == 0 or parallelism_type not in ['fsdp', 'ddp']):
                            log_dict = {"Loss": loss.item(), "Perplexity": pplx, "Learning Rate": lr}
                            if track_grad_norm:
                                log_dict.update(grad_norm_dict)
                            wandb.log(log_dict)

                        # Validation
                        if steps % val_steps == 0:
                            model.eval()
                            with torch.no_grad():
                                X_val, y_val = _get_eval_data(val_data_path, eval_size=val_size)
                                X_val, y_val = X_val.to(device), y_val.to(device)
                                with autocast(device_type='cuda', dtype=dtype) if mixed_precision else torch.no_grad():
                                    logits_val = model(X_val)
                                logits_val = logits_val.view(-1, logits_val.size(-1))
                                targets_val = y_val.view(-1)
                                per_token_loss_val = F.cross_entropy(logits_val, targets_val, reduction='none')
                                loss_val = _reduce_loss(
                                    per_token_loss_val, targets_val, pad_token_id, parallelism_type)
                                pplx_val = math.exp(loss_val.item())

                            _clear_cache(clr_cuda_cache=True, gc_=True,
                                         X=X_val, y=y_val, targets=targets_val, mask=(targets_val != pad_token_id).float(), logits=logits_val)
                            if wandb_:
                                wandb.log({"Validation Loss": loss_val.item(), "Validation Perplexity": pplx_val})
                            print(f'VALIDATION | Loss: {loss_val.item()} | PPLX: {pplx_val}')
                            model.train()

                        dist.barrier()

                        # Checkpointing
                        if steps % save_checkpoint_steps == 0:
                            torch.cuda.synchronize()
                            dist.barrier()
                            _save_model_checkpoint(
                                model, optim, scheduler, loss, pplx, epoch, i,
                                rank, steps, save_checkpoint_path,
                                (_model_key, _optim_key, _scheduler_key), parallelism_type
                            )
                            print('Saved Checkpoint, now exiting')
                            _clear_cache(model_state_dict=model.module.state_dict() if parallelism_type in ['dp','ddp'] else model.state_dict(),
                                         optim_state_dict=optim.state_dict() if optim else None, gc_=True)
                            dist.barrier()

                except RuntimeError as e:
                    if 'out of memory in' in str(e):
                        print('OOM, clearing VRAM cache')
                        _clear_cache(clr_cuda_cache=True)
                        continue
                    raise

        print(f"Training completed over {total_epochs} epochs, {steps} steps")

    except KeyboardInterrupt:
        if is_main_process():
            print("KeyboardInterrupt - Stopping training safely.")

    finally:
        final_m = model.module.state_dict() if parallelism_type in ['dp','ddp'] else model.state_dict()
        final_o = optim.state_dict() if optim else None
        save_checkpoint(
            model_state_dict=final_m,
            optim_state_dict=final_o,
            scheduler=scheduler,
            loss=loss,
            pplx=pplx,
            epoch=epoch,
            batch_idx=i,
            rank=rank,
            global_steps=steps,
            save_checkpoint_path=save_checkpoint_path,
            _model_key=_model_key,
            _optim_key=_optim_key,
            _scheduler_key=_scheduler_key,
            parallelism_type=parallelism_type,
        )
        print("Saved Checkpoint, now exiting.")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

def _assert_config(save_steps, save_path, clip_grad_norm, max_grad_norm):
    """Validate training configuration parameters.

    Args:
        save_steps (int): Number of steps between checkpoints.
        save_path (str): Directory path to save checkpoints.
        clip_grad_norm (bool): Whether to clip gradient norms.
        max_grad_norm (int): Maximum gradient norm for clipping.

    Raises:
        AssertionError: If save_steps or save_path is not set.
        ValueError: If max_grad_norm is not set when clip_grad_norm is True.
    """
    assert save_steps and save_path, 'Both save_checkpoint_steps and save_checkpoint_path must be set.'
    if clip_grad_norm and not max_grad_norm:
        raise ValueError('max_grad_norm must be set if grad_norm is True')


def _reduce_loss(per_token_loss, targets, pad_token_id, parallelism_type):
    """Compute reduced loss with optional padding and distributed reduction.

    Args:
        per_token_loss (torch.Tensor): Per-token loss values.
        targets (torch.Tensor): Ground truth targets.
        pad_token_id (int, optional): ID of padding token, or None.
        parallelism_type (str): Type of parallelism ('fsdp', 'ddp', or None).

    Returns:
        torch.Tensor: Reduced loss value.
    """
    if pad_token_id is not None:
        mask = (targets != pad_token_id).float()
        local_loss = (per_token_loss * mask).sum()
        token_count = mask.sum()
    else:
        local_loss = per_token_loss.sum()
        token_count = per_token_loss.numel()

    if parallelism_type in ['ddp', 'fsdp']:
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    loss = (local_loss / token_count) / dist.get_world_size()
    return loss


def _save_model_checkpoint(model, optim, scheduler, loss, pplx, epoch, batch_idx,
                           rank, steps, save_path, keys, parallelism_type):
    """Save model checkpoint for training state.

    Args:
        model (nn.Module): Model to save.
        optim (torch.optim.Optimizer): Optimizer to save, or None.
        scheduler: Scheduler to save.
        loss (torch.Tensor): Current loss.
        pplx (float): Current perplexity.
        epoch (int): Current epoch.
        batch_idx (int): Current batch index.
        rank (int): Process rank.
        steps (int): Current global step count.
        save_path (str): Directory to save checkpoint.
        keys (tuple): Tuple of (_model_key, _optim_key, _scheduler_key) for checkpoint dictionary.
        parallelism_type (str): Type of parallelism ('fsdp', 'ddp', or None).
    """
    _model_key, _optim_key, _scheduler_key = keys
    if parallelism_type == 'fsdp':
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            m_sd = model.state_dict()
        o_sd = FSDP.full_optim_state_dict(model, optim) if optim else None
    else:
        m_sd = model.module.state_dict() if parallelism_type in ['dp', 'ddp'] else model.state_dict()
        o_sd = optim.state_dict() if optim else None

    if rank == 0:
        save_checkpoint(
            model_state_dict=m_sd,
            optim_state_dict=o_sd,
            scheduler=scheduler,
            loss=loss,
            pplx=pplx,
            epoch=epoch,
            batch_idx=batch_idx,
            rank=rank,
            global_steps=steps,
            save_checkpoint_path=save_path,
            _model_key=_model_key,
            _optim_key=_optim_key,
            _scheduler_key=_scheduler_key,
            parallelism_type=parallelism_type,
        )


def get_custom_scheduler(optimizer, warmup_steps, constant_steps, decay_steps, max_lr, min_lr):
    """Create a custom learning rate scheduler with warmup, constant, and cosine decay phases.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to apply the scheduler to.
        warmup_steps (int): Number of steps for linear warmup.
        constant_steps (int): Number of steps to maintain maximum learning rate.
        decay_steps (int): Number of steps for cosine decay.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        LambdaLR: Scheduler with custom learning rate schedule.
    """
    print(f'MAX LR {max_lr}')
    print(f'MIN LR {min_lr}') 

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

        # Normalize by the initial learning rate
        return lr / optimizer.param_groups[0]['initial_lr']

    return LambdaLR(optimizer, lr_lambda)


def get_config(type_: str):
    """Load configuration from a JSON file based on the specified type.

    Args:
        type_ (str): Type of configuration to load ('opt', 'model', 'lr', 'loss', 'train', 
                     'dataloader', 'parallel', 'wandb').

    Returns:
        dict: Configuration dictionary loaded from the JSON file.

    Raises:
        AssertionError: If type_ is not one of the valid configuration types.
    """
    valid_types = ['opt', 'model', 'lr', 'loss', 'train', 'dataloader', 'parallel', 'wandb']
    assert type_ in valid_types, f"type_ must be one of {', '.join(valid_types)}."

    config_path = f'src/config/{type_}_config.json'
    with open(config_path, 'r') as f:
        return json.load(f)


def load_checkpoint(
    model,
    optim,
    scheduler,
    load_checkpoint_path,
    _model_key,
    _optim_key,
    _scheduler_key,
    parallelism_type
):
    """Load a training checkpoint for model, optimizer, and scheduler.

    Args:
        model (nn.Module): Model to load state into.
        optim (torch.optim.Optimizer): Optimizer to load state into.
        scheduler: Scheduler to load state into.
        load_checkpoint_path (str): Path to the checkpoint file.
        _model_key (str): Key for model state in checkpoint.
        _optim_key (str): Key for optimizer state in checkpoint.
        _scheduler_key (str): Key for scheduler state in checkpoint.
        parallelism_type (str): Type of parallelism ('fsdp', 'ddp', 'dp', or None).

    Returns:
        tuple: Updated model, optimizer, epoch, global_steps, loss, and batch_idx.
    """
    print(f"Loading training checkpoint from {load_checkpoint_path}")

    checkpoint = torch.load(load_checkpoint_path, map_location='cpu')

    # Load model and optimizer state dicts depending on parallelism type
    if parallelism_type == 'fsdp':
        model.load_state_dict(checkpoint[_model_key])
        full_optim_state_dict = checkpoint[_optim_key]
        sharded_optim_state_dict = FSDP.shard_full_optim_state_dict(full_optim_state_dict, model)
        optim.load_state_dict(sharded_optim_state_dict)
    else:
        # For DDP and DP, model might be wrapped inside another module (i.e., DataParallel)
        if parallelism_type in ['dp', 'ddp'] and hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint[_model_key])
        else:
            model.load_state_dict(checkpoint[_model_key])
        optim.load_state_dict(checkpoint[_optim_key])

    # Load scheduler state dict and other checkpoint metadata
    scheduler.load_state_dict(checkpoint[_scheduler_key])
    epoch = checkpoint['epoch']
    global_steps = checkpoint['global_steps']
    loss = checkpoint.get('loss', None)
    batch_idx = checkpoint.get('batch_idx', 0)

    return model, optim, epoch, global_steps, loss, batch_idx

def _check_parallelism(parallelism_type, backend=None, verbose=False):
    """Check if parallelism type and backend are valid.

    Args:
        parallelism_type (str): Type of parallelism ('fsdp', 'ddp', or None).
        backend (str, optional): Backend for distributed training ('nccl', 'gloo'). Defaults to None.
        verbose (bool, optional): If True, print setup information. Defaults to False.

    Raises:
        ValueError: If backend is not specified for 'fsdp' or 'ddp', or if backend is invalid.
    """
    _par_check = ['fsdp', 'ddp']
    _backend_check = ['nccl', 'gloo']
    
    if verbose and parallelism_type in _par_check:
        print(f'Setting up {parallelism_type} using the {backend} backend')
    
    if parallelism_type in _par_check:
        if backend is None:
            raise ValueError('Backend must be specified for fsdp or ddp')
        if backend not in _backend_check:
            raise ValueError(f"Backend must be one of {', '.join(_backend_check)}")


def setup_parallel(parallelism_type, backend=None, sleep=3, verbose=False):
    """Set up parallel training environment and return configuration.

    Args:
        parallelism_type (str): Type of parallelism ('fsdp', 'ddp', 'dp', or None).
        backend (str, optional): Backend for distributed training ('nccl', 'gloo'). Defaults to None.
        sleep (int, optional): Seconds to sleep if CPU is used. Defaults to 3.
        verbose (bool, optional): If True, print setup information. Defaults to False.

    Returns:
        dict: Configuration with device, world_size, rank, and local_rank (if applicable).

    Raises:
        ValueError: If parallelism_type or backend is invalid.
    """
    _check_parallelism(parallelism_type, backend, verbose)

    rank, local_rank, world_size, device = None, None, None, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if parallelism_type in ['fsdp', 'ddp']:
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        local_rank = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        device = _set_device(local_rank, parallelism_type)
    elif parallelism_type in ['dp', None] and device.type == 'cpu':
        warnings.warn('CUDA unavailable, using CPU')
        time.sleep(sleep)

    out = {'device': device, 'world_size': world_size, 'rank': rank}
    if parallelism_type in ['fsdp', 'ddp']:
        out['local_rank'] = local_rank
    return out


def get_model_parallel(module, device='cuda', parallelism_type=None, type_=None, min_num_params=None, backend=None):
    """Prepare model for parallel training.

    Args:
        module (nn.Module): Model to parallelize.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
        parallelism_type (str, optional): Type of parallelism ('fsdp', 'ddp', 'dp', or None). Defaults to None.
        type_ (str, optional): FSDP wrapping type ('size' or 'transformer'). Defaults to None.
        min_num_params (int, optional): Minimum number of parameters for FSDP size-based wrapping. Defaults to None.
        backend (str, optional): Backend for distributed training ('nccl', 'gloo'). Defaults to None.

    Returns:
        nn.Module: Parallelized model.

    Raises:
        ValueError: If parallelism_type, type_, or min_num_params is invalid.
    """
    _parallel_type_list = ['fsdp', 'ddp', 'dp', None]
    if parallelism_type not in _parallel_type_list:
        raise ValueError("parallelism_type must be 'dp', 'ddp', 'fsdp', or None")
    
    print(f"Preparing model using {parallelism_type}{' and ' + backend if backend else ''}")

    if parallelism_type == 'dp':
        local_rank = 0
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        module = module.to(device)
        device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(module, device_ids=device_ids)
    elif parallelism_type == 'ddp':
        local_rank = dist.get_rank() % torch.cuda.device_count()
        device = f'cuda:{local_rank}'
        module = module.to(device)
        model = DDP(module, device_ids=[local_rank])
    elif parallelism_type == 'fsdp':
        if type_ is None or min_num_params is None:
            raise ValueError("type_ and min_num_params must be specified for 'fsdp'")
        local_rank = dist.get_rank() % torch.cuda.device_count()
        device = f'cuda:{local_rank}'
        module = module.to(device)
        model = _wrap_fsdp(module, type_, min_num_params)
    else:  # None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = module.to(device)
    
    return model


def save_checkpoint(model_state_dict, optim_state_dict, scheduler, loss, pplx, epoch, batch_idx, rank, global_steps,
                    save_checkpoint_path, _model_key, _optim_key, _scheduler_key, parallelism_type=None):
    """Save training checkpoint.

    Args:
        model_state_dict (dict): Model state dictionary.
        optim_state_dict (dict): Optimizer state dictionary.
        scheduler: Learning rate scheduler.
        loss (float or torch.Tensor): Training loss.
        pplx (float): Perplexity.
        epoch (int): Current epoch.
        batch_idx (int): Current batch index.
        rank (int): Process rank.
        global_steps (int): Total training steps.
        save_checkpoint_path (str): Directory to save checkpoint.
        _model_key (str): Key for model state in checkpoint.
        _optim_key (str): Key for optimizer state in checkpoint.
        _scheduler_key (str): Key for scheduler state in checkpoint.
        parallelism_type (str, optional): Type of parallelism ('fsdp', 'ddp', or None). Defaults to None.
    """
    print(f"Saving checkpoint at epoch {epoch} after {global_steps} steps on rank {rank}")
    os.makedirs(save_checkpoint_path, exist_ok=True)
    
    checkpoint = {
        _model_key: model_state_dict,
        _optim_key: optim_state_dict,
        _scheduler_key: scheduler.state_dict(),
        'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
        'pplx': pplx,
        'epoch': epoch,
        'batch_idx': batch_idx,
        'global_steps': global_steps
    }

    if parallelism_type in ['ddp', 'fsdp'] and rank != 0:
        return

    save_path = os.path.join(save_checkpoint_path, f'checkpoint_epoch_{epoch}_step_{batch_idx}_global_step_{global_steps}')
    print(f'Saving checkpoint to {save_path} on rank {rank}')
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path} on rank {rank}")


def _wrap_fsdp(module, type_='size', min_num_params=None, transformer_layer_cls=None, **fsdp_kwargs):
    """Wrap model with FSDP for distributed training.

    Args:
        module (nn.Module): Model to wrap.
        type_ (str, optional): Wrapping type ('size' or 'transformer'). Defaults to 'size'.
        min_num_params (int, optional): Minimum number of parameters for size-based wrapping. Defaults to None.
        transformer_layer_cls (type, optional): Transformer layer class for transformer-based wrapping. Defaults to None.
        **fsdp_kwargs: Additional FSDP arguments.

    Returns:
        FSDP: Wrapped FSDP model.

    Raises:
        ValueError: If type_ or min_num_params is invalid.
    """
    _type_list = ['size', 'transformer']
    if type_ not in _type_list:
        raise ValueError("type_ must be either 'size' or 'transformer'")
    
    if type_ == 'size':
        if min_num_params is None:
            raise ValueError("min_num_params must be specified for 'size'")
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    else:  # transformer
        if transformer_layer_cls is None:
            transformer_layer_cls = TransformerBlock
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls})
    
    return FSDP(module, auto_wrap_policy=auto_wrap_policy, **fsdp_kwargs)


def _set_device(rank, parallelism_type):
    """Set device for training based on rank and parallelism type.

    Args:
        rank (int): Process rank.
        parallelism_type (str): Type of parallelism ('fsdp', 'ddp', or None).

    Returns:
        torch.device: Selected device.

    Raises:
        RuntimeError: If rank exceeds available GPUs or CUDA is unavailable for 'fsdp' or 'ddp'.
    """
    if torch.cuda.is_available():
        if rank >= torch.cuda.device_count():
            raise RuntimeError(f"Rank {rank} exceeds available GPUs ({torch.cuda.device_count()})")
        torch.cuda.set_device(rank)
        return torch.device(f'cuda:{rank}')
    
    if parallelism_type in ['fsdp', 'ddp']:
        raise RuntimeError(f"CUDA unavailable for {parallelism_type}")
    
    warnings.warn('Using CPU')
    return torch.device('cpu')

def init_wandb(**wandb_kwargs):
    """Initialize Weights & Biases for experiment tracking.

    Args:
        **wandb_kwargs: Keyword arguments including api_key, project, dir, name, and id.

    Raises:
        AssertionError: If api_key is not provided.
    """
    assert wandb_kwargs.get('api_key'), "WandB API key must be provided"
    os.environ['WANDB_API_KEY'] = wandb_kwargs['api_key']
    os.environ['WANDB_INIT_TIMEOUT'] = '600'
    
    wandb.login()
    wandb.init(
        project=wandb_kwargs.get('project'),
        dir=wandb_kwargs.get('dir'),
        name=wandb_kwargs.get('name'),
        id=wandb_kwargs.get('id')
    )

def get_grad_norm(model):
    """Calculate gradient norms for model parameters.

    Args:
        model (nn.Module): Model to compute gradient norms for.

    Returns:
        dict: Dictionary mapping parameter names to their gradient norms.
    """
    grad_norm_dict = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            clean_name = name.replace('_fsdp_wrapped_module.', '').replace('._flat_param', '')
            param_norm = p.grad.norm(2)
            grad_norm_dict[clean_name] = param_norm
    return grad_norm_dict

def _save_config(input_dir: str, run: str, output_dir: str) -> None:
    """
    Reads multiple JSON files from the input directory and writes their contents to a 
    single `.txt` file inside a run-specific subdirectory of the output directory. 
    Each JSON file's name is used as a separator in the output file.
    
    Args:
        input_dir (str): Directory containing JSON files to be read.
        run (str or int): Run identifier used to create a subdirectory within output_dir.
        output_dir (str): Base directory where the output file will be saved.

    Returns:
        None: This function performs side effects (writing to a file).
    """
    run_subdir = Path(output_dir) / f"run_{run}"
    run_subdir.mkdir(parents=True, exist_ok=True)
    
    output_file = run_subdir / 'configs.txt'
    
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in os.listdir(input_dir):
                if filename.endswith('.json'):
                    file_path = Path(input_dir) / filename
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            json_data = json.load(infile)
                        outfile.write(f"--- {filename} ---\n")
                        outfile.write(json.dumps(json_data, indent=2))
                        outfile.write("\n\n")
                    except json.JSONDecodeError:
                        print(f"Warning: '{filename}' contains invalid JSON. Skipping.")
                    except Exception as e:
                        print(f"Warning: Error reading '{filename}': {e}. Skipping.")
        
        print(f"Successfully wrote data to '{output_file}'.")
    
    except Exception as e:
        print(f"Error: Failed to write to '{output_file}': {e}.")


def _get_eval_data(eval_data_path: str, eval_size: int) -> tuple:
    """
    Loads the evaluation data from disk, returning the validation inputs and targets 
    sliced to the specified size.

    Args:
        eval_data_path (str): Path to the directory containing 'X_val.pt' and 'Y_val.pt'.
        eval_size (int): Number of samples to retrieve from the validation dataset.

    Returns:
        tuple: A tuple containing the validation inputs (X_val) and targets (y_val).
    """
    X_val_path = Path(eval_data_path) / 'X_val.pt'
    Y_val_path = Path(eval_data_path) / 'Y_val.pt'
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X = torch.load(X_val_path, map_location='cpu')
        y = torch.load(Y_val_path, map_location='cpu')

    X_val = X[:eval_size, :512]
    y_val = y[:eval_size, :512]  
    assert X_val.shape == y_val.shape
    
    return X_val, y_val


def _clear_cache(clr_cuda_cache: bool = False, gc_: bool = False, *args, **kwargs) -> None:
    """
    Clears cache by deleting tensors and optionally clearing the CUDA cache or invoking garbage collection.

    Args:
        clr_cuda_cache (bool): Whether to clear the CUDA cache (default is False).
        gc_ (bool): Whether to call garbage collection (default is False).
        *args: Any tensors to delete (optional).

    Returns:
        None: This function performs side effects (deletes tensors, clears cache).
    """
    for i in args:
        if isinstance(i, torch.Tensor): 
            del i 
    if clr_cuda_cache:
        torch.cuda.empty_cache()
    if gc_:
        gc.collect()


def is_main_process() -> bool:
    """
    Determines whether the current process is the main process in a distributed setting.

    Returns:
        bool: True if the current process is the main process, otherwise False.
    """
    return dist.get_rank() == 0 if dist.is_initialized() else True


def __exit() -> None:
    """
    Destroys the distributed process group and exits the program.
    
    Returns:
        None: This function performs side effects (terminates the program).
    """
    dist.destroy_process_group()
    sys.exit(0)