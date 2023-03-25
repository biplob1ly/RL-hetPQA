import collections
import logging
import glob
import os

import torch
import torch.nn as nn
from torch.serialization import default_restore_location
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger()
CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict',
                                          'optimizer_dict',
                                          'scheduler_dict',
                                          'offset',
                                          'epoch',
                                          'encoder_params'])


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model


def get_model_file(cfg, file_prefix) -> str:
    if cfg.DPR.MODEL.MODEL_FILE and os.path.exists(cfg.DPR.MODEL.MODEL_FILE):
        return cfg.DPR.MODEL.MODEL_FILE

    out_cp_files = glob.glob(os.path.join(cfg.OUTPUT_PATH, file_prefix + "*")) if cfg.OUTPUT_PATH else []
    logger.info("Checkpoint files %s", out_cp_files)
    model_file = None

    if len(out_cp_files) > 0:
        model_file = max(out_cp_files, key=os.path.getctime)
        logger.info('Selected file to load: %s', model_file)
    else:
        logger.info('No checkpoint file found at output path: %s', cfg.OUTPUT_PATH)
    return model_file


def load_states_from_checkpoint(checkpoint_file_path: str) -> CheckpointState:
    logger.info('Reading saved model from %s', checkpoint_file_path)
    state_dict = torch.load(checkpoint_file_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


def setup_for_distributed_mode(model: nn.Module, optimizer: torch.optim.Optimizer, device: str, n_gpu: int = 1,
                               LOCAL_RANK: int = -1,
                               fp16: bool = False,
                               fp16_opt_level: str = "O1") -> (nn.Module, torch.optim.Optimizer):
    model.to(device)
    if fp16:
        try:
            import apex
            from apex import amp
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        if optimizer is None:
            model = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if LOCAL_RANK != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK],
                                                          output_device=LOCAL_RANK,
                                                          find_unused_parameters=True)
    return model, optimizer


def get_schedule_linear(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):

    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_device(value, device)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)