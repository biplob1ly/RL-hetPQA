import os
import random
import torch
import socket
import logging
import numpy as np
logger = logging.getLogger()


def get_encoder_checkpoint_params_names():
    return ['PRETRAINED_MODEL_CFG',
            'ENCODER_MODEL_TYPE',
            'PRETRAINED_FILE',
            'POOLING_PROJECTION_DIM',
            'SEQUENCE_PROJECTION_DIM',
            'SEQUENCE_LENGTH',
            'DO_LOWER_CASE']


def get_encoder_params_state(cfg):
    """
     Selects the param values to be saved in a checkpoint, so that a trained model faile can be used for downstream
     tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    params_to_save = get_encoder_checkpoint_params_names()
    r = {}
    for param in params_to_save:
        r[param] = getattr(cfg.DPR.MODEL, param)
    return r


def set_encoder_params_from_state(state, cfg):
    if not state:
        return
    params_to_save = get_encoder_checkpoint_params_names()

    override_params = [(param, state[param]) for param in params_to_save if param in state and state[param]]
    for param, value in override_params:
        if hasattr(cfg.DPR.MODEL, param):
            logger.warning('Overriding cfg parameter value from checkpoint state. Param = %s, value = %s', param,
                           value)
        setattr(cfg.DPR.MODEL, param, value)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_cfg_gpu(cfg):
    """
     Setup arguments CUDA, GPU & distributed training
    """

    if cfg.LOCAL_RANK == -1 or cfg.NO_CUDA:  # single-node multi-gpu (or cpu) mode
        device = torch.device("cuda" if torch.cuda.is_available() and not cfg.NO_CUDA else "cpu")
        cfg.N_GPU = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(cfg.LOCAL_RANK)
        device = torch.device("cuda", cfg.LOCAL_RANK)
        if not torch.distributed.is_initialized:
            torch.distributed.init_process_group(backend="nccl")
        cfg.N_GPU = 1
    cfg.DEVICE = str(device)
    ws = os.environ.get('WORLD_SIZE')
    cfg.DISTRIBUTED_WORLD_SIZE = int(ws) if ws else 1

    logger.info(
        'Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d', socket.gethostname(),
        cfg.LOCAL_RANK, cfg.DEVICE,
        cfg.N_GPU,
        cfg.DISTRIBUTED_WORLD_SIZE)
    logger.info("16-bits training: %s ", cfg.FP16)
    return cfg
