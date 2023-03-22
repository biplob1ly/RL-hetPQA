"""
train model
Usage:
    train.py [--path_output=<path>] [--path_cfg_data=<path>]  [--path_cfg_override=<path>]
    train.py -h | --help
Options:
    -h --help               show this screen help
    --path_output=<path>               output path
    --path_cfg_data=<path>       data config path
    --path_cfg_override=<path>            training config path
"""
# [default: configs/data.yaml]
import os
import warnings
from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode
from pathlib import Path
from docopt import docopt

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# importing default as a global singleton
# cfg = _C
_C.DESCRIPTION = 'Default config from the Singleton'
_C.VERSION = 0
_C.OUTPUT_PATH = './output/'
_C.EXP = None
_C.SEED = 42
_C.DEVICE = None
_C.LOCAL_RANK = -1
_C.DISTRIBUTED_WORLD_SIZE = None
_C.NO_CUDA = False
_C.N_GPU = None
_C.FP16 = False
# For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#        "See details at https://nvidia.github.io/apex/amp.html
_C.FP16_OPT_LEVEL = 'O1'


_C.DPR = CfgNode()
# -----------------------------------------------------------------------------
# DPR.DATA
# -----------------------------------------------------------------------------
_C.DPR.DATA = CfgNode()
_C.DPR.DATA.NAME = 'hetPQA'
_C.DPR.DATA.TRAIN_DATA_PATH = './data/evidence_ranking_train_grouped.json'
_C.DPR.DATA.VAL_DATA_PATH = './data/evidence_ranking_dev_grouped.json'
_C.DPR.DATA.HARD_NEGATIVES = 1
_C.DPR.DATA.OTHER_NEGATIVES = 0

# -----------------------------------------------------------------------------
# DPR_MODEL
# -----------------------------------------------------------------------------
_C.DPR.MODEL = CfgNode()
# config name for DPR.MODEL initialization
_C.DPR.MODEL.PRETRAINED_MODEL_CFG = 'bert-base-uncased'
# DPR.MODEL type. One of [hf_bert, pytext_bert, fairseq_roberta]
_C.DPR.MODEL.ENCODER_MODEL_TYPE = 'hf_bert'
# Some encoders need to be initialized from a file
_C.DPR.MODEL.PRETRAINED_FILE = None
# Extra linear layer on top of standard bert/roberta encoder
_C.DPR.MODEL.PROJECTION_DIM = 0
# Max length of the encoder input sequence
_C.DPR.MODEL.SEQUENCE_LENGTH = 512
# Whether to lower case the input text. Set True for uncased DPR.MODELs, False for the cased ones.
_C.DPR.MODEL.DO_LOWER_CASE = True
_C.DPR.MODEL.CHECKPOINT_FILE_NAME = 'dpr_biencoder'
# A trained bi-encoder checkpoint file to initialize the model
_C.DPR.MODEL.MODEL_FILE = None
_C.DPR.MODEL.FIX_CTX_ENCODER = False

_C.DPR.SOLVER = CfgNode()
_C.DPR.SOLVER.TRAIN_BATCH_SIZE = 2
_C.DPR.SOLVER.VAL_BATCH_SIZE = 4
_C.DPR.SOLVER.NUM_TRAIN_EPOCH = 1
_C.DPR.SOLVER.EVAL_PER_EPOCH = 1
_C.DPR.SOLVER.VAL_AV_RANK_START_EPOCH = 1
_C.DPR.SOLVER.VAL_AV_RANK_HARD_NEG = 30
_C.DPR.SOLVER.VAL_AV_RANK_OTHER_NEG = 30
_C.DPR.SOLVER.VAL_AV_RANK_BSZ = 128
_C.DPR.SOLVER.VAL_AV_RANK_MAX_QS = 1000
# Logging interval in terms of batches
_C.DPR.SOLVER.LOG_BATCH_STEP = 2
_C.DPR.SOLVER.TRAIN_ROLLING_LOSS_STEP = 2

_C.DPR.SOLVER.DROPOUT = 0.1
_C.DPR.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1
# Linear warmup over warmup_steps.
_C.DPR.SOLVER.WARMUP_STEPS = 100
_C.DPR.SOLVER.MAX_GRAD_NORM = 1.0

_C.DPR.SOLVER.OPTIMIZER = CfgNode()
_C.DPR.SOLVER.OPTIMIZER.NAME = 'AdamW'
_C.DPR.SOLVER.OPTIMIZER.BASE_LR = 1e-5
_C.DPR.SOLVER.OPTIMIZER.EPS = 1e-8
_C.DPR.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.0
_C.DPR.SOLVER.OPTIMIZER.BETAS = (0.9, 0.999)
_C.DPR.SOLVER.OPTIMIZER.RESET = False


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()


def update_cfg_using_dotenv() -> list:
    """
    In case when there are dotenvs, try to return list of them.
    # It is returning a list of hard overwrite.
    :return: empty list or overwriting information
    """
    # If .env not found, bail
    if find_dotenv() == '':
        warnings.warn(".env files not found. YACS config file merging aborted.")
        return []

    # Load env.
    load_dotenv(find_dotenv(), verbose=True)

    # Load variables
    list_key_env = {
        "DPR.DATA.TRAIN_DATA_PATH",
        "DPR.DATA.VAL_DATA_PATH",
        "DPR_MODEL.BACKBONE.PRETRAINED_PATH",
        "DPR_SOLVER.LOSS.LABELS_WEIGHTS_PATH"
    }

    # Instantiate return list.
    path_overwrite_keys = []

    # Go through the list of key to be overwritten.
    for key in list_key_env:

        # Get value from the env.
        value = os.getenv("path_overwrite_keys")

        # If it is none, skip. As some keys are only needed during training and others during the prediction stage.
        if value is None:
            continue

        # Otherwise, adding the key and the value to the dictionary.
        path_overwrite_keys.append(key)
        path_overwrite_keys.append(value)

    return path_overwrite_keys


def combine_cfgs(path_cfg_data: Path=None, path_cfg_override: Path=None):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data = Path(path_cfg_data)
    if path_cfg_override is not None:
        path_cfg_override=Path(path_cfg_override)
    # Path order of precedence is:
    # Priority 1, 2, 3, 4 respectively
    # .env > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 4:
    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    # Priority 3:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # Merge from other cfg_path files to further reduce effort
    # Priority 2:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    # Merge from .env
    # Priority 1:
    list_cfg = update_cfg_using_dotenv()
    if list_cfg is not []:
        cfg_base.merge_from_list(list_cfg)

    return cfg_base


if __name__ == '__main__':
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    output_path = arguments['--path_output']
    data_path = arguments['--path_cfg_data']
    cfg_path = arguments['--path_cfg_override']
    cfg = get_cfg_defaults()
    if data_path is not None:
        print(data_path)
        cfg.merge_from_file(data_path)
    if cfg_path is not None:
        cfg.merge_from_file(cfg_path)
    if output_path is not None:
        cfg.OUTPUT_PATH = output_path
    # train(cfg)

    # Make result folders if they do not exist
    exp_dir = os.path.join(cfg.OUTPUT_PATH, cfg.EXP)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    print(cfg)
    cfg.dump(stream=open(os.path.join(exp_dir, f'config_{cfg.EXP}.yaml'), 'w'))
    # python - m src.tools.train - o experiments/exp10 --cfg src/config/experiments/exp10.yaml