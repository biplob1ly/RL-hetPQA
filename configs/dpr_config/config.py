"""
config test
Usage:
    config.py --path_output=<path> --path_data=<path> [--path_train_data=<path>] [--path_val_data=<path>] [--path_cfg_exp=<path>]
    config.py -h | --help

Options:
    -h --help                   show this screen help
    --path_output=<path>        output path
    --path_data=<path>          data path
    --path_train_data=<path>    train data path
    --path_val_data=<path>      validation data path
    --path_cfg_exp=<path>       experiment config path
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
_C.EXP = 'default'
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
_C.DPR.DO_TRAIN = True
_C.DPR.DO_TEST = True
# -----------------------------------------------------------------------------
# DPR.DATA
# -----------------------------------------------------------------------------
_C.DPR.DATA = CfgNode()
_C.DPR.DATA.NAME = 'hetPQA'
_C.DPR.DATA.DATA_PATH = './data/evidence_ranking/'
_C.DPR.DATA.TRAIN_DATA_PATH = './data/evidence_ranking/train.json'
_C.DPR.DATA.VAL_DATA_PATH = './data/evidence_ranking/dev.json'
_C.DPR.DATA.TEST_DATA_PATH = './data/evidence_ranking/test.json'
_C.DPR.DATA.NUM_POSITIVE_CONTEXTS = 1
_C.DPR.DATA.NUM_TOTAL_CONTEXTS = 5
_C.DPR.DATA.INSERT_SOURCE = True
_C.DPR.DATA.NORMALIZE = False
_C.DPR.DATA.FLATTEN_ATTRIBUTE = False

# -----------------------------------------------------------------------------
# DPR_MODEL
# -----------------------------------------------------------------------------
_C.DPR.MODEL = CfgNode()
_C.DPR.MODEL.MODEL_PATH = None
# config name for DPR.MODEL initialization
_C.DPR.MODEL.PRETRAINED_MODEL_CFG = 'bert-base-uncased'
_C.DPR.MODEL.CROSS_INTERACTION = False
# Extra linear layer on top of standard bert/roberta encoder
_C.DPR.MODEL.POOLING = 'cls'
_C.DPR.MODEL.PROJECTION_DIM = None
# Max length of the encoder input sequence
_C.DPR.MODEL.QUESTION_MAX_LENGTH = 64
_C.DPR.MODEL.CONTEXT_MAX_LENGTH = 96
# Whether to lower case the input text. Set True for uncased DPR.MODELs, False for the cased ones.
_C.DPR.MODEL.DO_LOWER_CASE = True
_C.DPR.MODEL.CHECKPOINT_FILE_NAME = 'dpr_ckpt'
_C.DPR.MODEL.DROPOUT = 0.1

_C.DPR.SOLVER = CfgNode()
_C.DPR.SOLVER.TRAIN_BATCH_SIZE = 2
_C.DPR.SOLVER.TEST_BATCH_SIZE = 1
_C.DPR.SOLVER.TEST_CTX_BSZ = 32
_C.DPR.SOLVER.TOTAL_TRAIN_STEPS = 1000
_C.DPR.SOLVER.NUM_STEP_PER_EVAL = 300
_C.DPR.SOLVER.CP_SAVE_LIMIT = 1
# Logging interval in terms of batches
_C.DPR.SOLVER.RESET_CHECKPOINT_STEP = False

_C.DPR.SOLVER.LEVEL = 'pooled'
_C.DPR.SOLVER.BROADCAST = 'local'
_C.DPR.SOLVER.FUNC = 'cosine'
_C.DPR.SOLVER.TEMPERATURE = 0.5

_C.DPR.SOLVER.OPTIMIZER = CfgNode()
_C.DPR.SOLVER.OPTIMIZER.NAME = 'AdamW'
# Linear warmup over warmup_steps.
_C.DPR.SOLVER.OPTIMIZER.WARMUP_STEPS = 100
_C.DPR.SOLVER.OPTIMIZER.BASE_LR = 1e-5
_C.DPR.SOLVER.OPTIMIZER.EPS = 1e-8
_C.DPR.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.00
_C.DPR.SOLVER.OPTIMIZER.BETAS = (0.9, 0.999)
_C.DPR.SOLVER.OPTIMIZER.RESET = False
_C.DPR.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS = 1
_C.DPR.SOLVER.OPTIMIZER.MAX_GRAD_NORM = 1.0

# Amount of top docs to return
_C.DPR.SOLVER.TOP_RETRIEVE_COUNT = 5
# Temporal memory data buffer size (in samples) for indexer
_C.DPR.SOLVER.INDEX_BUFFER_SIZE = 500


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
    data_path = arguments['--path_data']
    train_data_path = arguments['--path_train_data']
    val_data_path = arguments['--path_val_data']
    exp_cfg_path = arguments['--path_cfg_exp']
    config = get_cfg_defaults()
    if data_path is not None:
        config.DPR.DATA.DATA_PATH = data_path
        config.DPR.DATA.TRAIN_DATA_PATH = os.path.join(data_path, 'train.json')
        config.DPR.DATA.VAL_DATA_PATH = os.path.join(data_path, 'dev.json')
    if train_data_path is not None:
        config.DPR.DATA.TRAIN_DATA_PATH = train_data_path
    if val_data_path is not None:
        config.DPR.DATA.VAL_DATA_PATH = val_data_path
    if exp_cfg_path is not None:
        config.merge_from_file(exp_cfg_path)
    if output_path is not None:
        config.OUTPUT_PATH = output_path

    # Make result folders if they do not exist
    print(config)
    # config.dump(stream=open(os.path.join(exp_dir, f'config_{config.EXP}.yaml'), 'w'))
    # python - m src.tools.train - o experiments/exp10 --cfg src/config/experiments/exp10.yaml