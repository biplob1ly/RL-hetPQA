# Credit: https://github.com/facebookresearch/DPR

import importlib


def init_hf_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from dpr.models.hf_models import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)