from typing import Tuple

import torch
from torch import Tensor as T
from torch import FloatTensor as FT
from torch import nn
from transformers.models.bert.modeling_bert import BertModel, BertConfig
from transformers import BertTokenizer
from transformers.optimization import AdamW

from dpr.utils.data_utils import Tensorizer
from .biencoder import BiEncoder


class HFBertEncoder(BertModel):

    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else 'bert-base-uncased')
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[FT, ...]:
        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel.forward
        model_out = super().forward(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

        sequence_output, pooled_output, hidden_states = model_out.last_hidden_state, model_out.pooler_output, model_out.hidden_states
        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True, pad_token_id: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(title,
                                              text_pair=text,
                                              add_special_tokens=add_special_tokens,
                                              max_length=self.max_length,
                                              pad_to_max_length=False)
        else:
            token_ids = self.tokenizer.encode(text,
                                              add_special_tokens=add_special_tokens,
                                              max_length=self.max_length,
                                              pad_to_max_length=False)

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        output = tokens_tensor != self.pad_token_id
        output[:, 0] = True
        return output

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


def get_ance_tensorizer(cfg, tokenizer=None):
    if not tokenizer:
        if cfg.DPR.MODEL.PRETRAINED_MODEL_CFG and 'bert-' in cfg.DPR.MODEL.PRETRAINED_MODEL_CFG:
            tokenizer = get_bert_tokenizer(cfg.DPR.MODEL.PRETRAINED_MODEL_CFG, do_lower_case=cfg.DPR.MODEL.DO_LOWER_CASE)
        else:
            raise NotImplementedError

    if cfg.DPR.MODEL.PRETRAINED_MODEL_CFG and 'bert-' in cfg.DPR.MODEL.PRETRAINED_MODEL_CFG:
        return BertTensorizer(tokenizer, cfg.DPR.MODEL.SEQUENCE_LENGTH, pad_token_id=0)
    else:
        raise NotImplementedError


def get_optimizer(model: nn.Module, learning_rate: float = 1e-5, adam_eps: float = 1e-8,
                  weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps, correct_bias=False)
    return optimizer


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.DPR.SOLVER.DROPOUT if hasattr(cfg.DPR.SOLVER, 'DROPOUT') else 0.0

    # modify based on cfg so it could load both bert and roberta models
    if 'hf' in cfg.DPR.MODEL.ENCODER_MODEL_TYPE:
        if cfg.DPR.MODEL.PRETRAINED_MODEL_CFG and 'bert-' in cfg.DPR.MODEL.PRETRAINED_MODEL_CFG:
            question_encoder = HFBertEncoder.init_encoder(cfg.DPR.MODEL.PRETRAINED_MODEL_CFG,
                                                          projection_dim=cfg.DPR.MODEL.PROJECTION_DIM, dropout=dropout, **kwargs)
            ctx_encoder = HFBertEncoder.init_encoder(cfg.DPR.MODEL.PRETRAINED_MODEL_CFG,
                                                     projection_dim=cfg.DPR.MODEL.PROJECTION_DIM, dropout=dropout, **kwargs)
        else:
            raise NotImplementedError
        tensorizer = get_ance_tensorizer(cfg)
    else:
        raise NotImplementedError(f'{cfg.DPR.MODEL.ENCODER_MODEL_TYPE} is not implemented yet.')

    fix_ctx_encoder = cfg.DPR.MODEL.FIX_CTX_ENCODER if hasattr(cfg.DPR.MODEL, 'fix_ctx_encoder') else False
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = get_optimizer(biencoder,
                              learning_rate=cfg.DPR.SOLVER.OPTIMIZER.BASE_LR,
                              adam_eps=cfg.DPR.SOLVER.OPTIMIZER.EPS,
                              weight_decay=cfg.DPR.SOLVER.OPTIMIZER.WEIGHT_DECAY,
                              ) if not inference_only else None
    return tensorizer, biencoder, optimizer
