from typing import Tuple

import string
import torch
from torch import Tensor as T
from torch import FloatTensor as FT
from torch import nn
from transformers.models.bert.modeling_bert import BertModel, BertConfig
from transformers import BertTokenizer
# Use the PyTorch implementation torch.optim.AdamW instead
from torch.optim import AdamW

from dpr.utils.data_utils import Tensorizer
from .biencoder import BiEncoder


class HFBertEncoder(BertModel):

    def __init__(self, config, dpr_config, skip_list, pad_token):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'

        self.skip_list = skip_list
        self.pad_token = pad_token
        self.pooling_layer = nn.Linear(config.hidden_size, dpr_config.POOLING_PROJECTION_DIM) if dpr_config.POOLING_PROJECTION_DIM != 0 else None
        self.sequence_layer = nn.Linear(config.hidden_size, dpr_config.SEQUENCE_PROJECTION_DIM) if dpr_config.SEQUENCE_PROJECTION_DIM != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, dpr_config, skip_list, pad_token, **kwargs) -> BertModel:
        bert_config = BertConfig.from_pretrained(dpr_config.PRETRAINED_MODEL_CFG if dpr_config.PRETRAINED_MODEL_CFG else 'bert-base-uncased')
        if dpr_config.DROPOUT != 0:
            bert_config.attention_probs_dropout_prob = dpr_config.DROPOUT
            bert_config.hidden_dropout_prob = dpr_config.DROPOUT
        return cls.from_pretrained(
            dpr_config.PRETRAINED_MODEL_CFG,
            config=bert_config,
            dpr_config=dpr_config,
            skip_list=skip_list,
            pad_token=pad_token, **kwargs)

    def pool(self, outputs, attention_mask, batch_size, item_count):
        if self.model_args.pooler_type == 'cls_without_pool':
            pooled_output = outputs.last_hidden_state[:, 0]
        elif self.model_args.pooler_type == 'avg':
            # last_hidden_state: [B x Item_count, S, H]
            pooled_output = (outputs.last_hidden_state * attention_mask.unsqueeze(-1).sum(1)) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.model_args.pooler_type == 'avg_first_last':
            first_last = (outputs.last_hidden_state[0] + outputs.last_hidden_state[-1]) / 2.0
            pooled_output = (first_last * attention_mask.unsqueeze(-1).sum(1)) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.model_args.pooler_type == 'avg_top2':
            top2 = (outputs.last_hidden_state[-1] + outputs.last_hidden_state[-2]) / 2.0
            pooled_output = (top2 * attention_mask.unsqueeze(-1).sum(1)) / attention_mask.sum(-1).unsqueeze(-1)
        else:
            pooled_output = outputs.last_hidden_state[:, 0]

        pooled_output = pooled_output.view((batch_size, item_count, pooled_output.size(-1)))
        if self.model_args.pooler_type == 'cls':
            pooled_output = self.activation(self.fc(pooled_output))
        return pooled_output

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[FT, ...]:
        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel.forward
        model_out = super().forward(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

        sequence_output, pooled_output, hidden_states = model_out.last_hidden_state, model_out.pooler_output, model_out.hidden_states
        pooled_output = sequence_output[:, 0, :]
        if self.sequence_layer:
            sequence_output = self.sequence_layer(sequence_output)
            mask = torch.tensor(self.mask(input_ids), device=sequence_output.device).unsqueeze(2).float()
            sequence_output = sequence_output * mask
            # sequence_output = torch.nn.functional.normalize(sequence_output, p=2, dim=2)
        if self.pooling_layer:
            pooled_output = self.pooling_layer(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.pooling_layer:
            return self.pooling_layer.out_features
        return self.config.hidden_size

    def mask(self, input_ids):
        mask = [[(x not in self.skip_list) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()]
        return mask


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

    def get_skip_list(self, tokens):
        return {w: True
                for symbol in tokens
                for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}


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
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    # modify based on cfg so it could load both bert and roberta models
    if 'hf' in cfg.DPR.MODEL.ENCODER_MODEL_TYPE:
        tensorizer = get_ance_tensorizer(cfg)
        skip_list = tensorizer.get_skip_list(string.punctuation)
        pad_token = tensorizer.get_pad_id()
        if cfg.DPR.MODEL.PRETRAINED_MODEL_CFG and 'bert-' in cfg.DPR.MODEL.PRETRAINED_MODEL_CFG:
            question_encoder = HFBertEncoder.init_encoder(cfg.DPR.MODEL, [], pad_token, **kwargs)
            ctx_encoder = HFBertEncoder.init_encoder(cfg.DPR.MODEL, skip_list, pad_token, **kwargs)
        else:
            raise NotImplementedError
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
