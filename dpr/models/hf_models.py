import logging
from typing import Tuple, Optional
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertModel, BertConfig
from .pooling import CLSPooling, ProjectionPooling, MeanPooling, MinPooling, MaxPooling, AttentionPooling
logger = logging.getLogger()


class HFBertEncoder(BertModel):
    def __init__(self, config, dpr_config, skip_list, special_tokens):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'

        self.skip_list = skip_list
        self.special_tokens = special_tokens
        self.dpr_config = dpr_config
        self.projection_layer = None
        if dpr_config.CROSS_INTERACTION and dpr_config.PROJECTION_DIM:
            self.projection_layer = nn.Linear(config.hidden_size, dpr_config.PROJECTION_DIM)
        self.pooling_layer = self.init_pooling(dpr_config.POOLING)
        self.init_weights()

    @classmethod
    def init_encoder(cls, dpr_config, skip_list, special_tokens, **kwargs) -> BertModel:
        bert_config = BertConfig.from_pretrained(dpr_config.PRETRAINED_MODEL_CFG if dpr_config.PRETRAINED_MODEL_CFG else 'bert-base-uncased')
        if dpr_config.DROPOUT != 0:
            bert_config.attention_probs_dropout_prob = dpr_config.DROPOUT
            bert_config.hidden_dropout_prob = dpr_config.DROPOUT
        return cls.from_pretrained(
            dpr_config.PRETRAINED_MODEL_CFG,
            config=bert_config,
            dpr_config=dpr_config,
            skip_list=skip_list,
            special_tokens=special_tokens, **kwargs)

    def init_pooling(self, pool_type):
        if pool_type == 'mean':
            return MeanPooling()
        elif pool_type == 'min':
            return MinPooling()
        elif pool_type == 'max':
            return MaxPooling()
        elif pool_type == 'attention':
            if self.dpr_config.CROSS_INTERACTION and self.dpr_config.PROJECTION_DIM:
                return AttentionPooling(self.dpr_config.PROJECTION_DIM)
            elif not self.dpr_config.PROJECTION_DIM:
                return AttentionPooling(self.config.PROJECTION_DIM)
            else:
                raise ValueError('Can not specify projection dim with attention pooling')
        elif pool_type == 'projection':
            if self.dpr_config.PROJECTION_DIM:
                return ProjectionPooling(self.config.hidden_size, self.dpr_config.PROJECTION_DIM)
            else:
                raise ValueError('Should specify projection dim with projection pooling')
        else:
            return CLSPooling()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            **kwargs
    ):
        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel.forward
        model_out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        last_hidden_state = model_out.last_hidden_state
        # pooled_output = model_out.pooler_output
        # hidden_states = model_out.hidden_states
        if self.dpr_config.CROSS_INTERACTION:
            if self.dpr_config.PROJECTION_DIM:
                last_hidden_state = self.projection_layer(last_hidden_state)
            mask = torch.tensor(self.mask(input_ids), device=last_hidden_state.device).unsqueeze(2).float()
            last_hidden_state = last_hidden_state * mask
            # last_hidden_state = torch.nn.functional.normalize(last_hidden_state, p=2, dim=2)
        if self.dpr_config.POOLING == 'projection':
            pooled_output = self.pooling_layer(model_out.last_hidden_state, attention_mask)
        else:
            pooled_output = self.pooling_layer(last_hidden_state, attention_mask)
        return last_hidden_state, pooled_output

    def get_out_size(self):
        return self.dpr_config.PROJECTION_DIM if self.dpr_config.PROJECTION_DIM else self.config.hidden_size

    def mask(self, input_ids):
        mask = [[(x not in self.skip_list) and (x not in self.special_tokens) for x in d] for d in input_ids.cpu().tolist()]
        return mask


