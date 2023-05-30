import collections

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


BiEncoderOutput = collections.namedtuple(
    'BiEncoderOutput',
    [
        'q_pooled',
        'q_seq',
        'ctx_pooled',
        'ctx_seq'
    ]
)


class Interaction:
    def __init__(self, level='pooled', broadcast='local', func='cosine'):
        self.level = level
        self.broadcast = broadcast
        self.func = func

    def compute_score(self, q_vectors, ctx_vectors):
        if self.level == 'pooled':
            Q, D = list(q_vectors.size())
            if self.broadcast == 'local':
                # q*c x d -> q x c x d
                ctx_vectors = ctx_vectors.view((Q, -1, D))
            if self.func == 'cosine':
                # For global: q_vector: q x d -> q x 1 x d, ctx_vectors:   q*c x d, result: q x q*c
                # For local : q_vector: q x d -> q x 1 x d, ctx_vectors: q x c x d, result: q x c
                return F.cosine_similarity(q_vectors.unsqueeze(1), ctx_vectors, dim=-1)
            else:
                # For global: q_vector: q x d -> q x 1 x d, ctx_vectors: q*c x d -> d x q*c, result: q x q*c
                # For local : q_vector: q x d -> q x 1 x d, ctx_vectors: q x c x d -> q x d x c, result: q x c
                return (q_vectors.unsqueeze(1) @ ctx_vectors.transpose(-1, -2)).squeeze(1)
        else:
            Q, _, D = list(q_vectors.size())
            if self.broadcast == 'local':
                W = ctx_vectors.size(1)
                # q*c x w x d -> q x c x w x d
                ctx_vectors = ctx_vectors.view((Q, -1, W, D))
            if self.func == 'cosine':
                # For global:
                # q_vector   :   q x w x d -> q x  1  x w x 1 x d
                # ctx_vectors: q*c x w x d ->     q*c x 1 x w x d
                # result:                     q x q*c x w x w     -> q x q*c x w -> q x q*c
                # For local :
                # q_vector   :     q x w x d -> q x 1 x w x 1 x d
                # ctx_vectors: q x c x w x d -> q x c x 1 x w x d
                # result:                       q x c x w x w     -> q x c x w -> q x c
                return F.cosine_similarity(
                    q_vectors.unsqueeze(2).unsqueeze(1),
                    ctx_vectors.unsqueeze(-3),
                    dim=-1
                ).max(-1).values.sum(-1)
            else:
                # global:
                # q_vector   :   q x w x d -> q x  1  x w x d
                # ctx_vectors: q*c x w x d ->     q*c x d x w
                # result:                     q x q*c x w x w -> q x q*c x w -> q x q*c
                # local:
                # q_vector   :     q x w x d -> q x 1 x w x d,
                # ctx_vectors: q x c x w x d -> q x c x d x w,
                # result:                       q x c x w x w -> q x c x w -> q x c
                return (q_vectors.unsqueeze(1) @ ctx_vectors.transpose(-1, -2)).max(-1).values.sum(-1)


class BiEncoderNllLoss:
    def __init__(self, level='pooled', broadcast='local', func='cosine', temperature=1.0):
        self.level = level
        self.broadcast = broadcast
        self.func = func
        self.temperature = temperature
        self.interaction = Interaction(level=level, broadcast=broadcast, func=func)

    def compute_loss(
        self,
        model_output: BiEncoderOutput,
        cids_per_qid: list,
        pos_cids_per_qid: list
    ):
        if self.level == "pooled":
            # q_vectors: q x d
            # ctx_vectors: c x d
            q_vectors = model_output.q_pooled
            ctx_vectors = model_output.ctx_pooled
        else:
            # q_vectors: q x w x d
            # ctx_vectors: c x w x d
            q_vectors = model_output.q_seq
            ctx_vectors = model_output.ctx_seq

        scores = self.interaction.compute_score(q_vectors, ctx_vectors) / self.temperature

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)
        pos_ctx_indices = []
        start = 0
        for cids, pos_cids in zip(cids_per_qid, pos_cids_per_qid):
            cum_pos = start + cids.index(pos_cids[0])
            pos_ctx_indices.append(cum_pos)
            if self.broadcast == 'global':
                start += len(cids)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(pos_ctx_indices).to(softmax_scores.device),
            reduction='mean'
        )

        # max_score, max_idxs = torch.max(softmax_scores, 1)
        # correct_predictions_count = (max_idxs == torch.tensor(pos_ctx_indices).to(max_idxs.device)).sum()
        return loss


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, question_model: nn.Module, ctx_model: nn.Module, is_cross_interaction: bool):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.is_cross_interaction = is_cross_interaction

    @staticmethod
    def get_representation(
            sub_model: nn.Module,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None
    ):
        last_hidden_state, pooled_output = None, None
        if input_ids is not None:
            last_hidden_state, pooled_output = sub_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=False,
                output_attentions=False
            )

        return last_hidden_state, pooled_output

    def forward(
            self,
            q_input_ids, q_attention_mask, q_token_type_ids,
            ctx_input_ids, ctx_attention_mask, ctx_token_type_ids
    ):
        q_seq, q_pooled = self.get_representation(
            self.question_model,
            q_input_ids, q_attention_mask, q_token_type_ids
        )
        ctx_seq, ctx_pooled = self.get_representation(
            self.ctx_model,
            ctx_input_ids, ctx_attention_mask, ctx_token_type_ids
        )
        if self.is_cross_interaction:
            return BiEncoderOutput(
                None,
                q_seq,
                None,
                ctx_seq
            )
        else:
            return BiEncoderOutput(
                q_pooled,
                None,
                ctx_pooled,
                None
            )
