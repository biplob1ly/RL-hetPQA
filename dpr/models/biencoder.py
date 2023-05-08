import collections
import numpy as np
import random

import torch
from torch import Tensor as T
import torch.nn.functional as F
from torch import nn
from typing import Tuple, List
from dpr.utils.data_utils import Tensorizer, BiEncoderSample

BiEncoderBatch = collections.namedtuple(
    'BiENcoderInput',
    [
        'question_ids',
        'question_segments',
        'context_ids',
        'ctx_segments',
        'positive_ctx_indices',
        'hard_neg_ctx_indices'
    ]
)

BiEncoderSingle = collections.namedtuple(
    'BiENcoderSingle',
    [
         'question_ids',
         'question_segments',
         'context_ids',
         'ctx_segments'
    ]
)

BiEncoderOutput = collections.namedtuple(
    'BiEncoderOutput',
    [
        'q_pooled',
        'q_seq',
        'ctx_pooled',
        'ctx_seq'
    ]
)


def repr_dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vectors:
    :param ctx_vectors:
    :return:
    """
    # q_vector: q x d, ctx_vectors: c x d, result: q x c
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def repr_cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: q x d -> q x 1 x d, ctx_vectors: c x d, result: q x c
    return F.cosine_similarity(q_vector.unsqueeze(1), ctx_vectors, dim=-1)


def cross_dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vectors:
    :param ctx_vectors:
    :return:
    """
    # q_vector: q x w x d -> q x 1 x w x d, ctx_vectors: c x w x d -> c x d x w, result: q x c x w x w -> q x c x w -> q x c
    return (q_vectors.unsqueeze(1) @ ctx_vectors.permute(0, 2, 1)).max(3).values.sum(2)


def cross_cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: q x w x d -> q x 1 x w x 1 x d, ctx_vectors: c x w x d -> c x 1 x w x d, result: q x c x w x w -> q x c x w -> q x c
    return F.cosine_similarity(q_vector.unsqueeze(2).unsqueeze(1), ctx_vectors.unsqueeze(1), dim=-1).max(3).values.sum(2)


map_comp_func = {
    'representaton_matching': {
        "dot_product": repr_dot_product_scores,
        "cos_sim": repr_cosine_scores
    },
    'cross_interaction': {
        "dot_product": cross_dot_product_scores,
        "cos_sim": cross_cosine_scores
    }
}


class BiEncoderNllLoss:
    def __init__(self, comparison_type: str = 'representaton_matching', comparison_function: str ='dot_product'):
        self.comparison_type = comparison_type
        self.comparison_function = map_comp_func[comparison_type][comparison_function]

    def calc(
        self,
        model_output: BiEncoderOutput,
        positive_idx_per_question: list,
        temperature: float = 1.0,
        hard_negatice_idx_per_question: list = None
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negatice_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        if self.comparison_type == "representaton_matching":
            q_vectors = model_output.q_pooled
            ctx_vectors = model_output.ctx_pooled
        elif self.comparison_type == "cross_interaction":
            q_vectors = model_output.q_seq
            ctx_vectors = model_output.ctx_seq
        else:
            raise ValueError

        scores = self.comparison_function(q_vectors, ctx_vectors) / temperature

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction='mean'
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        return loss, correct_predictions_count

    def get_comparison_function(self):
        return self.comparison_function


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, question_model: nn.Module, ctx_model: nn.Module, fix_q_encoder: bool = False,
                 fix_ctx_encoder: bool = False):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(sub_model: nn.Module, ids: T, segments: T, attn_mask: T, fix_encoder: bool = False) -> (
    T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(ids, segments, attn_mask)

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(ids, segments, attn_mask)

        return sequence_output, pooled_output, hidden_states

    def forward(self, question_ids: T, question_segments: T, question_attn_mask: T, context_ids: T, ctx_segments: T,
                ctx_attn_mask: T) -> Tuple[T, T, T, T]:

        q_seq, q_pooled, _q_hidden = self.get_representation(self.question_model, question_ids, question_segments,
                                                                  question_attn_mask, self.fix_q_encoder)
        ctx_seq, ctx_pooled, _ctx_hidden = self.get_representation(self.ctx_model, context_ids, ctx_segments,
                                                                        ctx_attn_mask, self.fix_ctx_encoder)

        # return q_pooled_out, ctx_pooled_out
        # return q_pooled_out, ctx_pooled_out, _q_seq, _ctx_seq
        return BiEncoderOutput(
            q_pooled,
            q_seq,
            ctx_pooled,
            ctx_seq
        )

    @classmethod
    def create_biencoder_single(cls,
                                sample: BiEncoderSample,
                                tensorizer: Tensorizer,
                                insert_title: bool = False
                                ) -> Tuple:
        """
        Creates a batch of the biencoder training tuple.
        :param sample: list of data items (from json) to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :return: tuple
        """
        question = sample.query
        positive_ctxs = sample.positive_passages
        positive_ctx_ids = [ctx.cid for ctx in positive_ctxs]
        neg_ctxs = sample.negative_passages
        hard_neg_ctxs = sample.hard_negative_passages
        all_ctxs = positive_ctxs + neg_ctxs + hard_neg_ctxs
        all_ctx_ids, all_ctx_tensors = [], []
        ctx_id_to_source = {}
        for ctx in all_ctxs:
            all_ctx_ids.append(ctx.cid)
            ctx_id_to_source[ctx.cid] = ctx.source
            all_ctx_tensors.append(
                tensorizer.text_to_tensor(
                    text=ctx.text,
                    title=ctx.source if (insert_title and ctx.source) else None
                )
            )
        questions_tensor = tensorizer.text_to_tensor(question).unsqueeze(0)
        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in all_ctx_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return all_ctx_ids, positive_ctx_ids, ctx_id_to_source, BiEncoderSingle(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments
        )

    @classmethod
    def create_biencoder_input(cls,
                               samples: List,
                               tensorizer: Tensorizer,
                               insert_title: bool = False,
                               num_hard_negatives: int = 0,
                               num_other_negatives: int = 0,
                               shuffle: bool = True,
                               shuffle_positives: bool = False,
                               ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of data items (from json) to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only
            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)
            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    text=ctx.text,
                    title=ctx.source if (insert_title and ctx.source) else None
                ) for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx
                    )
                ]
            )

            question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices
        )
