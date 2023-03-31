import logging
import glob
import json
import re
import math
import random
import collections
from typing import List, Iterator, Callable
from flatten_json import flatten

import torch
from torch.utils.data import Dataset
from torch import Tensor as T

logger = logging.getLogger()
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["cid", "text", "title"])


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question


def normalize_passage_1(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text


def remove_key(json_dt, key='normalized_value'):
    if isinstance(json_dt, dict):
        return {k.strip(): remove_key(v) for k, v in json_dt.items() if k != key}
    else:
        return json_dt.strip() if isinstance(json_dt, str) else json_dt


def normalize_attr_passage(s, do_flatten=False):
    # We'll start by removing any extraneous white spaces
    s = re.sub('(\d+\.)}', '\g<1>0}', s)
    s = re.sub(r'\\"', ' inches', s)
    if ';' in s:
        prefix_len = s.index(':')+1
        s = s[:prefix_len] + '[' + re.sub(';', ',', s[prefix_len:]) + ']'
    try:
        t = re.sub('(\w+):', '"\g<1>":', s)
        json_dt = json.loads('{' + t + '}')
        json_dt = remove_key(json_dt)
        out = json.dumps(flatten(json_dt) if do_flatten else json_dt)
    except:
        s = re.sub('\"', '', s)
        s = re.sub(':([^[,}{]+)(,|})', ':"\g<1>"\g<2>', s)
        s = re.sub('([^}{\s\d"]+):', '"\g<1>":', s)
        try:
            json_dt = json.loads('{' + s + '}')
            json_dt = remove_key(json_dt)
            out = json.dumps(flatten(json_dt) if do_flatten else json_dt)
        except:
            out = s
    out = re.sub('["\[\]}{]', '', out)
    out = re.sub(':', ': ', out)
    out = re.sub('\s+', ' ', out)
    return out


def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % path)
            data = json.load(f)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


class BiEncoderSample:
    qid: str
    query: str
    source: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class QADataset(Dataset):
    def __init__(
        self,
        special_token: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
        encoder_type: str = None,
    ):
        self.special_token = special_token
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix
        self.data = []

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        raise NotImplementedError

    def calc_total_data_len(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix
        return query


class JsonQADataset(QADataset):
    def __init__(
        self,
        file: str,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        flatten_attr: bool = False,
        query_special_suffix: str = None,
        # tmp: for cc-net results only
        exclude_gold: bool = False,
    ):
        super().__init__(
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        self.file = file
        self.data_files = []
        self.normalize = normalize
        self.flatten_attr = flatten_attr
        self.exclude_gold = exclude_gold

    def calc_total_data_len(self):
        if not self.data:
            self._load_all_data()
        return len(self.data)

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        if not self.data:
            self._load_all_data()
        if start_pos >= 0 and end_pos >= 0:
            logger.info("Selecting subset range from %d to %d", start_pos, end_pos)
            self.data = self.data[start_pos:end_pos]

    def _load_all_data(self):
        logger.info("Loading all data")
        self.data_files = glob.glob(self.file)
        logger.info("Data files: %s", self.data_files)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]      # TODO: remove subscript
        logger.info("Total cleaned data size: %d", len(self.data))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.qid = json_sample["qid"]
        r.query = self._process_query(json_sample["question"])

        positive_ctxs = json_sample["positive_ctxs"]
        r.source = positive_ctxs[0].get("source", None)
        negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []

        def create_passage(ctx: dict):
            if self.normalize:
                if ctx.get("source", None) == "attribute":
                    ctx["text"] = normalize_attr_passage(ctx["text"], self.flatten_attr)
            return BiEncoderPassage(
                ctx.get("cid", None),
                ctx["text"],
                ctx.get("title", None)
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        return r


class SharedDataIterator:
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """
    def __init__(
        self,
        dataset: QADataset,
        shard_id: int = 0,
        num_shards: int = 1,
        batch_size: int = 1,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        strict_batch_size: bool = False,
    ):
        self.dataset = dataset

        logger.info("Calculating shard positions")
        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)
        total_size = dataset.calc_total_data_len()
        samples_per_shard = math.ceil(total_size / self.shards_num)
        self.shard_start_idx = self.shard_id * samples_per_shard
        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)
        logger.debug(
            'samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d',
            samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations)

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size

    def total_data_len(self) -> int:
        return len(self.dataset)

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for sample in self.dataset:
            visitor_func(sample)

    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices[self.shard_start_idx : self.shard_end_idx]
        return shard_indices

    def iterate_data(self, epoch: int = 0) -> Iterator[List]:
        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        max_iterations = self.max_iterations - self.iteration
        shard_indices = self.get_shard_indices(epoch)

        for i in range(self.iteration * self.batch_size, len(shard_indices), self.batch_size):
            items_idxs = shard_indices[i : i + self.batch_size]
            if self.strict_batch_size and len(items_idxs) < self.batch_size:
                logger.debug("Extending batch to max size")
                items_idxs.extend(shard_indices[0 : self.batch_size - len(items)])
            self.iteration += 1
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        while self.iteration < max_iterations:
            logger.debug("Fulfilling non complete shard=".format(self.shard_id))
            self.iteration += 1
            items_idxs = shard_indices[0 : self.batch_size]
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        logger.info("Finished iterating, iteration={}, shard={}".format(self.iteration, self.shard_id))
        # reset the iteration status
        self.iteration = 0