import logging
import random
import glob
import collections
import torch
from torch.utils.data import Dataset
from utils import normalize_question, normalize_attr_passage, read_data_from_json_files

logger = logging.getLogger()

GenSample = collections.namedtuple(
    "GenSample",
    [
        "index",
        "question",
        "contexts",
        "answer"
    ]
)

GenBatch = collections.namedtuple(
    "GenBatch",
    [
        "indices",
        "prompt_ids",
        "prompt_masks",
        "target_ids"
    ]
)


class GenDataset(Dataset):
    def __init__(
            self,
            file: str,
            n_context: int,
            normalize: bool = False,
            flatten_attr: bool = False,
            insert_title: bool = True
    ):
        self.file = file
        self.data = []
        self.data_files = []
        self.n_context = n_context
        self.normalize = normalize
        self.flatten_attr = flatten_attr
        self.insert_title = insert_title
        self.question_prefix = 'question:'
        self.ctx_prefix = 'context:'
        self.title_prefix = 'title:'
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        answer = random.choice(example['answers']) #  + ' </s>'
        ctxs = example['ctxs'][:self.n_context]
        while self.n_context and len(ctxs) < self.n_context:
            ctxs.append(random.choice(example['ctxs']))
        contexts = []
        for ctx in ctxs:
            if self.insert_title and ctx.get("source", None):
                context = self.title_prefix + " " + ctx["source"] + " " + self.ctx_prefix + " " + ctx["text"]
            else:
                context = self.ctx_prefix + " " + ctx["text"]
            contexts.append(context)
        return GenSample(
            index,
            question,
            contexts,
            answer
        )

    def load_data(self):
        if not self.data:
            logger.info("Loading all answer generation data")
            self.data_files = glob.glob(self.file)
            logger.info("Data files: %s", self.data_files)
            self.data = read_data_from_json_files(self.data_files)    # TODO: remove subscript here
            logger.info("Total cleaned data size: %d", len(self.data))

    def get_example(self, index):
        return self.data[index]


class Collator:
    def __init__(self, tokenizer, prompt_max_len=None, answer_max_len=None):
        self.tokenizer = tokenizer
        self.prompt_max_len = min(prompt_max_len, tokenizer.model_max_length) if prompt_max_len else tokenizer.model_max_length
        self.answer_max_len = min(answer_max_len, tokenizer.model_max_length) if answer_max_len else tokenizer.model_max_length

    def encode_targets(self, batch_target):
        enc = self.tokenizer(
            batch_target,
            max_length=self.answer_max_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        target_ids = enc["input_ids"]
        target_mask = enc["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)
        return target_ids

    def encode_prompts(self, batch_prompts):
        prompt_ids, prompt_masks = [], []
        for sample_prompts in batch_prompts:
            enc = self.tokenizer(
                sample_prompts,
                max_length=self.prompt_max_len,
                padding='max_length',
                return_tensors='pt',
                truncation=True
            )
            prompt_ids.append(enc['input_ids'].unsqueeze(0))
            prompt_masks.append(enc['attention_mask'].unsqueeze(0))
        # print(prompt_ids)
        prompt_ids = torch.cat(prompt_ids, dim=0)
        prompt_masks = torch.cat(prompt_masks, dim=0)
        return prompt_ids, prompt_masks.bool()

    def __call__(self, batch):
        indices = torch.tensor([sample.index for sample in batch])
        batch_target = [sample.answer for sample in batch]
        target_ids = self.encode_targets(batch_target)
        batch_prompts = [[f"{sample.question} {context}" for context in sample.contexts] for sample in batch]
        prompt_ids, prompt_masks = self.encode_prompts(batch_prompts)
        return GenBatch(
            indices,
            prompt_ids,
            prompt_masks,
            target_ids
        )
