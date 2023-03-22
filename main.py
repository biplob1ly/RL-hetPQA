import os
import logging
import random
import numpy as np
import argparse

import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


from dpr.models.hf_models import get_bert_biencoder_components
from dpr.indexer.faiss_indexers import DenseFlatIndexer
from ranker import Ranker

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

CAND_MAX_LEN = 450
QUERY_MAX_LEN = 150
RESPONSE_MAX_LEN = 200

BOS = "<|BOS|>"
EOS = "<|EOS|>"
UNK = "<|UNK|>"
PAD = "<|PAD|>"
SEP = "<|SEP|>"
PCODE = "<|PCODE|>"
RCODE = "<|RCODE|>"
CAND = "<|CAND|>"
CUSTOMER = "<|CUSTOMER|>"
AGENT = "<|AGENT|>"
EXTRA_TOKENS = [PCODE, RCODE, CAND, CUSTOMER, AGENT]
MAX_LENGTH = CAND_MAX_LEN + QUERY_MAX_LEN + RESPONSE_MAX_LEN

SPECIAL_TOKENS = {
    'bos_token': BOS,
    'eos_token': EOS,
    'unk_token': UNK,
    'pad_token': PAD,
    'sep_token': SEP,
    'additional_special_tokens': EXTRA_TOKENS
}


def get_args():
    parser = argparse.ArgumentParser()
    """
        Common parameters to initialize an encoder-based model
    """
    parser.add_argument("--pretrained_model_cfg", default=None, type=str, help="config name for model initialization")
    parser.add_argument("--encoder_model_type", default=None, type=str,
                        help="model type. One of [hf_bert, pytext_bert, fairseq_roberta]")
    parser.add_argument('--pretrained_file', type=str, help="Some encoders need to be initialized from a file")
    parser.add_argument("--model_file", default=None, type=str,
                        help="Saved bi-encoder checkpoint file to initialize the model")
    parser.add_argument("--projection_dim", default=0, type=int,
                        help="Extra linear layer on top of standard bert/roberta encoder")
    parser.add_argument("--sequence_length", type=int, default=512, help="Max length of the encoder input sequence")

    parser.add_argument("--adam_eps", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="")

    parser.add_argument('--model_name_or_path', type=str,
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=512)

    args = parser.parse_args()
    return args


def set_random_seed(seed):
    """
    Set the random seed everywhere
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def prepare_model(load_model_path=None):
    """
    Initializes the model, vanilla GPT-2 or model from path.
    """
    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    config = AutoConfig.from_pretrained(
        load_model_path if load_model_path else 'gpt2',
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        output_hidden_states=False
    )

    model = AutoModelForCausalLM.from_pretrained(load_model_path if load_model_path else 'gpt2', config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    return tokenizer, model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)

    tensorizer, biencoder, biencoder_optimizer = get_bert_biencoder_components(args)
    indexer = DenseFlatIndexer(biencoder.question_model.get_out_size())
    retriever = Ranker(biencoder.question_model, tensorizer, indexer)

    generator_tokenizer, generator = prepare_model(args.pretrained_file)
    generator_grad_params = filter(lambda p: p.requires_grad, generator.parameters())
    total_params = sum([np.prod(p.size()) for p in generator_grad_params])
    logger.info('Number of parameter = {}'.format(total_params))

    generator_params = list(generator.named_parameters())
    no_decay = ['bias', 'ln']
    generator_grouped_parameters = [
        {'params': [p for n, p in generator_params
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in generator_params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    generator_optimizer = Adam(generator_grouped_parameters, cfg.DPR.SOLVER.OPTIMIZER.BASE_LR, max_grad_norm=1.0)
    if args.fp16:
        logger.info('in fp16, using FusedAdam')
        try:
            import apex
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex "
                "to use distributed and fp16 training.")

        model, optimizer = amp.initialize(generator, generator_optimizer, opt_level=args.fp16_opt_level)
