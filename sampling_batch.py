import os
from shutil import copyfile
import datetime
from collections import namedtuple
from tqdm import tqdm, trange
from argparse import ArgumentParser
from pprint import pprint, pformat

import random
import yaml
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers import GPT2Tokenizer
from models import GPT2ConditionalLMHeadModel

from utils import dotdict, maybe_mkdir, top_k_top_p_filtering
from utils import get_data_loaders
from constants import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS


logger = logging.getLogger(__file__)


def main(args):
    # Load a pre-defined tokenizer (GPT-2), create config and model
    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path, cache_dir=args.dataset_cache)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    tokenizer.sep_token = '<sep>'

    qgen = GPT2ConditionalLMHeadModel.from_pretrained(args.model_path, cache_dir=args.dataset_cache)
    qgen.resize_token_embeddings(len(tokenizer))
    qgen.to(args.device)
    qgen.eval()

    bos, eos, ctx, ans, que, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    if args.n_gpu > 1:
        logger.info("Setting model to DataParallel.")
        qgen = torch.nn.DataParallel(qgen)

    logger.info("Prepare datasets")
    dataloader = get_data_loaders(args, tokenizer, shuffle=False)
    f = open(os.path.join(args.checkpoint, 'questions.txt'), 'w')

    for epoch in range(args.n_epochs):
        for batch in tqdm(dataloader):
            _, _, _, input_ids, _, token_type_ids = tuple(input_tensor.to(args.device) for input_tensor in batch)
            contexts = input_ids
            
            past = None
            all_logits, questions = [], []
            for idx in range(args.max_length):
                with torch.no_grad():
                    logits, past = qgen(input_ids=input_ids, token_type_ids=token_type_ids, past=past)
                # print(input_ids.shape, token_type_ids.shape, past[0].shape, logits.shape)
                logits = logits[:, -1, :] / args.temperature                                            # bs x seq_len x V
                logits = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                probs = F.softmax(logits, dim=-1)                                                       # bs x V
                outputs = torch.multinomial(probs, num_samples=1)                                       # bs x 1
                outputs = torch.where(input_ids[:, -1:] == eos, input_ids[:, -1:], outputs)
                
                # correctly shape inputs for next round
                input_ids = outputs
                token_type_ids = token_type_ids[:, -1:]
                
                # book-keeping
                all_logits.append(logits.unsqueeze(1)) # add dim=1 to store the seq_length
                questions.append(outputs)
                
                # if all the outputs are special tokens
                if (outputs == eos).all():
                    break
            # append an extra <eos> in case max length is reached
            questions.append(torch.zeros_like(outputs).fill_(eos))
        
            logits = torch.cat(all_logits, dim=1)
            questions = torch.cat(questions, dim=1)
            
            for i, question in enumerate(questions):
                question = question.tolist()
                idx = question.index(eos)
                question = tokenizer.decode(question[:idx])
                print(question.replace('\n',' '), file=f)
                f.flush()


def run():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", default='config/config.yaml', help="The default config file.")
    parser.add_argument("-mq", "--model_path", type=str, required=True, help='Pretrained model path to local checkpoint for Question Generator')
    parser.add_argument("-e", "--exp_name", type=str, default='qgen', help='The name of experiment')
    args = parser.parse_args()


    # Read config from yaml file.
    config_file = args.config_path
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
        config = dotdict(config)
    
    for k, v in vars(args).items():
        config[k] = v
    
    config.checkpoint = os.path.join(config.model_path, "sampling", config.exp_name)
    maybe_mkdir(config.checkpoint)
    copyfile(config.config_path, os.path.join(config.checkpoint, "config.yaml"))

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count()
    config.n_gpu = 1

    # logging is set to INFO
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(config))
    logger.info("device: {}, n_gpu {}".format(config.device, config.n_gpu))
    
    random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    main(config)



if __name__ == "__main__":
    run()

