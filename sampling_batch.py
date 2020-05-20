import os
from shutil import copyfile
from tqdm import tqdm
from argparse import ArgumentParser
from pprint import pformat

import random
import yaml
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from models import GPT2ConditionalLMHeadModel

from utils import dotdict
from top_k_top_p import top_k_top_p_filtering
from utils import get_data_loaders, trim_batch
from constants import SPECIAL_TOKENS

logger = logging.getLogger(__file__)


def main(args):

    debug = False
    # Load a pre-defined tokenizer (GPT-2), create config and model
    logger.info("Prepare tokenizer, pretrained model and optimizer - add \
                special tokens for fine-tuning")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path,
                                              cache_dir=args.dataset_cache)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    tokenizer.sep_token = '<sep>'

    if 'amr' in args.dataset_type:
        qgen = GPT2LMHeadModel.from_pretrained(args.model_path,
                                               cache_dir=args.dataset_cache)
    else:
        qgen = GPT2ConditionalLMHeadModel.\
            from_pretrained(args.model_path, cache_dir=args.dataset_cache)
    qgen.resize_token_embeddings(len(tokenizer))
    qgen.to(args.device)
    qgen.eval()

    logsoftmax = nn.LogSoftmax(dim=0)

    bos, eos, ctx, ans, que, pad, gen = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    if args.n_gpu > 1:
        logger.info("Setting model to DataParallel.")
        qgen = torch.nn.DataParallel(qgen)

    logger.info("Prepare datasets")
    if "amr" in args.dataset_type:
        logger.info("Decoding with AMR dev set")
        dataloader = get_data_loaders(args, tokenizer, qgen,
                                      dataset_name=args.output_data,
                                      shuffle=False)
    else:
        dataloader = get_data_loaders(args, tokenizer, qgen, shuffle=False)

    if 'amr' in args.dataset_type:
        if args.output_data.lower() == "test":
            ref = os.path.join(args.dataset_path, "test.tok.text")
        else:
            ref = os.path.join(args.dataset_path, "dev.tok.text")
        ref = open(ref).readlines()

    logger.info("Decode: "+args.decoder)

    # Output file name
    f = open(os.path.join(args.checkpoint, 'output.txt'), 'w')
    text_outputs = list()

    # beam search variables
    beam_size = 1 if args.beam_size is None else args.beam_size
    output_size = 1 if args.output_size is None else args.output_size
    beam_candidates = args.beam_candidates

    # General variables
    max_length = args.max_input_length

    instance = 0
    for batch in tqdm(dataloader):

        batch = trim_batch(batch, pad)
        _, _, _, _, input_ids, _, token_type_ids, attention_mask = \
            tuple(input_tensor.to(args.device) for input_tensor in batch)

        past = None

        o = 0
        all_probs = torch.zeros(beam_size, 1).to(args.device)
        original_input_len = input_ids.shape[1]
        start = True

        # general variables
        questions = []

        for idx in range(max_length):
            ###################
            # Greedy decoding
            ###################
            if args.decoder == "greedy":
                with torch.no_grad():
                    logits, past = qgen(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        past=past)
                outputs = torch.argmax(logits[0, -1, :])
                outputs = outputs.unsqueeze(0).unsqueeze(0)

            ###################
            # Nucleous Sampling
            ###################
            elif args.decoder == "sampling":
                with torch.no_grad():
                    logits, past = qgen(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        past=past)
                # bs x seq_len x V
                logits = logits[:, -1, :] / args.temperature
                logits = top_k_top_p_filtering(logits, top_k=args.top_k,
                                               top_p=args.top_p)
                # bs x V
                probs = F.softmax(logits, dim=-1)
                # bs x 1
                outputs = torch.multinomial(probs, num_samples=1)
                outputs = torch.where(input_ids[:, -1:] ==
                                      eos, input_ids[:, -1:], outputs)

            ###################
            # BEAM Search
            ###################
            elif args.decoder == "beam":
                # Beam search

                with torch.no_grad():
                    logits = qgen(input_ids)[0]

                out_paths = None
                probs = None

                for k in range(logits.shape[0]):
                    log_p = logsoftmax(logits[k, -1, :])
                    p = log_p+all_probs[k]

                    if start:
                        predicted_top_k = torch.topk(p, beam_size)
                        start = False
                    else:
                        predicted_top_k = torch.topk(p, beam_candidates)

                    p_top_k_tokens = predicted_top_k.indices[:, None]
                    p_top_k_probs = predicted_top_k.values[:, None]

                    # Store paths
                    if out_paths is None:
                        out_paths = torch.cat((input_ids[k].expand(
                            p_top_k_tokens.shape[0],
                            input_ids.shape[1]), p_top_k_tokens), 1)

                    else:
                        out_paths = torch.cat((out_paths, torch.cat((
                            input_ids[k].expand(p_top_k_tokens.shape[0],
                                                input_ids.shape[1]),
                            p_top_k_tokens), 1)), 0)
                    if probs is None:
                        probs = p_top_k_probs
                    else:
                        probs = torch.cat((probs, p_top_k_probs), 0)

                global_top_k = torch.topk(probs, k=beam_size, dim=0)
                input_ids = out_paths[global_top_k.indices[:, 0], :]
                all_probs = global_top_k.values
                o += 1

            else:
                raise Exception('Not valid decoder ' + args.decoder)

            #######################
            # Termination condition
            #######################
            if not args.decoder == 'beam':
                # correctly shape inputs for next round
                input_ids = outputs
                token_type_ids = token_type_ids[:, -1:]

                # if all the outputs are special tokens
                questions.append(outputs)
                if (outputs == eos).all():
                    break
            else:
                outputs = input_ids[:, original_input_len:]

                if ((outputs == eos).sum(dim=1) > 0).all():
                    break

        ################
        # Output to file
        ################
        if args.decoder != 'beam':
            # append an extra <eos> in case max length is reached
            questions.append(torch.zeros_like(outputs).fill_(eos))
            questions = torch.cat(questions, dim=1)

        else:
            questions.append(outputs)
            questions = questions[0]

        for i, question in enumerate(questions):
            question = question.tolist()

            if eos in question:
                idx = question.index(eos)
            else:
                idx = -1

            question = tokenizer.decode(question[:idx])
            if '<generate>' in question:
                question = question.split('<generate>')[1]

            # Print outputs to file and save in text_outputs
            print(question.replace('\n', ' '), file=f)
            f.flush()
            text_outputs.append(question.replace('\n', ' ').lower())

            # Limit number of outputs to output_size
            if i >= output_size-1:
                break

        if 'amr' in args.dataset_type and debug:
            print("GOLD: ", ref[instance])
            print("final: ", text_outputs[instance])

        instance += 1


def run():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", default='config/config.yaml',
                        help="The default config file.")
    # obligatory arguments
    parser.add_argument(
        "--dataset_path",
        help="Input data folder",
        required=True)
    parser.add_argument(
        "--dataset_cache",
        help="Cache for input data folder",
        required=True)
    parser.add_argument(
        "-mq", "--model_path", type=str, required=True,
        help='Pretrained model path to local checkpoint')
    parser.add_argument(
        "-e", "--exp_name", type=str, default='qgen',
        help='The name of experiment')
    args = parser.parse_args()

    # Read config from yaml file.
    config_file = args.config_path
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
        config = dotdict(config)

    # overload with command line arguments
    for k, v in vars(args).items():
        config[k] = v

    config.checkpoint = os.path.join(config.model_path,
                                     "sampling", config.exp_name)
    os.makedirs(config.checkpoint, exist_ok=True)
    copyfile(config.config_path, os.path.join(config.checkpoint,
                                              "config.yaml"))

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
