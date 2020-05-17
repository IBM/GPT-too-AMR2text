import os
import sys
import logging
from itertools import chain
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from dataamr import AMRData
from constants import SPECIAL_TOKENS, MODEL_INPUTS, AMR_SPECIAL_TOKENS

logger = logging.getLogger(__file__)


def tokenize_and_encode(dataset, tokenizer):
    encoded_dataset = list()
    for data_inst in tqdm(dataset):
        tok_amr = tokenizer.convert_tokens_to_ids(
                                            tokenizer.tokenize(data_inst[0]))
        tok_txt = tokenizer.convert_tokens_to_ids(
                                            tokenizer.tokenize(data_inst[1]))
        encoded_dataset.append((tok_amr, tok_txt))
    return encoded_dataset

# Candidate to be removed

# Split list given a list of breaking elements


def split_list(L, S):
    output = list()
    for s in S:
        if s in L:
            idx = L.index(s)
            output.append(L[:idx])
            L = L[idx+1:]
    return output


def pre_process_amr_leftpad(
        amr_graph, text, tokenizer, max_input_length, cap_length,
        with_text=True, with_masking=True, split_sent=False):
    bos, eos, ctx, ans, que, pad, gen = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    padded = []

    amr = [bos] + amr_graph + [gen]
    text = (text + [eos] if with_text else [])

    max_len = int((max_input_length-3)/2)

    if len(amr) > max_len:
        amr = amr[:max_len]
        amr[-1] = gen
    if len(text) > max_len:
        text = text[:max_len]

    combined = list(chain(amr, text))
    len_combined = len(combined)

    if len_combined < max_input_length:
        len_reamining = max_input_length - len_combined
        padded = [pad] * len_reamining

    instance = {}
    instance["input_ids"] = list(chain(padded, amr, text))
    instance["token_type_ids"] = [pad] * len(padded) + [ctx]   \
        * len(amr) + [ans] * len(text)
    instance["attention_mask"] = [0]*len(padded) \
        + [1]*(max_input_length-len(padded))

    if with_masking:
        instance["labels"] = [-1] * (len(padded) + len(amr)) + text
    else:
        instance["labels"] = [-1] * len(padded) + list(chain(amr, text))

    return instance


def pre_process_amr(amr_graph, text, tokenizer, input_len, cap_length,
                    with_text, with_masking=False):

    instance = {}

    bos, eos, ctx, ans, que, pad, gen = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    if not with_text:
        text = []
        end = []
    else:
        end = [eos]

    input_ids = np.full((input_len), pad, dtype=np.int64)
    token_type_ids = np.full((input_len), pad, dtype=np.int64)
    labels = np.full((input_len), -1, dtype=np.int64)
    att_mask = np.full((input_len), 0, dtype=np.int64)

    amr_and_text = [
        bos] + amr_graph[:cap_length] + [gen] + text[:cap_length] + end
    type_list = [bos] + [ctx]*len(amr_graph[:cap_length]) \
        + [gen] + [ans]*len(text[:cap_length]) + end
    att_mask_list = [1] * len(amr_and_text)
    if with_masking:
        label_list = [-1] + [-1]*len(amr_graph[:cap_length]
                                     ) + [-1] + text[:cap_length] + end
    else:
        label_list = amr_and_text

    input_ids[:len(amr_and_text)] = amr_and_text
    token_type_ids[:len(amr_and_text)] = type_list
    labels[:len(amr_and_text)] = label_list
    att_mask[:len(amr_and_text)] = att_mask_list

    instance['input_ids'] = input_ids
    instance['token_type_ids'] = token_type_ids
    instance['labels'] = labels
    instance['attention_mask'] = att_mask

    return instance


def pre_process_amr_datasets_decode(
        encoded_datasets, input_len, cap_length, start_token, delimiter_token,
        eos_token, pad_token):
    """ Pre-process datasets containing lists of tuples(story,
        1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length)
        comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length]
        + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, int(input_len/2)+1), dtype=np.int64)
        for i, (amr_graph, text), in enumerate(dataset):
            amr_and_text = [
                start_token] + amr_graph[:cap_length] + [delimiter_token]
            input_ids[i, :len(amr_and_text)] = amr_and_text

        tensor_datasets.append(torch.tensor(input_ids))

    return tensor_datasets


def load_amr(args):

    dataset_path = args.dataset_path
    train_file = os.path.join(dataset_path, "train.txt")
    dev_file = os.path.join( dataset_path, "dev.txt")
    test_file = os.path.join(dataset_path, "test.txt")
    silver_train_file = "silver.txt"

    amr = AMRData(
        train_file,
        dev_file,
        test_file,
        silver_train_file,
        use_silver_data=args.use_silver_data,
        small=args.small)
    amr.load_data()

    return amr


def update_model(tokenizer, model, amr):
    tokenizer.add_tokens(amr.edges)
    model.resize_token_embeddings(len(tokenizer))


def read_amr(tokenizer, amr, args):

    print("Reading AMR dataset")

    input_format = args.input_format
    small = args.small
    use_silver_data = args.use_silver_data

    dataset_train_silver = None
    logger.info("Number of new tokens added to the vocabulary " +
                str(len(amr.edges)))
    logger.info("train size "+str(len(amr.X_train)))
    logger.info("dev size "+str(len(amr.X_dev)))
    logger.info("train size "+str(len(amr.X_test)))
    if args.use_silver_data:
        logger.info("silver size "+str(len(amr.X_silver_train)))
    logger.info("Encoding dataset...")
    logger.info(" * Prepare")

    if args.tokenized_input:
        Y_train = amr.Y_train_tok
        Y_dev = amr.Y_dev_tok
        Y_test = amr.Y_test_tok
    else:
        Y_train = amr.Y_train
        Y_dev = amr.Y_dev
        Y_test = amr.Y_test
        Y_silver_train = amr.Y_silver_train

    # Using the correct input for the experiment
    if input_format == "linearized_with_attributes" or \
            input_format == "linearized_simple":
        if small:
            # Small only for debugging
            dataset_train = (
                [(" ".join(x),
                  y) for x, y in zip(
                     amr.X_train_simple[: 50],
                     Y_train[: 50])])
            if use_silver_data:
                dataset_train_silver = ([(" ".join(x), y) for x, y in zip(
                    amr.X_silver_train_simple[:50], Y_silver_train[:50])])

        else:
            dataset_train = ([(" ".join(x), y)
                              for x, y in zip(amr.X_train_simple, Y_train)])
            if use_silver_data:
                dataset_train_silver = ([(" ".join(x), y) for x, y in zip(
                    amr.X_silver_train_simple, Y_silver_train)])
        dataset_dev = ([(" ".join(x), y)
                        for x, y in zip(amr.X_dev_simple, Y_dev)])
        dataset_test = ([(" ".join(x), y)
                         for x, y in zip(amr.X_test_simple, Y_test)])
    elif input_format == "only_nodes":
        if small:
            # Small only for debugging
            dataset_train = ([(" ".join(x), y) for x, y in zip(
                amr.X_train_simple_only_nodes[:50], Y_train[:50])])
            if use_silver_data:
                dataset_train_silver = ([(" ".join(x), y) for x, y in zip(
                    amr.X_silver_train_simple_only_nodes[:50],
                    Y_silver_train[:50])])

        else:
            dataset_train = ([(" ".join(x), y) for x, y in zip(
                amr.X_train_simple_only_nodes, Y_train)])
            if use_silver_data:
                dataset_train_silver = ([(" ".join(x), y) for x, y in zip(
                    amr.X_silver_train_simple_only_nodes, Y_silver_train)])

        dataset_dev = ([(" ".join(x), y)
                        for x, y in zip(amr.X_dev_simple_only_nodes, Y_dev)])
        dataset_test = ([(" ".join(x), y) for x, y in
                         zip(amr.X_test_simple_only_nodes, Y_test)])

    elif input_format == "original":
        if small:
            # Small only for debugging
            dataset_train = (
                [(x, y) for x, y in zip(
                     amr.X_train_raw[: 50],
                     Y_train[: 50])])
            if use_silver_data:
                dataset_train_silver = (
                    [(x, y) for x,
                     y
                     in zip(
                         amr.X_silver_train_raw[: 50],
                         Y_silver_train[: 50])])

        else:
            dataset_train = ([(x, y)
                              for x, y in zip(amr.X_train_raw, Y_train)])
            if use_silver_data:

                dataset_train_silver = ([(x, y) for x, y in zip(
                    amr.X_silver_train_raw, Y_silver_train)])

        dataset_dev = ([(x, y) for x, y in zip(amr.X_dev_raw, Y_dev)])
        dataset_test = ([(x, y) for x, y in zip(amr.X_test_raw, Y_test)])
    else:
        logger.info(input_format+" is not a valid input format")
        sys.exit()
    return dataset_train, dataset_dev, dataset_test, dataset_train_silver


def tokenize_amr(tokenizer, args, dataset_train, dataset_dev,
                 dataset_test, dataset_train_silver):
    logger.info(" * Tokenize train set")

    encoded_dataset_train_silver = None
    encoded_dataset_train = tokenize_and_encode(dataset_train, tokenizer)
    if args.use_silver_data:
        logger.info("Encoding silver dataset")
        encoded_dataset_train_silver = tokenize_and_encode(
            dataset_train_silver, tokenizer)
    encoded_dataset_dev = tokenize_and_encode(dataset_dev, tokenizer)
    encoded_dataset_test = tokenize_and_encode(dataset_test, tokenizer)

    total_tokens = len(encoded_dataset_train)
    token_count = 0
    tmp_encoded_dataset_train = list()

    # Remove training examples with bigger size than max_size
    for x_inst, y_inst in encoded_dataset_train:
        if len(x_inst) > args.max_length:
            token_count += 1
        else:
            tmp_encoded_dataset_train.append((x_inst, y_inst))

    if args.exclude_large:
        logger.info(" * [exclude_large] \
Removing the training instances bigger than max_size")
        encoded_dataset_train = tmp_encoded_dataset_train

    print("Training:", round((token_count/total_tokens)*100, 2),
          "% :", token_count, "instances from", total_tokens)

    return encoded_dataset_train, encoded_dataset_dev, encoded_dataset_test, \
        encoded_dataset_train_silver


def preproc_amr(args, tokenizer, encoded_dataset, with_text=True, ):

    datasets = defaultdict(list)

    # Split amr graphs if flag activated and the graph is large
    if args.split_sent:
        logger.info(" * Splitting amr graph for big graphs")
        tmp_encoded_dataset = list()
        special = tokenizer.convert_tokens_to_ids(AMR_SPECIAL_TOKENS)
        multi_sent = special[0]
        join = special[1]
        sents = special[2:]

        for idx, (amr_graph, text) in enumerate(encoded_dataset):
            if len(amr_graph) > 100:

                if multi_sent in amr_graph:
                    amr_graph = amr_graph[4:]
                    amr_graph = amr_graph[:-1]
                    amr_split = split_list(amr_graph, sents)

                    for amr_sub in amr_split:
                        if amr_sub:
                            tmp_encoded_dataset.append([[join]+amr_sub, text])
                else:
                    tmp_encoded_dataset.append([amr_graph, text])
            else:
                tmp_encoded_dataset.append([amr_graph, text])

        encoded_dataset = tmp_encoded_dataset

    logger.info(" * Prepear input vectors")
    for idx, (amr_graph, text) in enumerate(encoded_dataset):
        if idx > args.max_num_examples:
            break

        instance_que = pre_process_amr_leftpad(
            amr_graph,
            text,
            tokenizer,
            args.max_input_length,
            args.max_length,
            with_text=with_text,
            with_masking=args.with_masking,
            split_sent=args.split_sent)
        for input_name, input_array in instance_que.items():
            datasets[input_name].append(input_array)

    tensor_datasets = []
    datasets_padded = datasets
    for input_name in MODEL_INPUTS:
        padded = datasets_padded[input_name]

        tensor_datasets.append(torch.tensor(padded))

    return tensor_datasets
