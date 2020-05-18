import logging
import os
import re
import string
from itertools import chain
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from constants import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
from amr_utils import read_amr, load_amr, update_model, tokenize_amr, \
    preproc_amr

from constants import SPECIAL_TOKENS_BERT

logger = logging.getLogger(__file__)
eps = np.finfo(np.float32).eps.item()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_sub_list(sl, l):
    "find starting and ending indices of sublist in list"
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            return ind, ind+sll-1
    return False


def get_best_indexes(logits, n_best_size=1):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1],
                             reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def convert_input_to_text(input_ids, tokenizer, decode=True):
    bos, eos, ctx, ans, que, pad, _ = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    a_idx = (input_ids == ans).nonzero().item()
    c_idx = (input_ids == ctx).nonzero().item()
    q_idx = (input_ids == que).nonzero().item()
    eos_idx = (input_ids == eos).nonzero().item()
    c = input_ids[c_idx+1:a_idx].tolist()
    a = input_ids[a_idx+1:q_idx].tolist()
    q = input_ids[q_idx+1:eos_idx].tolist()

    triplet = [c, a, q]
    if decode:
        return [tokenizer.decode(element) for element in triplet]
    return triplet


def convert_question_to_text(question, tokenizer, decode=True):
    eos = tokenizer.convert_tokens_to_ids("<eos>")
    eos_idx = (question == eos).nonzero()[0].item()
    q = question[:eos_idx].tolist()
    if decode:
        return tokenizer.decode(q)
    return q


def trim_pad(input_ids, lm_labels, token_type_ids, attention_mask, pad):
    min_idx = (input_ids != pad).nonzero()[:, 1].min()

    return [input_ids[:, min_idx:], lm_labels[:, min_idx:],
            token_type_ids[:, min_idx:], attention_mask[:, min_idx:]]


def trim_pad_max(input_ids, lm_labels, token_type_ids, attention_mask, pad):
    min_idx = (input_ids != pad).nonzero()[:, 1].max()

    return [input_ids[:, min_idx:], lm_labels[:, min_idx:],
            token_type_ids[:, min_idx:], attention_mask[:, min_idx:]]


def trim_batch(batch, pad):
    input_ids, lm_labels, token_type_ids, attention_mask, partial_input_ids,\
        partial_lm_labels, partial_token_type_ids,\
        partial_attention_mask = batch
    return trim_pad(input_ids, lm_labels, token_type_ids, attention_mask, pad)\
        + trim_pad(partial_input_ids, partial_lm_labels,
                   partial_token_type_ids, partial_attention_mask, pad)


def apply_loss(idx, optimizer, loss, args, retain_graph=False):
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    loss /= args.gradient_accumulation_steps
    loss.backward(retain_graph=retain_graph)
    if args.max_norm is not None:
        params = optimizer.param_groups[0]['params']
        torch.nn.utils.clip_grad_norm_(params, args.max_norm)
    if idx % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    return loss


def pad_dataset(dataset, padding=0, max_input_length=float('inf')):
    """ Pad the dataset. This could be optimized by defining a
    Dataset class and padd only batches but this is simpler. """
    logger.info("Pad inputs and convert to Tensor")
    max_l = min(max(len(x) for x in dataset["input_ids"]), max_input_length)
    for name in PADDED_INPUTS:
        dataset[name] = [[padding if name != "labels" else -1]
                         * (max_l - len(x)) + x[:max_l] for x in dataset[name]]
    return dataset


def build_que_input_from_segments(context, answer, question, tokenizer,
                                  max_input_length=1000, with_eos=True,
                                  with_labels=True):
    """ Build a sequence of input from 3 segments:
        context, answer, question """
    bos, eos, ctx, ans, que, pad, gen =  \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    padded = []
    context = [bos, ctx] + context
    answer = [ans] + answer
    question = [que] + question + ([eos] if with_eos else [])

    combined = list(chain(context, answer, question))
    len_combined = len(combined)

    if len_combined > max_input_length:
        len_context = max_input_length - len(answer) - len(question)
        context = context[:len_context]
    elif len_combined < max_input_length:
        len_reamining = max_input_length - len_combined
        padded = [pad] * len_reamining

    instance = {}
    instance["input_ids"] = list(chain(padded, context, answer, question))
    instance["token_type_ids"] = [pad] * len(padded) + [ctx] * len(context)\
        + [ans] * len(answer) + [que] * len(question)
    if with_labels:
        instance["labels"] = [-1] * (len(padded) + len(context) + len(answer)
                                     + 1) + question[1:]
    return instance


def build_ans_input_from_segments_bert(context, answer, question, tokenizer,
                                       max_input_length):
    """ Build a sequence of input from 3 segments:
        context, question, answer """
    cls, sep, pad, unused0 =  \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS_BERT)

    padded = []
    question = [cls] + question + [sep]
    context = context + [sep]

    combined = list(chain(context, question))
    len_combined = len(combined)

    if len_combined >= max_input_length:
        len_context = max_input_length - len(question)
        context = context[:len_context]
    else:
        remaining_len = max_input_length - len(question) - len(context)
        padded = [pad] * remaining_len

    instance = {}
    instance["input_ids"] = list(chain(question, context, padded))
    instance["token_type_ids"] = [pad] * len(question) + [unused0] \
        * len(context) + [pad] * len(padded)
    instance["attention_mask"] = [unused0] * len(question) + [unused0] \
        * len(context) + [pad] * len(padded)

    pair = find_sub_list(answer, instance["input_ids"])
    # TODO: Find why pair fails?
    if not pair:
        return None
    instance["start_positions"], instance["end_positions"] = pair
    return instance


def build_ans_input_from_segments(context, answer, question, tokenizer,
                                  with_eos=True):
    """ Build a sequence of input from 3 segments:
        context, question, answer """
    bos, eos, ctx, ans, que, _, _ = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    context = [bos, ctx] + context
    question = [que] + question
    answer = [ans] + answer + ([eos] if with_eos else [])

    instance = {}
    instance["input_ids"] = list(chain(context, question, answer))
    instance["token_type_ids"] = [ctx] * len(context) + [que]\
        * len(question) + [ans] * len(answer)
    instance["labels"] = [-1] * len(context) + [-1] \
        * len(question) + [-1] + answer[1:]
    return instance


# ----------------------------------------------------------------------------------
# Dataset utils
# ----------------------------------------------------------------------------------


def read_cardie(dataset_path):
    dataset_path_context = dataset_path + '.context'
    dataset_path_answers = dataset_path + '.answers'
    dataset_path_questions = dataset_path + '.questions'

    contexts, questions, answers = [], [], []
    with open(dataset_path_context, "r", encoding="utf-8",
              errors='ignore') as f:
        contexts_dataset = f.readlines()
    with open(dataset_path_answers, "r", encoding="utf-8",
              errors='ignore') as f:
        answers_dataset = f.readlines()
    with open(dataset_path_questions, "r", encoding="utf-8",
              errors='ignore') as f:
        questions_dataset = f.readlines()
    for idx, (context, answer, question) in enumerate(zip(contexts_dataset,
                                                          answers_dataset,
                                                          questions_dataset)):
        context = context.strip()
        answer = answer.strip().split('\t')[0]
        question = question.strip()
        contexts.append(context)
        answers.append(answer)
        questions.append(question)
        if idx >= 86633*4:
            break
    return [contexts, answers, questions]


def read_drop(dataset_path):
    from allennlp.data.dataset_readers import DropReader
    reader = DropReader(instance_format="bert")
    contexts, questions, answers = [], [], []
    for idx, instance in enumerate(reader.read(dataset_path)):
        try:
            answers.append(instance['metadata']['answer_texts'][0])
        except:
            logger.info(f"Skipping {idx} while reading drop.")
            continue
        contexts.append(instance['metadata']['original_passage'])
        questions.append(instance['metadata']['original_question'])
    return [contexts, answers, questions]


def read_natural_questions(dataset_path):
    from mrqa_reader import MRQAReader
    reader = MRQAReader()
    contexts, questions, answers = [], [], []
    for idx, instance in enumerate(reader.read(dataset_path)):
        instance = instance['metadata']
        has_answer = instance["has_answer"]
        if not has_answer:
            logger.info(f"Skipping {idx} without answer.")
            continue

        answer = instance["answer_texts_list"]
        if len(answer):
            answers.append(answer[0])
        else:
            logger.info(f"Skipping {idx} while reading natural questions.")
            continue

        context = instance["original_passage"].split("[SEP]")[1]
        context = re.sub(r'<.*?>', '', context)
        contexts.append(context)

        question = " ".join(instance["question_tokens"]) + "?"
        questions.append(question)

    return [contexts, answers, questions]


def read_squad(dataset_path):
    from mrqa_reader import MRQAReader
    reader = MRQAReader()
    contexts, questions, answers = [], [], []
    for idx, instance in enumerate(reader.read(dataset_path)):
        instance = instance['metadata']
        answer = instance["answer_texts_list"]
        has_answer = instance["has_answer"]
        if not has_answer:
            logger.info(f"Using None for question {idx}.")
            answer = ['None']

        if len(answer):
            answers.append(answer[0])
        else:
            logger.info(f"Skipping {idx} while reading natural questions.")
            continue

        context = instance["original_passage"].split("[SEP]")[1]
        # context = re.sub(r'<.*?>', '', context)
        contexts.append(context)

        question = " ".join(instance["question_tokens"])
        questions.append(question)

    return [contexts, answers, questions]


def read_squad2(dataset_path):
    from allennlp.data.dataset_readers import SquadReader
    reader = SquadReader()
    contexts, questions, answers = [], [], []
    for idx, instance in enumerate(reader.read(dataset_path)):
        try:
            answers.append(instance['metadata']['answer_texts'][0])
        except:
            # using None for non-answerable questions
            answers.append('None')
        contexts.append(" ".join(instance['metadata']['passage_tokens']))
        questions.append(" ".join(instance['metadata']['question_tokens']))
    return [contexts, answers, questions]


def get_dataset(tokenizer, dataset_path, dataset_cache_dir=None,
                dataset_type=None, re_tokenize=False):
    """ Get dataset from path """
    # dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = \
        os.path.join(dataset_cache_dir, os.path.basename(dataset_path))
    logger.info("Check dataset cache at %s", dataset_cache)
    if dataset_cache and os.path.isfile(dataset_cache) and not re_tokenize:
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Get dataset from %s", dataset_path)

        if 'squad' in dataset_type:
            dataset = read_squad2(dataset_path)
        elif 'dusquad' in dataset_type:
            dataset = read_squad2(dataset_path)
        elif 'squad2' in dataset_type:
            dataset = read_squad2(dataset_path)
        elif 'drop' in dataset_type:
            dataset = read_drop(dataset_path)
        elif 'natural_questions' in dataset_type:
            dataset = read_natural_questions(dataset_path)
        elif 'cardie' in dataset_type:
            dataset = read_cardie(dataset_path)
        else:
            NotImplementedError

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.encode(obj)
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        if dataset_cache:
            os.makedirs(dataset_cache_dir, exist_ok=True)
            logger.info(
                f'Saving tokenized dataset to cache at {dataset_cache}')
            torch.save(dataset, dataset_cache)

    return dataset


def get_datasets(args, tokenizer, with_question=False):
    # Download and tokenize training dataset
    dataset = []
    if 'drop' in args.dataset_type:
        dataset_path = os.path.join(args.dataset_path,
                                    f'drop_dataset_{args.traintype}.json')
        datasubset = get_dataset(tokenizer, dataset_path, args.dataset_cache,
                                 args.dataset_type, args.re_tokenize)
        dataset.append(datasubset)
    if 'natural_questions' in args.dataset_type:
        dataset_path = \
            os.path.join(args.dataset_path,
                         f'NaturalQuestionsShort.{args.traintype}.jsonl.gz')
        datasubset = get_dataset(tokenizer, dataset_path, args.dataset_cache,
                                 args.dataset_type, args.re_tokenize)
        dataset.append(datasubset)
    if 'squad' in args.dataset_type:
        dataset_path = os.path.join(args.dataset_path,
                                    f'SQuAD.{args.traintype}.json')
        datasubset = get_dataset(tokenizer, dataset_path, args.dataset_cache,
                                 args.dataset_type, args.re_tokenize)
        dataset.append(datasubset)
    if 'dusquad' in args.dataset_type:
        dataset_path = os.path.join(args.dataset_path,
                                    f'{args.traintype}-v1.1.json')
        datasubset = get_dataset(tokenizer, dataset_path, args.dataset_cache,
                                 args.dataset_type, args.re_tokenize)
        dataset.append(datasubset)
    if 'squad2' in args.dataset_type:
        dataset_path = os.path.join(args.dataset_path,
                                    f'SQuAD2.{args.traintype}.json')
        datasubset = get_dataset(tokenizer, dataset_path, args.dataset_cache,
                                 args.dataset_type, args.re_tokenize)
        dataset.append(datasubset)
    if 'cardie' in args.dataset_type:
        dataset_path = os.path.join(args.dataset_path,
                                    f'du-cardie.ca.{args.traintype}')
        datasubset = get_dataset(tokenizer, dataset_path, args.dataset_cache,
                                 args.dataset_type, args.re_tokenize)
        dataset.append(datasubset)

    datasets = defaultdict(list)
    for datasubset in dataset:
        for idx, (context, answer, question) in enumerate(zip(*datasubset)):
            if idx > args.max_num_examples:
                break

            if with_question:
                with_eos = True
            else:
                # for autoregressive question: []
                # no <eos> string at the end
                question = []
                with_eos = False
            instance_que = build_que_input_from_segments(
                context, answer, question, tokenizer,
                max_input_length=args.max_input_length, with_eos=with_eos)
            for input_name, input_array in instance_que.items():
                datasets[input_name].append(input_array)

    tensor_datasets = []
    datasets_padded = datasets
    for input_name in MODEL_INPUTS:
        tensor_datasets.append(torch.tensor(datasets_padded[input_name]))
    return tensor_datasets


def get_data_loaders(args, tokenizer, model, dataset_name="Train",
                     shuffle=True):
    if 'amr' in args.dataset_type:
        logger.info("The data set is AMR")

        file_name_train = f'amr_{args.input_format}_train'
        file_name_dev = f'amr_{args.input_format}_dev'
        file_name_test = f'amr_{args.input_format}_test'
        file_name_silver = 'amr_silver_data.txt'

        dataset_cache_silver = os.path.join(args.dataset_cache,
                                            file_name_silver)
        dataset_cache = os.path.join(args.dataset_cache, file_name_train)
        dataset_cache_dev = os.path.join(args.dataset_cache, file_name_dev)
        dataset_cache_test = os.path.join(args.dataset_cache, file_name_test)

        logger.info("Loding AMR")
        amr = load_amr(args)
        update_model(tokenizer, model, amr)

        if dataset_cache and os.path.isfile(dataset_cache) \
                and not args.re_tokenize:
            logger.info("Loding tokenized AMR from cache")
            logger.info(
                f'Load tokenized dataset from cache at {dataset_cache}'
            )
            encoded_dataset_train = torch.load(dataset_cache)
            encoded_dataset_dev = torch.load(dataset_cache_dev)
            encoded_dataset_test = torch.load(dataset_cache_test)
            if args.use_silver_data:
                encoded_dataset_silver = torch.load(dataset_cache_silver)
            else:
                encoded_dataset_silver = None
        else:
            logger.info("Tokenize AMR")
            datasets = read_amr(tokenizer, amr, args)
            encoded_dataset_train, encoded_dataset_dev, encoded_dataset_test,\
                encoded_dataset_silver = \
                tokenize_amr(tokenizer, args, *datasets)
            if dataset_cache:
                os.makedirs(args.dataset_cache, exist_ok=True)
                logger.info(
                    f'Saving tokenized dataset to cache at {dataset_cache}')
                torch.save(encoded_dataset_train, dataset_cache)
                torch.save(encoded_dataset_dev, dataset_cache_dev)
                torch.save(encoded_dataset_test, dataset_cache_test)
                if args.use_silver_data:
                    torch.save(encoded_dataset_silver, dataset_cache_silver)

        logger.info("Using " + dataset_name)
        if dataset_name.lower() == "train":
            encoded_dataset = encoded_dataset_train
        elif dataset_name.lower() == "dev":
            encoded_dataset = encoded_dataset_dev
        elif dataset_name.lower() == "test":
            encoded_dataset = encoded_dataset_test
        elif dataset_name.lower() == "silver":
            print("Encoding data set we are going to use is Silver")
            print(len(encoded_dataset_silver))
            encoded_dataset = encoded_dataset_silver

        tensor_datasets = \
            preproc_amr(args, tokenizer, encoded_dataset, with_text=True) \
            + preproc_amr(args, tokenizer, encoded_dataset, with_text=False)

        logger.info("Build train and validation dataloaders")
        dataset = TensorDataset(*tensor_datasets)
        train_loader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=shuffle, pin_memory=True)

    else:
        tensor_datasets = get_datasets(args, tokenizer, with_question=True)\
                        + get_datasets(args, tokenizer, with_question=False)

        logger.info("Build train and validation dataloaders")
        train_dataset = TensorDataset(*tensor_datasets)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=shuffle, pin_memory=True)

        logger.info("Train dataset (Batch, Seq length): {}"
                    .format(train_dataset.tensors[0].shape))
    return train_loader
