import os
import random   
import datetime
from collections import namedtuple
from tqdm import tqdm, trange
from argparse import ArgumentParser
import pprint
from collections import defaultdict

import yaml
import logging
import nltk

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import CosineAnnealingScheduler, create_lr_scheduler_with_warmup, ProgressBar, PiecewiseLinear

from pytorch_transformers import GPT2Tokenizer, CONFIG_NAME
from pytorch_transformers import BertConfig, BertTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule
from models import GPT2ConditionalLMHeadModel

from utils import dotdict, maybe_mkdir, apply_loss, top_k_top_p_filtering, get_best_indexes
from utils import pad_dataset, get_data_loaders, build_ans_input_from_segments_bert, build_que_input_from_segments
from utils import convert_input_to_text, convert_question_to_text, trim_batch
from constants import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS


logger = logging.getLogger(__file__)


def main(args):
    # Load a pre-defined tokenizer (GPT-2), create config and model
    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")

    gpt_tokenizer = GPT2Tokenizer.from_pretrained(args.qgen_model_path, cache_dir=args.dataset_cache)
    gpt_tokenizer.add_tokens(SPECIAL_TOKENS)
    gpt_tokenizer.sep_token = '<sep>'

    qgen = GPT2ConditionalLMHeadModel.from_pretrained(args.qgen_model_path, cache_dir=args.dataset_cache)
    qgen.resize_token_embeddings(len(gpt_tokenizer))
    qgen.to(args.device)
    qgen_optimizer = AdamW(qgen.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    
    bos, eos, ctx, ans, que, pad = gpt_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    if args.n_gpu > 1:
        qgen = torch.nn.DataParallel(qgen)

    logger.info("Prepare datasets")
    dataloader = get_data_loaders(args, gpt_tokenizer)

    # Define training function
    def update(engine, batch):
        # remove extra pad from batches
        batch = trim_batch(batch, pad)
        qgen.train()

        loss = 0.0
        ###################################
        # MLE training with teacher forcing
        ###################################
        if 'sl' in args.learning:
            input_ids, lm_labels, token_type_ids, _, _, _ = tuple(input_tensor.to(args.device) for input_tensor in batch)
            loss_ce = qgen(input_ids=input_ids, labels=lm_labels, token_type_ids=token_type_ids)[0]
            loss = apply_loss(engine.state.iteration, qgen_optimizer, loss_ce, args)
        return loss.item()
    trainer = Engine(update)
    

    # Add progressbar with loss
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(qgen_optimizer, "lr", [(0, args.learning_rate), (args.n_epochs * len(dataloader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Save checkpoints
    checkpoint_handler = ModelCheckpoint(args.checkpoint, 'checkpoint', save_interval=1, n_saved=6, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(qgen, 'module', qgen)})  # "getattr" take care of distributed encapsulation
    
    # save training config
    torch.save(dict(args), os.path.join(args.checkpoint, 'training_args.bin'))
    getattr(qgen, 'module', qgen).config.to_json_file(os.path.join(args.checkpoint, CONFIG_NAME))
    gpt_tokenizer.save_vocabulary(args.checkpoint)

    trainer.run(dataloader, max_epochs=args.n_epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", default='config/config.yaml', help="The default config file.")
    parser.add_argument("-mq", "--qgen_model_path", type=str, default='gpt2-medium', help='Pretrained model path to local checkpoint for Question Generator')
    parser.add_argument("-e", "--exp_name", type=str, default='qgen', help='The name of experiment')
    args = parser.parse_args()


    # Read config from yaml file.
    config_file = args.config_path
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
        config = dotdict(config)
    
    for k, v in vars(args).items():
        config[k] = v
    
    assert len(config.learning) != 0, "Required atleast one of sl or rl for learning."
    
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.checkpoint = os.path.join(config.checkpoint, "{}-{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), config.exp_name))
    config.n_gpu = torch.cuda.device_count()
    maybe_mkdir(config.checkpoint)

    with open(os.path.join(config.checkpoint, 'config'), 'wt') as f:
        pprint.pprint(config, stream=f)

    # logging is set to INFO
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pprint.pformat(config))
    logger.info("device: {}, n_gpu {}".format(config.device, config.n_gpu))
    
    random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    main(config)
    
