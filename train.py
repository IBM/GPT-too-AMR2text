import os
import random
from argparse import ArgumentParser
import pprint

import yaml
import logging

import torch

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, CONFIG_NAME
from pytorch_transformers import AdamW
from models import GPT2ConditionalLMHeadModel

from utils import dotdict, apply_loss
from utils import get_data_loaders
from utils import trim_batch
from constants import SPECIAL_TOKENS, AMR_SPECIAL_TOKENS


logger = logging.getLogger(__file__)


def main(args):
    # Load a pre-defined tokenizer (GPT-2), create config and model
    logger.info("Prepare tokenizer, pretrained model and optimizer - \
                add special tokens for fine-tuning")

    gpt_tokenizer = GPT2Tokenizer.from_pretrained(
        args.qgen_model_path, cache_dir=args.dataset_cache)
    gpt_tokenizer.sep_token = '<sep>'

    gpt_tokenizer.add_tokens(SPECIAL_TOKENS)
    gpt_tokenizer.add_tokens(AMR_SPECIAL_TOKENS)
    if 'amr' in args.dataset_type:
        qgen = GPT2LMHeadModel.from_pretrained(
            args.qgen_model_path, cache_dir=args.dataset_cache)
    else:
        qgen = GPT2ConditionalLMHeadModel.from_pretrained(
            args.qgen_model_path, cache_dir=args.dataset_cache)

    logger.info("Adjust model size to new tokens")
    qgen.resize_token_embeddings(len(gpt_tokenizer))
    logger.info("Set model to GPU usage")
    qgen.to(args.device)
    logger.info("Set up optimizer")
    qgen_optimizer = AdamW(
        qgen.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon)

    bos, eos, ctx, ans, que, pad, gen = \
        gpt_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    # if args.n_gpu > 1:
    if False:
        logger.info("More then 1 GPU for training")
        qgen = torch.nn.DataParallel(qgen)

    logger.info("Prepare datasets")
    if args.use_silver_data:
        data_type = 'Silver'
    else:
        data_type = 'Train'

    dataloader = get_data_loaders(
        args, gpt_tokenizer, qgen, dataset_name=data_type)

    # Define training function
    def update(engine, batch):

        # remove extra pad from batches
        batch = trim_batch(batch, pad)

        qgen.train()

        loss = torch.tensor([0.0])
        ###################################
        # MLE training with teacher forcing
        ###################################
        if 'sl' in args.learning:
            input_ids, lm_labels, token_type_ids, attention_mask, _, _, _, _ =\
                tuple(input_tensor.to(args.device) for input_tensor in batch)
            loss_ce = qgen(
                input_ids=input_ids,
                labels=lm_labels,
                token_type_ids=token_type_ids)[0]
            loss = apply_loss(
                engine.state.iteration,
                qgen_optimizer,
                loss_ce,
                args)
        return loss.item()
    trainer = Engine(update)

    # Add progressbar with loss
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(
        qgen_optimizer, "lr", [
            (0, args.learning_rate), (args.n_epochs * len(dataloader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Save checkpoints
    checkpoint_handler = ModelCheckpoint(
        args.checkpoint,
        'checkpoint',
        save_interval=1,
        n_saved=20,
        require_empty=False)

    # "getattr" take care of distributed encapsulation
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(
                qgen, 'module', qgen)})

    # save training config
    torch.save(dict(args), os.path.join(args.checkpoint, 'training_args.bin'))
    getattr(
        qgen,
        'module',
        qgen).config.to_json_file(
        os.path.join(
            args.checkpoint,
            CONFIG_NAME))
    gpt_tokenizer.save_vocabulary(args.checkpoint)

    trainer.run(dataloader, max_epochs=args.n_epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
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
        "--checkpoint",
        help="Where model will be stored",
        required=True)
    # rest
    parser.add_argument(
        "-c",
        "--config_path",
        default='config/config.yaml',
        help="The default config file.")
    parser.add_argument(
        "-mq",
        "--qgen_model_path",
        type=str,
        default='gpt2-medium',
        help='Pretrained model path to local checkpoint \
        for Question Generator')
    parser.add_argument(
        "-ma",
        "--qa_model_path",
        type=str,
        default='bert-base-uncased',
        help='Pretrained model path to local checkpoint \
        for Question Answering')
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        default='qgen',
        help='The name of experiment')
    args = parser.parse_args()

    # Read config from yaml file
    config_file = args.config_path
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
        config = dotdict(config)

    # overload with command line arguments
    for k, v in vars(args).items():
        config[k] = v

    assert len(config.learning) != 0, "Required atleast sl for learning."

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.checkpoint = os.path.join(
        config.checkpoint,
        "{}".format(config.exp_name)
    )
    config.n_gpu = torch.cuda.device_count()

    # Make folder for checkpoint
    os.makedirs(config.checkpoint, exist_ok=True)
    # Write config with overloads
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
