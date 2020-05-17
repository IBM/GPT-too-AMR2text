# AMR2text from Pretrained Transformer


## Install with pip

Assuming that you have `Python3` and `virtualenv` available on your command
line you can use

    bash install_pip.sh

This will install the needed modules as well download the `stog` preprocessing
tools and make the acessible in a virtualenv called `venv`. To activate the
virtualenv do

. venv/bin/activate

## Install with conda in PowerPC

There is also a script for installation PowerPCs available. For this run

    bash install_conda_ppc.sh

the installation will be carried out on a conda environment (name will be shown
at the end of the install). To activate the environment

    conda activate <environment name>

this assumes `CONDA_PREFIX` is set and pointing to you PowerPC conda
installation

## Train / Generate

To train a model

```bash
python train.py --config_path config/amr_raw.yaml
```

To sample sentences from a trained model

```bash
python sampling_batch.py --config_path config/config-sampling.yaml --model_path [MODEL_PATH]
```

Experimental code for the sentence splitting

```bash
python generate.py --config_path config/config-sampling.yaml --model_path [MODEL_PATH]
```

Relevant model configs are

## Training / Decoding configs

- config/amr_raw.yaml
- config/amr_raw_small.yaml
- config/amr_raw_large.yaml

## Decoding

- config/amr-beam-decode.yaml
- config/amr-sample-decode.yaml
- config/amr-raw-decoding.yaml

## Non-autoencoding experiments

- config/amr_raw_masked_decode.yaml
- config/amr_raw_masked_large.yaml
- config/amr_raw_masked_small.yaml

# A note on tokenization

