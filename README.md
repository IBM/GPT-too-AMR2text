# Question Generation

This model generates questions from given passages and answer phrases.


## Install

To install the needed modules

```
    pip install -r requirements.txt
```

## Data

Download the du-split of SQuAD

```
    git clone https://github.com/tomhosking/squad-du-split DATA
```

## Train 

```
    python train.py --config_path config/config.yaml
```

## Sample questions from a trained model

In batch

```
	python sampling_batch.py --config_path config/config-sampling.yaml --model_path [MODEL_PATH]
```

