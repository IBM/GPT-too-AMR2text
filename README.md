# AMR2text from Pre-trained Transformer

Code to reproduce 

**GPT-too: A Language-Model-First Approach for AMR-to-Text-Generation**

training GPT-2 models to attain state-of-the art AMR-to-text generation. 

## Install

Code was tested for Python 3.6 on Linux machines both for x86 and PowerPC
architectures.

All scripts source an environment script that can be used to activate virtual
environments or setup system variables specific to the project. If you do not
need any of this just set it to an empty file

    touch set_environment.sh

examples of `set_environment.sh` for a pip install with virtual environment

```bash
[ ! -d venv ] && virtualenv venv
. venv/bin/activate
```

and with a conda environment

```bash
eval "$(/path/to/my/ppc64/miniconda3/bin/conda shell.bash hook)"
[ ! -d cenv_ppc ] && conda create -y -p ./cenv_ppc
conda activate ./cenv_ppc
```

the installers also source `set_environment.sh` and download all the necessary
modules and tools, available installers

```bash
bash scripts/amr2txt/install_x86_with_pip.sh
bash scripts/amr2txt/install_x86_with_conda.sh
bash scripts/amr2txt/install_ppc_with_conda.sh
```

Full pip install for x86, conda install for x86 and conda install for Power PC
respectively. You can also run the commands on those scripts one by one for a
manual install.

## Run full train and test

The following script trains a GPT-2 medium version of the AMR2txt system. It
also selects bes model in dev according to BLEU and runs beam decoding on it.

```bash
bash scripts/amr2txt/experiment.sh scripts/amr2txt/configs/acl2020.sh
```

a similar config is also available for GPT-2 large
