set -e errexit
set -e pipefail
# Use this script to activate your PPC conda and/or virtual environment
. set_environment.sh
set -e nounset 

# install python version to be used
conda install python=3.6.9 -y 

# install modules we can from powerai
conda env update -f scripts/amr2txt/conda_environment.yml

# install the rest with pip
pip install -r scripts/amr2txt/requirements_conda.txt

# install transformers v2.1.1 minus unnecesary modules
rm -Rf transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 1.2.0
sed 's@^sentencepiece@#sentencepiece@' -i requirements.txt
sed "s@'sentencepiece',@@" -i setup.py
#sed "s@'sacremoses'@@" -i setup.py
pip install --editable .
cd ..

# AMR dependencies

# stog
rm -Rf stog
git clone https://github.com/sheng-z/stog.git
cd stog
# get an specific version to avoid new changes
git checkout f541f004d3c016ae35099b708979b1bca15a13bc
# stog still uses old HF transformers, we need to replace the calls
sed 's@from pytorch_pretrained_bert.tokenization import BertTokenizer@from transformers.pytorch_transformers.tokenization_bert import BertTokenizer@' -i stog/data/tokenizers/bert_tokenizer.py
# stog does not have an installer we create one with only the depencencies
# we need
echo "
from setuptools import setup
setup(
    name='stog',
    description='AMR Parsing as Sequence-to-Graph Transduction',
    py_modules=['stog'],
    install_requires=['penman==0.6.2', 'networkx', 'overrides', 'ftfy'],
)
" > setup.py
# editable just in case
python setup.py develop
cd ..

# Dowload meteor
[ ! -f meteor-1.5.tar.gz ] && wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xf meteor-1.5.tar.gz

# Download top-k nucleous sampler
[ ! -f "top-k-top-p.py" ] && wget https://gist.githubusercontent.com/thomwolf/1a5a29f6962089e871b94cbd09daf317/raw/91eea55d0353eaa18f7935e900d5e8d5e503e89c/top-k-top-p.py
