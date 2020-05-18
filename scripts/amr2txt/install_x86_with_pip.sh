set -e errexit
set -e pipefail
# Use this script to activate your PPC conda and/or virtual environment
. set_environment.sh
set -e nounset 

# Python modules
pip install -r requirements.txt

# AMR dependencies
rm -Rf stog
git clone https://github.com/sheng-z/stog.git

# stog does not have an installer we create one with only the depencencies
# we need
echo "
from setuptools import setup
setup(
    name='stog',
    description='AMR Parsing as Sequence-to-Graph Transduction',
    py_modules=['stog'],
    install_requires=['penman==0.6.2', 'networkx'],
)
" > stog/setup.py
# editable just in case
pip install --editable stog

# Dowload meteor
[ ! -f meteor-1.5.tar.gz ] && wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xf meteor-1.5.tar.gz

# Download top-k nucleous sampler
[ ! -f "top-k-top-p.py" ] && wget https://gist.githubusercontent.com/thomwolf/1a5a29f6962089e871b94cbd09daf317/raw/91eea55d0353eaa18f7935e900d5e8d5e503e89c/top-k-top-p.py
