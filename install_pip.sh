# Python modules
if [ ! -d venv ];then
    virtualenv venv
    . venv/bin/activate
    pip install -r requirements.txt
fi

# Data
if [ ! -d DATA ];then
    git clone https://github.com/tomhosking/squad-du-split DATA
fi
