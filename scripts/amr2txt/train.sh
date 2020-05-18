set -o pipefail
set -o errexit
config=$1
[ -z "$config" ] && echo "$0 /path/to/config.sh" && exit 1
. set_environment.sh
set -o nounset

# load config
. $config

# PREPROCESS DATA 
if [ ! -f "$FEATURES_FOLDER/test.tok.text" ];then

    mkdir -p $FEATURES_FOLDER

    # extract tok.txt files, fix AMR to have tok field readable by stog)
    # train data
    python scripts/amr2txt/preproces.py \
        $PREPROCESS_ARGS \
        --in-amr $AMR_TRAIN_FILE \
        --out-amr $FEATURES_FOLDER/train.amr \
        --out-tokens $FEATURES_FOLDER/train.tok.text 
    
    # dev data
    python scripts/amr2txt/preproces.py \
        $PREPROCESS_ARGS \
        --in-amr $AMR_DEV_FILE \
        --out-amr $FEATURES_FOLDER/dev.amr \
        --out-tokens $FEATURES_FOLDER/dev.tok.text \
    
    # test data
    python scripts/amr2txt/preproces.py \
        $PREPROCESS_ARGS \
        --in-amr $AMR_DEV_FILE \
        --out-amr $FEATURES_FOLDER/test.amr \
        --out-tokens $FEATURES_FOLDER/test.tok.text 
fi

# TRAINING
python train.py $TRAINING_ARGS
