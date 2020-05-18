# CAPITALIZED variables are expected to be used outside of this file. You need
# to understand their effect if you change them

# configuration 
# original data
AMR_TRAIN_FILE=DATA/train.amr
AMR_DEV_FILE=DATA/dev.amr
AMR_TEST_FILE=DATA/test.amr

# PREPROCESSING
# This will process and rename files above as
# $FEATURES_FOLDER/{train,dev,test}.amr This name is defined by
# --dataset_path='DATA' in config and the file names are hard-coded in
# amr_utils.py see preprocess.py for flag configuration --stog-fix modifies AMR
# files to make tokenized sentences readable by stog
PREPRO_TAG=basic
FEATURES_FOLDER=DATA/features/$PREPRO_TAG/
PREPROCESS_ARGS="
    --stog-fix
"

# TRAINING
# scripts/amr2txt/train.sh
# preprocess.py
# train.py
base_model=gpt2-medium
train_config=amr-raw
exp_name="${PREPRO_TAG}_${base_model}_${train_config}"
# NOTE: --exp_name will determine this path. Modify exp_name to change it
# NOTE: We explicity define data location, cache and model folder. Internally
# it will store the models under:
CHECKPOINTS_DIR_ROOT=DATA/models/$exp_name/
TRAINING_ARGS="
    --config_path config/${train_config}.yaml
    --dataset_path $FEATURES_FOLDER
    --dataset_cache $FEATURES_FOLDER/cache/
    --checkpoint DATA/models/
    -mq $base_model
    --exp_name $exp_name
"

# TEST
# Checkpoint used for training
test_config=amr-raw-decoding
dev_exp_name="$test_config"
RESULTS_FOLDER=$CHECKPOINTS_DIR_ROOT/sampling/$dev_exp_name/
TEST_ARGS="
    --config_path config/${test_config}.yaml \
    --dataset_path $FEATURES_FOLDER
    --dataset_cache $FEATURES_FOLDER/cache/
    --model_path $CHECKPOINTS_DIR_ROOT
    --exp_name $dev_exp_name
"

# POSTPROCESSING
POSTPROCESS_ARGS="
    --remove-repetitions
"
