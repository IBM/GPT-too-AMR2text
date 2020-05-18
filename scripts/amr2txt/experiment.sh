set -o pipefail
set -o errexit
config=$1
[ -z "$config" ] && echo "$0 /path/to/config.sh" && exit 1
set -o nounset

# load config
. $config

# TRAINING
# this will store checkpoints and model files
mkdir -p $CHECKPOINTS_DIR_ROOT
if [ ! -f "$CHECKPOINTS_DIR_ROOT/config.sh" ];then
    echo "cp $config $CHECKPOINTS_DIR_ROOT/config.sh"
    cp $config $CHECKPOINTS_DIR_ROOT/config.sh
else
    echo "Resuming from $CHECKPOINTS_DIR_ROOT/config.sh"
fi
# preprocessing + train script
bash scripts/amr2txt/train.sh $CHECKPOINTS_DIR_ROOT/config.sh

# test all epochs
bash scripts/amr2txt/epoch_tester.sh $CHECKPOINTS_DIR_ROOT/config.sh

# evaluate them
bash scripts/amr2txt/evaluate.sh $CHECKPOINTS_DIR_ROOT/config.sh

# Evaluate beam
best_checkpoint="$CHECKPOINTS_DIR_ROOT/checkpoint_best_BLEU.pth"
[ ! -f "$best_checkpoint" ] && echo "--link-best model failed?" && exit 1
# create beam testing config
[ ! -f "$CHECKPOINTS_DIR_ROOT/config_beam_test.sh" ] && \
    cp $config $CHECKPOINTS_DIR_ROOT/config_beam_test.sh && \
    sed "s@^test_config=.*@test_config=amr-beam-decode@" -i $CHECKPOINTS_DIR_ROOT/config_beam_test.sh && \
    echo "Created $CHECKPOINTS_DIR_ROOT/config_beam_test.sh"
bash scripts/amr2txt/test.sh $CHECKPOINTS_DIR_ROOT/config_beam_test.sh $best_checkpoint

# evaluate beam results
bash scripts/amr2txt/evaluate.sh $CHECKPOINTS_DIR_ROOT/config_beam_test.sh
