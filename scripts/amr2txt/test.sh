set -o pipefail
set -o errexit
. set_environment.sh
config=$1
[ -z $config ] && echo "$0 <config> <checkpoint>" && exit 1
checkpoint=$2
[ -z $checkpoint ] && echo "$0 <config> <checkpoint>" && exit 1
set -o nounset

[ ! -f "$checkpoint" ] && echo "Missing $checkpoint" && exit 1

# load config
. $config

# this will get used for all related file names
checkpoint_basename=$(basename $checkpoint .pth)
basepath="$RESULTS_FOLDER/$checkpoint_basename"

# TEST 
# we need to link the model and store the output
rm -f $CHECKPOINTS_DIR_ROOT/pytorch_model.bin
ln -s ${checkpoint_basename}.pth $CHECKPOINTS_DIR_ROOT/pytorch_model.bin
# test
python sampling_batch.py $TEST_ARGS
# rename result
mv ${RESULTS_FOLDER}/output.txt ${basepath}.txt

# POST-PROCESS 

# identify the beam size for post-processing
beam_size=$(python edit_config.py -i config/${test_config}.yaml -g beam_size)

# apply corresponding post processing
if [ "$beam_size" == 1 ];then

    python scripts/amr2txt/postprocess.py \
        $POSTPROCESS_ARGS \
        --in-tokens ${basepath}.txt \
        --out-tokens ${basepath}.post.txt 

else

    # Split beam into files
    python scripts/amr2txt/split_beams.py \
        --in-text ${basepath}.txt \
        --out-text "${basepath}.%K.txt" \
        --beam-size 15

    # process each file
    for n in $(seq $((beam_size - 1)));do
        python scripts/amr2txt/postprocess.py \
            $POSTPROCESS_ARGS \
            --in-tokens ${basepath}.$((n-1)).txt \
            --out-tokens ${basepath}.$((n-1)).post.txt 
    done

    # remove the original file
    rm ${basepath}.txt
    rm ${basepath}.post.txt

fi
