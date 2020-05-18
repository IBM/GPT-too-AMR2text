set -o pipefail
set -o errexit
# setup environment
. set_environment.sh
config=$1
[ -z $config ] && echo -e "\n$0 <config>\n" && exit 1
set -o nounset

# load config
. $config

# Loop over existing checkpoints backwards from the highest epoch
checkpoints=$(find $CHECKPOINTS_DIR_ROOT -iname 'checkpoint_mymodel_*.pth' | sort -r)
for this_checkpoint in $checkpoints;do

    # this will get used for all related file names
    basepath="$RESULTS_FOLDER/$(basename $this_checkpoint .pth)"

    # skip if results exist
    [ -f "${basepath}.post.results" ] && \
        echo "Skipping ${basepath}.post.results" && \
        continue

    # run test if the results file does not exist
    if [ ! -f ${basepath}.txt ];then 

        echo ${basepath}.pth
        rm -f $CHECKPOINTS_DIR_ROOT/pytorch_model.bin
        ln -s $(basename $this_checkpoint) \
            $CHECKPOINTS_DIR_ROOT/pytorch_model.bin
        # test
        echo "python sampling_batch.py $TEST_ARGS"
        python sampling_batch.py $TEST_ARGS
        # rename result, remove soft-link
        rm -f $CHECKPOINTS_DIR_ROOT/pytorch_model.bin
        mv ${RESULTS_FOLDER}/output.txt ${basepath}.txt

    fi
    
    # post-processing
    echo "${basepath}.post.txt"
    python scripts/amr2txt/postprocess.py \
        $POSTPROCESS_ARGS \
        --in-tokens ${basepath}.txt \
        --out-tokens ${basepath}.post.txt \
    
done
