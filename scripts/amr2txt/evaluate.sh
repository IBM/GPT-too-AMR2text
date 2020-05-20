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
output_texts=$(find $RESULTS_FOLDER -regextype sed -regex '.*/checkpoint_mymodel_[0-9\.]*.txt')

for output_text in $output_texts;do

    basepath=$(dirname $output_text)/$(basename $output_text .txt)

    # post process if needed
    [ ! -f "${basepath}.post.txt" ] && \
        python scripts/amr2txt/postprocess.py \
            $POSTPROCESS_ARGS \
            --in-tokens ${basepath}.txt \
            --out-tokens ${basepath}.post.txt \

    # evaluation
    echo "${basepath}.post.results"
    python extra/eval_sacrebleu.py \
        --in-reference-tokens $FEATURES_FOLDER/dev.tok.text \
        --in-tokens ${basepath}.post.txt \
        --out-results ${basepath}.post.results

    # TODO: Needs another version of transformers (2.0)
    # bert-score -r DATA/dev.tok.text -c ${basepath}.post.txt --lang en

    # METEOR
    # TODO: extract metric from log and store in ${basepath}.post.results
    # java -Xmx2G -jar meteor-1.5/meteor-1.5.jar DATA/dev.tok.text ${basepath}.post.txt

done

# locate best epoch and display results
python scripts/amr2txt/rank_model.py --checkpoints DATA/models/ --link-best
