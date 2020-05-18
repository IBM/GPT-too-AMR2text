import os
import json
import re
import argparse


def argument_parser():

    parser = argparse.ArgumentParser(description='Preprocess AMR data')
    # Multiple input parameters
    parser.add_argument(
        "--in-amr",
        help="input AMR file",
        type=str
    )
    parser.add_argument(
        "--out-amr",
        help="output (post-processed) AMR file",
        type=str
    )
    parser.add_argument(
        "--out-tokens",
        help="tokens from AMR",
        type=str
    )
    parser.add_argument(
        "--stog-fix",
        action='store_true',
        help="Reformat AMR token to be parseable by publict stog"
    )
    args = parser.parse_args()

    return args


def fix_tokens_file(file_path):
    """
    Replace each

    # ::tok sentence

    by json parsable version

    # ::token json-parseable-sentence

    so that

    sentence == json.loads(json-parseable-sentence)
    """

    token_line = re.compile('^# ::tok (.*)')

    # read and modifiy token lines
    new_amr = []
    tokens = []
    with open(file_path) as fid:
        for line in fid:
            fetch = token_line.match(line.rstrip())
            if fetch:
                sentence = fetch.groups()[0]
                tokens.append(sentence)
                json_str = json.dumps(sentence)
                new_amr.append(f'# ::tokens {json_str}\n')
            else:
                new_amr.append(line)

    return new_amr, tokens


if __name__ == '__main__':

    # Argument handlig
    args = argument_parser()

    assert os.path.isfile(args.in_amr), \
        f'{args.in_amr} is missing or is not a file'

    # create pre-processed AMR and extract tokens
    new_amr, tokens = fix_tokens_file(args.in_amr)
    assert tokens, "did not find tokens, AMR already formatted?"

    # write pre-processed AMR
    if args.stog_fix:
        print(args.out_amr)
        with open(args.out_amr, 'w') as fid:
            for line in new_amr:
                fid.write(line)

    # write tokens
    if args.out_tokens:
        print(args.out_tokens)
        with open(args.out_tokens, 'w') as fid:
            for tok_sent in tokens:
                fid.write(f'{tok_sent}\n')
