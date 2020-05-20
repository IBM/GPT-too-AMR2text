import sys
import argparse


def argument_parsing():

    # Argument hanlding
    parser = argparse.ArgumentParser(
        description='Splits output of beam decoding into individual files'
    )
    # jbinfo args
    parser.add_argument(
        '--in-text',
        type=str,
        required=True,
        help='One sentence per file beam sentences contiguous in beam'
    )
    parser.add_argument(
        '--out-text',
        type=str,
        required=True,
        help='Sentences for one beam index. Needs %K in name to identify index'
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        required=True,
        help='size of the beam'
    )
    args = parser.parse_args()

    assert '%K' in args.out_text, \
        "--out-text name must contain %K to be replaced by beam index"

    return args


if __name__ ==  '__main__':

    # argument handling
    args = argument_parsing()

    # read input file
    with open(args.in_text) as fid:
        text = fid.read().splitlines()

    # sanity check
    assert len(text) % args.beam_size == 0, \
        "Number of lines in {args.in_text} not a multiple of {args.beam_size}"
    
    # split into beam_size output files
    out_files = [list() for _ in range(args.beam_size)]
    for i, line in enumerate(text):
        out_files[i % args.beam_size].append(line)

    # write output files
    for n, out_file in enumerate(out_files):
        out_text = args.out_text.replace('%K', f'{n}')
        print(out_text)
        with open(out_text, 'w') as fid:
            for line in out_file:
                fid.write(f'{line}\n')
