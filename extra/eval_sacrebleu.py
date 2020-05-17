import sys
from typing import Iterable, Optional
import sacrebleu
import re


def tokenize_sentence(text, debug=False):
    text = re.sub(r"('ll|n't|'m|'s|'d|'re)", r" \1", text)
    text = re.sub(r"(\s+)", r" ", text)
    return text


def raw_corpus_bleu(hypothesis: Iterable[str], reference: Iterable[str],
                    offset: Optional[float] = 0.01) -> float:
    bleu = sacrebleu.corpus_bleu(hypothesis, reference, smooth_value=offset,
                                 force=True, use_effective_order=False,
                                 lowercase=True)
    return bleu.score


def raw_corpus_chrf(hypotheses: Iterable[str],
                    references: Iterable[str]) -> float:
    return sacrebleu.corpus_chrf(hypotheses, references,
                                 order=sacrebleu.CHRF_ORDER,
                                 beta=sacrebleu.CHRF_BETA,
                                 remove_whitespace=True)


if __name__ == '__main__':

    ref = open(sys.argv[1]).readlines()
    hyp = open(sys.argv[2]).readlines()

    # Lower evaluation
    for i in range(len(ref)):
        ref[i] = ref[i].lower()

    # Lower case output
    for i in range(len(hyp)):
        if '<generate>' in hyp[i]:
            hyp[i] = hyp[i].split('<generate>')[-1]
        hyp[i] = tokenize_sentence(hyp[i].lower())

    print(len(hyp), len(ref))

    # Run evaluation
    print("BLEU", round(raw_corpus_bleu(hyp, [ref]), 2))
    print("chrF++", round(raw_corpus_chrf(hyp, ref)*100, 2))
