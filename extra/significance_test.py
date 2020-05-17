import sys

import sacrebleu
from eval_sacrebleu import tokenize_sentence, raw_corpus_bleu, raw_corpus_chrf
from scipy.stats import ttest_ind

if __name__ == '__main__':

    ref = open(sys.argv[1]).readlines()
    hyp1 = open(sys.argv[2]).readlines()
    hyp2 = open(sys.argv[3]).readlines()

    # Lower evaluation
    for i in range(len(ref)):
        ref[i] = ref[i].lower()
     
    # Lower case output
    for i in range(len(hyp1)):
        if '<generate>' in hyp1[i]:
            hyp1[i] = hyp1[i].split('<generate>')[-1]
        hyp1[i] = tokenize_sentence(hyp1[i].lower())
        if '<generate>' in hyp2[i]:
            hyp2[i] = hyp2[i].split('<generate>')[-1]
        hyp2[i] = tokenize_sentence(hyp2[i].lower())
    

    # Run evaluation
    print("BLEU hyp1", round(raw_corpus_bleu(hyp1, [ref]),2))
    print("chrF++ hyp1", round(raw_corpus_chrf(hyp1, ref).score*100,2))
    

    print("BLEU hyp2", round(raw_corpus_bleu(hyp2, [ref]),2))
    print("chrF++ hyp2", round(raw_corpus_chrf(hyp2, ref).score*100,2))

    h1_sent_scores = list()
    h2_sent_scores = list()
    
    for r, h1, h2 in zip(ref, hyp1, hyp2):
        h1_sent_scores.append(sacrebleu.sentence_bleu(h1, r).score)
        h2_sent_scores.append(sacrebleu.sentence_bleu(h2, r).score)

    t = ttest_ind(h1_sent_scores, h2_sent_scores)
    print("BLEU P value:", "{:.20f}".format(t[1]))

    h1_sent_scores = list()
    h2_sent_scores = list()
    
    for r, h1, h2 in zip(ref, hyp1, hyp2):
        h1_sent_scores.append(sacrebleu.sentence_chrf(h1, r).score)
        h2_sent_scores.append(sacrebleu.sentence_chrf(h2, r).score)
 
    t = ttest_ind(h1_sent_scores, h2_sent_scores)
    print("ChrF++ P value:", "{:.20f}".format(t[1]))
