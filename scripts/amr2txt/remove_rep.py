import re
import string
import sys

def tokenize_sentence(text, debug=False):
    #debug = True
    if debug: print(text.replace("\n", " "))
    text = re.sub(r"('ll|n't|'m|'s|'d|'re)", r" \1", text)
    text = re.sub(r"(\s+)", r" ", text)
    if debug: print(text)
    return text


def remove_rep(text):
    pattern = ""
    stext = text.split(" ")
    if len(stext) < 10:
        return text
    one_gram = ""
    two_gram = ""
    three_gram = ""
    cut = False
    cutting_point = 0
    last = 0
    current = ""
    for w in reversed(stext):
        current += w + " " + w
        three_gram = w+" "+two_gram
        two_gram = w+" "+one_gram
        one_gram = w
        try:
            s_one_gram = [m for m in re.finditer(re.escape(one_gram), text)]
            s_two_gram = [m for m in re.finditer(re.escape(two_gram), text)]
            s_three_gram = [m for m in re.finditer(re.escape(three_gram), text)]
        except:
            import ipdb; ipdb.set_trace(context=5)
        prop_one_gram = len(s_one_gram)/len(stext)
        prop_two_gram = len(s_two_gram)/len(stext)
        prop_three_gram = len(s_three_gram)/len(stext)

        if prop_three_gram > 0.03 and  len(s_three_gram)>4:
            #print(s_three_gram)
            #print(s_three_gram[1].group(), s_one_gram[1].start(), last, len(text)-len(current))
            if last > len(text)-len(current):
                #print(text)
                #print(prop_three_gram, "-"*80)
                #print("-"*80)
                #print("THREE suggested cut:", text[:s_one_gram[1].start()])
                #input()
                return text[:s_one_gram[1].start()]
            if not cut:
                cut = True
                last = s_one_gram[-2].start()
        elif prop_two_gram > 0.04 and len(s_two_gram)>6 and len(s_two_gram)>3:
            #print(s_two_gram)
            #print(s_two_gram[1].group(), s_one_gram[1].start(), last, len(text)-len(current))
            if last > len(text)-len(current):
                #print(text)
                #print(prop_two_gram, "-"*80)
                #print("TWO suggested cut:", text[:s_one_gram[1].start()])
                #input()
                return text[:s_one_gram[1].start()]
            if not cut:
                cut = True
                last = s_one_gram[-2].start()
        elif prop_one_gram > 0.2  and len(s_one_gram)>10 and len(w) > 2 and w not in string.punctuation:
            #print(s_one_gram[1].group(), s_one_gram[1].start(), last, len(text)-len(current))
            #print(cut, last)
            if last > len(text)-len(current):
                #print(text)
                #print(prop_one_gram, "-"*80)
                #print("ONE suggested cut:", text[:s_one_gram[1].start()])
                #input()
                return text[:s_one_gram[1].start()]
            if not cut:
                cut = True
                last = s_one_gram[-2].start()
    return text

def preproc_doc(file_name, lower_case=True, tokenize=True):
    text = open(file_name).read().splitlines()
    output = list()
    for line in text:
        if lower_case:
            line = line.lower()
        if tokenize:
            line = tokenize_sentence(line)
        output.append(remove_rep(line))

    return output
 

if __name__ == '__main__': 
       output = preproc_doc(sys.argv[1])
       for line in output:
           print(line)
