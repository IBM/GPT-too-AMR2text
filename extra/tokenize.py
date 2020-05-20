import re


def tokenize_sentence(text, debug=False):
    text = re.sub(r"('ll|n't|'m|'s|'d|'re)", r" \1", text)
    text = re.sub(r"(\s+)", r" ", text)

    return text.strip()
