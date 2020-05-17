SPECIAL_TOKENS = ["<bos>", "<eos>", "<context>", "<answer>", "<question>",
                  "<pad>", "<generate>"]
SPECIAL_TOKENS_BERT = ["[CLS]", "[SEP]", "[PAD]", "[unused0]"]
MODEL_INPUTS = ["input_ids", "labels", "token_type_ids", "attention_mask"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids", "attention_mask"]
OTHERS = ["<unk2>"]
AMR_SPECIAL_TOKENS = ['multi-sentence', '<join>', ':snt1', ':snt2',
                      ':snt3', ':snt4', ':snt5', ':snt5', ':snt6', ':snt7',
                      ':snt8', ':snt9']

SPECIAL_TOKENS = SPECIAL_TOKENS
