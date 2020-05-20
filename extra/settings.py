import torch

# Special symbols
PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
BOS = "<BOS>" # start of sequence
OOV = "<OOV>" # unknown token

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
OOV_IDX = 3

CUDA = torch.cuda.is_available()
