import torch
import sys
from extra.settings import PAD_IDX, CUDA


def Tensor(*args):
    if CUDA:
        x = torch.tensor(*args, dtype=torch.float, device=torch.device("cuda"))
    else:
        x = torch.tensor(*args, dtype=torch.float)
    return x


def LongTensor(*args):
    if CUDA:
        x = torch.tensor(*args, dtype=torch.long, device=torch.device("cuda"))
    else:
        x = torch.tensor(*args, dtype=torch.long)
    return x


def zeros(*args):
    if CUDA:
        x = torch.zeros(*args, device=torch.device("cuda"))
    else:
        x = torch.zeros(*args)
    return x


def maskset(x):
    mask = x.data.eq(PAD_IDX)
    return (mask, x.size(1) - mask.sum(1))


def pause_or_exit():
    inp = input()
    if inp.lower() in ["e", "exit", "q", "quit"]:
        sys.exit()
