import re
import os
#from transformers import GPT2TokenizerFast
#from datasets import load_dataset
from itertools import chain
import numpy as np
import pandas as pd
import torch

import urllib.request
import zipfile
import requests
import json

from torch.utils.data import DataLoader, Dataset

class SeqDataset(Dataset):
    def __init__(self, data):
        self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in data['seq']]
        self.len_seq_data = [torch.tensor(len_seq, dtype=torch.long) for len_seq in data['len_seq']]
        self.next_data = [torch.tensor(next_val, dtype=torch.long) for next_val in data['next']]

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        if isinstance(idx, list):
        # Handle batch of indices by returning a batch of data
            return {'seq': [self.seq_data[i] for i in idx],
                    'len_seq': [self.len_seq_data[i] for i in idx],
                    'next': [self.next_data[i] for i in idx]}
        else:
        # Handle single index
            return {'seq': self.seq_data[idx], 'len_seq': self.len_seq_data[idx], 'next': self.next_data[idx]}

def get_seqdataloader(config):
    if config.training.data == "ATV":
        data_directory = config.data.ATV.path
        #seq_len = config.data.ATV.seq_len
        #item_num = config.data.ATV.item_num
    elif config.training.data == "ML1M":
        data_directory = config.data.ML1M.path
        #seq_len = config.data.ML1M.seq_len
        #item_num = config.data.ML1M.item_num
    #model_directory = './model/' + args.data +"/"
    elif config.training.data == "Steam":
        data_directory = config.data.Steam.path
    elif config.training.data == "ATG":
        data_directory = config.data.ATG.path
    elif config.training.data == "ASO":
        data_directory = config.data.ASO.path
    elif config.training.data == "Beauty":
        data_directory = config.data.Beauty.path

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    val_data = pd.read_pickle(os.path.join(data_directory, 'val_data.df'))
    test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))

    train_dataset = SeqDataset(train_data)
    val_dataset = SeqDataset(val_data)
    test_dataset = SeqDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=int(config.training.batch_size))
    val_loader = DataLoader(val_dataset, batch_size=int(config.training.batch_size))
    test_loader = DataLoader(test_dataset, batch_size=int(config.training.batch_size))
    
    return train_loader, val_loader, test_loader


# def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
#     if name == "wikitext103":
#         dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
#     elif name == "wikitext2":
#         dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
#     elif name == "ptb":
#         dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
#     elif name == "lambada":
#         dataset = get_lambada_test_dataset()
#     else:
#         dataset = load_dataset(name, cache_dir=cache_dir)

#     if name == "lambada":
#         data = dataset
#     else:
#         data = dataset[mode]

#     if name.startswith("wikitext"):
#         detokenizer = wt_detokenizer
#     elif name == "ptb":
#         detokenizer = ptb_detokenizer
#     elif name == "lm1b":
#         detokenizer = lm1b_detokenizer
#     elif name == "lambada":
#         detokenizer = lambada_detokenizer
#     else:
#         detokenizer = None

#     def _apply_detokenizer(detokenizer):
#         def detok(text):
#             for i, t in enumerate(text, 0):
#                  text[i] = detokenizer(t)
#             return text
#         return detok

#     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
#     EOS = tokenizer.encode(tokenizer.eos_token)[0]

#     def preprocess_and_tokenize(example):
#         if name == "ptb":
#             text = example['sentence']
#         else:
#             text = example["text"]
#         # print(list(example.keys()))
#         # exit()
        
#         if detokenizer is not None:
#             text = _apply_detokenizer(detokenizer)(text)

#         tokens = tokenizer(text, return_attention_mask=False)
#         # add in EOS token following 
#         # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
#         for token in tokens['input_ids']:
#             token.append(EOS)
#         return tokens
    
#     tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
#     if name == "ptb":
#         tokenized_dataset = tokenized_dataset.remove_columns('sentence')
#     else:
#         tokenized_dataset = tokenized_dataset.remove_columns('text')
    

#     def group_texts(examples):
#         # Concatenate all texts.
#         concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#         total_length = len(concatenated_examples[list(examples.keys())[0]])
#         # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
#         # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
#         total_length = (total_length // block_size) * block_size
#         # Split by chunks of max_len.
#         result = {
#             k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#             for k, t in concatenated_examples.items()
#         }
#         return result

#     chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
#     chunked_dataset = chunked_dataset.with_format('torch')

#     return chunked_dataset


# def get_dataloaders(config, distributed=True):
#     if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
#             raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
#     if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
#         raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


#     train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
#     valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)

#     if distributed:
#         train_sampler = DistributedSampler(train_set) 
#         test_sampler = DistributedSampler(valid_set)
#     else:
#         train_sampler = None
#         test_sampler = None
    

#     train_loader = cycle_loader(DataLoader(
#         train_set,
#         batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
#         sampler=train_sampler,
#         num_workers=4,
#         pin_memory=True,
#         shuffle=(train_sampler is None),
#         persistent_workers=True,
#     ))
#     valid_loader = cycle_loader(DataLoader(
#         valid_set,
#         batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
#         sampler=test_sampler,
#         num_workers=4,
#         pin_memory=True,
#         shuffle=(test_sampler is None),
#     ))
#     return train_loader, valid_loader

