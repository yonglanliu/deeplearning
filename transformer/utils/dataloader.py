import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.tokenizer import en_vocab_size, cn_vocab_size, batch_tokenize

class Textdataset(Dataset):
    def __init__(self, english_sentences, chinese_sentences):
        self.english_sentences = english_sentences
        self.chinese_sentences = chinese_sentences

    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self, idx):
        return self.english_sentences[idx], self.chinese_sentences[idx]
    
    
def batch_loader(data, BATCH_SIZE=16, split_ratio=0.8):

    english_sentences = data[:, 0]
    chinese_sentences = data[:, 1]

    dataset = Textdataset(english_sentences, chinese_sentences)

    # Split dataset into traning set and validation set
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    print(f'Training set size: {len(train_set)}\nValidation set size: {len(val_set)}')

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)

    return train_loader, val_loader


if __name__ == "__main__":
    from transformer.module.encoder import Encoder 
    from transformer.module.decoder import Decoder
    DATA_PATH="./dataset/dataset.npy"
    combined_sentences = np.load(DATA_PATH)
    BATCH_SIZE = 16

    train_loader, test_loader = batch_loader(combined_sentences, BATCH_SIZE=BATCH_SIZE)
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        #print(batch)
        if batch_num > 3:
            break
    en_tokenized, en_valid_lens, cn_tokenized, cn_valid_lens = batch_tokenize(batch)

    max_len = 256
    num_hiddens = 512
    ffn_hiddens = 48
    num_heads = 8

    encoder = Encoder(en_vocab_size, max_len, num_hiddens, ffn_hiddens, num_heads, 0.5, 2)
    out = encoder(en_tokenized, en_valid_lens)
    #print(out)
    print(out.shape)

    decoder = Decoder(cn_vocab_size, max_len, num_hiddens, ffn_hiddens, num_heads, 0.5, 2)
    out1 = decoder(out, en_valid_lens, cn_tokenized)
    #print(out1)
    #print(out1.shape)