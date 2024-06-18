from transformers import BertTokenizer
import torch

tokenizer_en = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer_cn = BertTokenizer.from_pretrained("bert-base-chinese")
en_vocab_size = tokenizer_en.vocab_size
cn_vocab_size = tokenizer_cn.vocab_size

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def tokenize(sentence, language, max_len=256, add_special_tokens=True, PRINT=False):
    if language == "en":
        tokenizer = tokenizer_en
    elif language == "cn":
        tokenizer = tokenizer_cn
    tokenized_seq = tokenizer.tokenize(sentence)
    tokenized_valid_len = len(tokenized_seq)
    tokenize_id = tokenizer.encode(sentence, max_length=max_len, add_special_tokens=add_special_tokens, padding="max_length", truncation=True)
    if PRINT==True:
        print(f'Original Sentence: {sentence}')
        print(f'Tokenized Sequence: {tokenized_seq}')
        print(f'Tokenized IDs: {tokenize_id}')
    return tokenize_id, tokenized_valid_len    

def batch_tokenize(batch):
    en_tokenized = []
    en_valid_lens = []
    cn_tokenized = []
    cn_valid_lens = []
    for en, cn in zip(batch[0], batch[1]):
        # obtain tokens
        en_token, en_valid_len = tokenize(en, language="en", add_special_tokens=False)
        cn_token, cn_valid_len = tokenize(cn, language="cn", add_special_tokens=True)
        # append token together
        en_tokenized.append(en_token)
        en_valid_lens.append(en_valid_len)
        cn_tokenized.append(cn_token)
        cn_valid_lens.append(cn_valid_len)
    # convert to torch tensor
    en_tokenized = torch.tensor(en_tokenized)
    en_valid_lens = torch.tensor(en_valid_lens)
    cn_tokenized = torch.tensor(cn_tokenized)
    cn_valid_lens = torch.tensor(cn_valid_lens)
    return en_tokenized, en_valid_lens, cn_tokenized, cn_valid_lens
    #return en_tokenized.to(get_device()), valid_lens.to(get_device), cn_tokenized.to(get_device)