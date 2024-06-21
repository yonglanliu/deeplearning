import torch
import torch.nn as nn
from transformer.module.encoder import Encoder
from transformer.module.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, 
                 en_vocab_size, 
                 cn_vocab_size,
                 max_len, 
                 num_hiddens, 
                 ffn_hiddens, 
                 num_heads, 
                 drop_prob,
                 num_layers,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._encoder = Encoder(en_vocab_size, max_len, num_hiddens, ffn_hiddens, num_heads, drop_prob, num_layers)
        self._decoder = Decoder(cn_vocab_size, max_len, num_hiddens, ffn_hiddens, num_heads, drop_prob, num_layers)
        self._linear = nn.Linear(num_hiddens, cn_vocab_size)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x, x_valid_lens, y):
        x = self._encoder(x, x_valid_lens)
        out = self._decoder(x, x_valid_lens, y)
        out = self._linear(out)
        return out
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def configure_loss(self):
        return nn.functional.nll_loss(ignore_index=0, reduction="none")
    
    
if __name__ == "__main__":
    from ..utils.dataloader import batch_loader
    from ..utils.tokenizer import en_vocab_size, cn_vocab_size, batch_tokenize
    BATCH_SIZE = 16

    train_loader, test_loader = batch_loader(BATCH_SIZE=16)
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
    num_layers = 2
    drop_prob = 0.5

    transformer = Transformer(en_vocab_size, cn_vocab_size, max_len, num_hiddens, ffn_hiddens, num_heads, drop_prob, num_layers)
    out1 = transformer(en_tokenized, en_valid_lens, cn_tokenized)
    print(out1)
    print(out1.shape)