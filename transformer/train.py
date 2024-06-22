from transformer.model import Transformer
import torch 
import torch.nn as nn
from utils.tokenizer import en_vocab_size, cn_vocab_size, batch_tokenize
from utils.dataloader import batch_loader
import torch.optim as optim
import copy
import time
import math
import os
import argparse
import logging
#from torch.utils.tensorboard import SummaryWriter

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class Trainer():
    def __init__(self, model: nn.Module, data, batch_size: int, max_epochs: int, grad_clip_val:float, restart:bool=False):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # prepare data
        self.prepare_data(data, batch_size)

        # prepare model
        self.model = model

        # Initialize parameters
        if restart == False:
            for params in self.model.parameters():
                if params.dim() > 1:
                    nn.init.xavier_uniform_(params)
        else:
            self.model.load_state_dict(torch.load("checkpoint/latest.pt"))
        
        # prepare loss function and optimizer
        self.criterian = self.prepare_criterian()
        self.optim = self.prepare_optim(self.model)

        self.max_epochs = max_epochs
        self.grad_clip_val = grad_clip_val
        self.restart = restart

    def prepare_data(self, data, batch_size):
        self.train_dataloader, self.val_dataloader = batch_loader(data, BATCH_SIZE=batch_size)
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
    
    def prepare_criterian(self):
        return nn.CrossEntropyLoss(ignore_index=0)
    
    def prepare_optim(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.1)

    def fit_epoch(self):
        """
        batch training for each epoch
        """
        # Training
        self.model.train()
        self.model.to(self.device)
        epoch_loss = 0
        train_iterator = iter(self.train_dataloader)
        for _, batch in enumerate(train_iterator):
            en_tokenized, en_valid_lens, cn_tokenized, _ = batch_tokenize(batch)
            self.optim.zero_grad()

            output = self.model(en_tokenized, en_valid_lens, cn_tokenized)
            output = output[1:].view(-1, output.shape[-1]).to(self.device)
            trg = cn_tokenized[1:].view(-1).to(self.device)
            train_loss = self.criterian(output, trg).to(self.device)
            
            train_loss.backward()
            if self.grad_clip_val > 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
            self.optim.step()
            epoch_loss += train_loss.item()

        # Validation 
        if self.val_dataloader is None:
            return
        self.model.eval()
        val_epoch_loss = 0
        val_iterator = iter(self.val_dataloader)
        for _, batch in enumerate(val_iterator):
            with torch.no_grad():
                val_en_tokenized, val_en_valid_lens, val_cn_tokenized, _ = batch_tokenize(batch)
                val_output = self.model(val_en_tokenized, val_en_valid_lens, val_cn_tokenized)
                val_output = val_output[1:].view(-1, val_output.shape[-1])
                val_trg = val_cn_tokenized[1:].view(-1)
                val_loss = self.criterian(val_output, val_trg)
                val_epoch_loss += val_loss.item()
        return epoch_loss / len(train_iterator), val_epoch_loss / len(val_iterator)

    def fit(self):
        self.setup_logging()
        #logger = SummaryWriter(os.path.join("logs", "training.log"))
        print(f"Training is preparing.")
        if self.restart == True:
            checkpoint = torch.load(os.path.join("checkpoint", "chk_latest.chk"))
            start_epoch = checkpoint['epoch'] + 1
            train_losses, val_losses = checkpoint["train_loss"], checkpoint["valid_loss"]
            best_valid_loss = checkpoint["best_valid_loss"]
        else:
            train_losses, val_losses = [], []
            start_epoch = 0
            best_valid_loss = float('inf')

        for epoch in range(start_epoch, self.max_epochs):
            start_time = time.time()
            logging.info(f"Start epoch {epoch} at {start_time}")
            train_loss, valid_loss = self.fit_epoch()
            end_time = time.time()
            logging.info(f"Finish epoch {epoch} at {end_time}")
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

            torch.save(self.model.cpu().state_dict(), f'checkpoint/epoch_{epoch}.pt')
            torch.save(self.model.cpu().state_dict(), f'checkpoint/latest.pt')
            train_losses.append(train_loss)
            val_losses.append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(self.model)

                torch.save(best_model.cpu().state_dict(), f'checkpoint/best.pt')
                torch.save({"epoch": epoch, "train_loss": train_losses, "valid_loss":val_losses, "best_valid_loss": best_valid_loss}, 
                           os.path.join("checkpoint", "chk_best.chk"))    
                    
            torch.save({"epoch": epoch, "train_loss": train_losses, "valid_loss":val_losses, "best_valid_loss": best_valid_loss}, 
                       os.path.join("checkpoint", "chk_latest.chk"))
            
        #self.plot(train_losses, val_losses)
        print(f'\t Val. Loss of best model: {valid_loss:.3f} |  Val. PPL of best model: {math.exp(valid_loss):7.3f}')
    
    def setup_logging(self):
        #os.makedirs("logs", exist_ok=True)
        #os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoint", exist_ok=True)

    def plot(self, train_loss, valid_loss):
        from matplotlib import pyplot as plt
        plt.plot(train_loss, label='train_loss')
        plt.plot(valid_loss,label='val_loss')
        plt.legend()
        plt.show

def train(args):
    import numpy as np
    model = Transformer(en_vocab_size, 
                        cn_vocab_size, 
                        max_len=args.max_len, 
                        num_hiddens=512, 
                        ffn_hiddens=2048, 
                        num_heads=8, 
                        drop_prob=args.drop_prob, 
                        num_layers=args.num_layers)
    data = np.load("dataset/dataset.npy")
    trainer = Trainer(model,
                      data=data,
                      batch_size=args.batch_size,
                      max_epochs=args.max_epochs, 
                      grad_clip_val=1.0,
                      restart=args.restart)
    trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=256, help="max length of sentences")
    parser.add_argument("--drop_prob", type=float, default=0.5, help="Drop out probability")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--max_epochs", type=int, default=10, help='number of epochs to train')
    parser.add_argument("--restart", type=bool, default=False, help="True: continue to training\nFalse:start from begining")
    args = parser.parse_args()
    train(args)