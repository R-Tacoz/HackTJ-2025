import os, math, time

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

def createFeatureVec(response):
    
    return []

class KeyMetAADataset(torch.utils.data.Dataset):
    def __init__(self, user, path="./processing/data/keystrokesdata.csv"):
        self.user = user
        self.path = path
        self.data_begin_idx = 2 # index 0 and 1 are username and password
        
        self.df = pd.read_csv(self.path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int) -> tuple:
        feature_vec = self.df.iloc[index, self.data_begin_idx:]
        label = 1 if self.df.iloc[index, 0] == self.user else 0
        return feature_vec, label
    

def train(
    model_name:  str,
    model:       torch.nn.Module, 
    train_data:  torch.utils.data.Dataset,
    val_data:    torch.utils.data.Dataset,
    criterion:   torch.nn.Module, 
    optimizer:   torch.optim.Optimizer, 
    scheduler:   object = None, 
    num_epochs:  int = 0, 
    batch_size:  int = 1,
    num_workers: int = 0,
    logits:      bool = True,
    device:      torch.device = torch.device("cpu"),
    verbose:     bool = True,
    exp:         int = -1,
    delay:       int = 100,
    logging:     str = None, # file path
) -> torch.nn.Module:
    
    print(f"Begining Training of {model_name}...")
    if verbose: print("\t\t\t Progress \t\t\t\t\t\t\t Last Epoch Stats:")
    start_time = time.perf_counter()
    
    loss_history, train_acc_history, val_acc_history = [], [], []
    last_acc = 0.0

    dataset_size = torch.Tensor([len(train_data)])
    data_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True, num_workers=num_workers)
    
    model.to(device)
    model.train()
    epoch_loss, epoch_acc, acc_dif, val_acc = torch.zeros(1),torch.zeros(1),torch.zeros(1), torch.zeros(1)
    epoch_iterator = tqdm(range(num_epochs), "Training") if not verbose else range(num_epochs)
    for epoch in epoch_iterator:
        if not verbose: epoch_iterator.set_description(f"Training; Loss: {epoch_loss.item():.3f} Val Acc: {val_acc.item():.2%}")
        running_loss = 0
        running_corrects = 0

        batch_iterator = tqdm(
            iterable = data_loader,
            desc = f"Epoch {epoch+1}/{num_epochs}",
            bar_format = "{desc}: |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
            postfix = f"Loss: {epoch_loss.item():.4f}\t\
Accuracy: {epoch_acc.item():.2%} ({'+' if acc_dif.item()>=0 else ''}{acc_dif.item():.2%})",
        ) if verbose else data_loader
        i = 0
        for x, gt in batch_iterator:
            x = x.to(dtype=torch.float32, device=device)
            gt = gt.to(dtype=torch.float32, device=device)
            
            # forward
            output = model(x)
            #_, preds = torch.max(output, 1)
            loss = criterion(output, gt) #TODO: loss for param share
            
            # backward                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            if i % delay == 0: loss_history.append(loss.detach().cpu().item())
            current_batch_size = x.size(0)
            running_loss += loss.item() * current_batch_size
            running_corrects += torch.sum(abs( (torch.sigmoid(output) if logits else output) - (gt >= 0.5).int() < 0.5)).cpu()
            #if abs((output - y).item()) < 0.5: running_corrects += 1
            
            i += 1

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / (gt.size(1) * dataset_size)
        #epoch_acc = torch.Tensor([running_corrects / dataset_size])
        val_acc = evaluate(model, val_data, device)
        
        if scheduler != None:
            if type(scheduler) == torch.optim.lr_scheduler.OneCycleLR:
                if not epoch > scheduler.total_steps:
                    scheduler.step()
            else:
                scheduler.step(epoch_acc)

        train_acc_history.append(epoch_acc.detach().cpu().item())
        val_acc_history.append(val_acc.detach().cpu().item())
        acc_dif = epoch_acc - last_acc
        last_acc = epoch_acc
        
    time_elapsed = time.perf_counter() - start_time
    print(f'Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Final Training Accuracy: {epoch_acc.item():.2%}')
    print(f"Final Validation Accuracy: {val_acc.item():.2%}")
    
    #plot loss
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.plot(range(0,len(loss_history)*delay, delay), loss_history)
    plt.grid(True)
    # if exp >= 0:
    #     plt.savefig(f"./runs/{model_name}/exp{exp}/iter_loss.png")
    plt.show()
    
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.plot(range(0,len(train_acc_history)), train_acc_history, label="Train")
    plt.plot(range(0, len(val_acc_history)), val_acc_history, label="Validation")
    plt.legend()
    plt.grid(True)
    # if exp >= 0:
    #     plt.savefig(f"./runs/{model_name}/exp{exp}/iter_accuracy.png")
    plt.show()

    return model

def evaluate(
    model:         torch.nn.Module,
    val_data:      torch.utils.data.Dataset,
    device:        torch.device=torch.device("cpu"),
    logits:        bool = True,
) -> torch.Tensor: 
    dataloader = DataLoader(val_data, batch_size=16)
    model.eval()
    corrects, total = 0,0
    with torch.no_grad():
        for x, gt in dataloader:
            x = x.to(device)
            gt = (gt >= 0.5).to(device=device, dtype=torch.int32)
            
            output = model(x)
            if logits: output = torch.sigmoid(output)
            
            corrects += torch.sum(abs(output - gt) < 0.5)
            total += x.size(0) * gt.size(1)
    
    return torch.Tensor([corrects / total])