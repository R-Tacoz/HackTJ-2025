import torch
from torch import nn

from model import KeyMetAAProfileModel
from utils import train, evaluate, KeyMetAADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_user_model(user):
    USERNAME =  user
    print("Using device: ", device)
    
    model = KeyMetAAProfileModel()
    print("Model Total Params", sum(p.numel() for p in model.parameters()))
    
    train_data = KeyMetAADataset(USERNAME, path="./data/main.csv")
    val_data = train_data
    
    LEARNING_RATE = 0.0008
    BATCH_SIZE = 128
    EPOCHS = 80
    NUM_WORKERS = 0
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train(
        f"{USERNAME}'s keymetaa", 
        model, 
        train_data, val_data,
        criterion, optimizer,
        logits=False,
        num_epochs=EPOCHS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        verbose=False,
    )
    
    torch.save(model.state_dict(), f"data/{USERNAME}.pt")
    
    return

def main():
    names = ['rocco', 'raymond', 'abhi']
    for name in names:
        train_user_model(name)
        print()

if __name__=="__main__":
    main()