import torch
from torch import nn

from model import KeyMetAAProfileModel
from utils import train, evaluate, KeyMetAADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("Using device: ", device)
    
    model = KeyMetAAProfileModel()
    print("Model Total Params", sum(p.numel() for p in model.parameters()))
    
    train_data = KeyMetAADataset("2025rzhang1")
    val_data = train_data
    
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 10
    NUM_WORKERS = 2
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
        
    print(train_data[2])
    # train(
    #     "keymetaa", 
    #     model, 
    #     train_data, val_data,
    #     criterion, optimizer,
    #     logits=True,
    #     num_epochs=EPOCHS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    # )
    
    torch.save(model.state_dict(), "test.pt")
    
    return


if __name__=="__main__":
    main()