import torch
from torch import nn

class KeyMetAAProfileModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        feature_vec_dims = 27*27 + 27 # a-z and spaces pairwise latency and press duration
        self.model = nn.Sequential(
            nn.Linear(feature_vec_dims, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        
        self.cnn = nn.Conv1d(1, 2, 3)
        
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
            
        return out