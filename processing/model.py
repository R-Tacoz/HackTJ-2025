import torch
from torch import nn

class KeyMetAAProfileModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        feature_vec_dims = 27*27 + 27 # a-z and spaces pairwise latency and press duration
        self.model = nn.Sequential(
            nn.Linear(feature_vec_dims, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
            
        return out