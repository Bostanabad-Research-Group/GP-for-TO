import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)                
    np.random.seed(seed)             
    torch.manual_seed(seed)          # PyTorch CPU seed
    if torch.cuda.is_available():    # If CUDA is available
        torch.cuda.manual_seed(seed)            
        torch.cuda.manual_seed_all(seed)        
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False  
