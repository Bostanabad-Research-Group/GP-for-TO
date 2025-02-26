import torch
import numpy as np

def setlevels(X, qual_index = None, return_label = False):
    labels = []
    if qual_index == []:
        return X
    if qual_index is None:
        qual_index = list(range(X.shape[-1]))
    # if type(X) == np.ndarray:
    #     temp = torch.from_numpy(X).detach().clone()
    temp = np.copy(X)
    if type(X) == torch.Tensor:
        temp = X.clone()
    if temp.ndim > 1:
        for j in qual_index:
            l = np.sort(np.unique(temp[..., j])).tolist()
            labels.append(l)
            #l =  torch.unique(temp[..., j], sorted = True).tolist()
            temp[..., j] = torch.tensor([*map(lambda m: l.index(m),temp[..., j])])
    else:
            l = torch.unique(temp, sorted = True)
            temp = torch.tensor([*map(lambda m: l.tolist().index(m), temp)])
    
    
    if temp.dtype == object:
        temp = temp.astype(float)
        if type(X) == np.ndarray:
            temp = torch.from_numpy(temp)
        
        if return_label:
            return temp, labels
        else:
            return temp
    else:
        if type(X) == np.ndarray:
            temp = torch.from_numpy(temp)
        if return_label:
            return temp, labels
        else:
            return temp
