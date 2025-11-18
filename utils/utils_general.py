import torch
import numpy as np
import json
import random

# def set_seed(seed):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
def set_seed0(seed):
    random.seed(seed)                
    np.random.seed(seed)             
    torch.manual_seed(seed)          # PyTorch CPU seed
    if torch.cuda.is_available():    # If CUDA is available
        torch.cuda.manual_seed(seed)            
        torch.cuda.manual_seed_all(seed)

def set_seed(seed):
    random.seed(seed)                
    np.random.seed(seed)             
    torch.manual_seed(seed)          # PyTorch CPU seed
    if torch.cuda.is_available():    # If CUDA is available
        torch.cuda.manual_seed(seed)            
        torch.cuda.manual_seed_all(seed)        
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False
    # Set deterministic behavior for other PyTorch operations 
    torch.use_deterministic_algorithms(True)

def load_job_tensors_from_json(file_path, job_name):
    """
    Loads tensors for a specific job from a JSON file and converts them back to PyTorch tensors.
    
    Args:
    - file_path (str): Path to the JSON file.
    - job_name (str): The name of the job to extract tensors for (e.g., 'Job-1').

    Returns:
    - dict: A dictionary where keys are tensor names and values are PyTorch tensors.
    """
    with open(file_path, 'r') as json_file:
        jobs_dict_serializable = json.load(json_file)
    
    # Extract the dictionary for the specific job
    if job_name in jobs_dict_serializable:
        job_tensors_serializable = jobs_dict_serializable[job_name]
        # Convert lists back to tensors
        job_tensors = {key: torch.tensor(value) for key, value in job_tensors_serializable.items()}
        return job_tensors
    else:
        raise ValueError(f"No tensors found for {job_name}")

def get_tkwargs():
    """
    Returns a dictionary with the device and dtype configuration
    for PyTorch, prioritizing CUDA, then MPS (Mac GPU), and finally CPU.
    """
    # Determine the device: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use NVIDIA GPU if available
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")  # Use Mac GPU (MPS) if available
    else:
        device = torch.device("cpu")  # Fallback to CPU
    
    # Define tkwargs with selected device and dtype
    tkwargs = {
        "dtype": torch.float32,
        "device": device,
    }
    
    #print(f"Using device: {device}")
    return tkwargs 

def central_diff_2nd(f, dx, dy):
    # f is a 2D torch tensor.
    # find the derivative using the 2nd order accuracy central difference method
    # First-order derivatives with second-order accuracy
    df_dx = (torch.roll(f, shifts=-1, dims=1) - torch.roll(f, shifts=1, dims=1)) / (2 * dx)
    df_dy = (torch.roll(f, shifts=-1, dims=0) - torch.roll(f, shifts=1, dims=0)) / (2 * dy)
    
    # Second-order derivatives with second-order accuracy
    d2f_dx2 = (torch.roll(f, shifts=-1, dims=1) - 2 * f + torch.roll(f, shifts=1, dims=1)) / (dx ** 2)
    d2f_dy2 = (torch.roll(f, shifts=-1, dims=0) - 2 * f + torch.roll(f, shifts=1, dims=0)) / (dy ** 2)
    
    # Mixed second derivative with second-order accuracy
    d2f_dxdy = (torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-1, dims=1)
                - torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=1, dims=1)
                - torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-1, dims=1)
                + torch.roll(torch.roll(f, shifts=1, dims=0), shifts=1, dims=1)) / (4 * dx * dy)
    
    return [df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]

def central_diff_4th(f, dx, dy):
    # f is a 2D torch tensor.
    # find the derivative using the 2nd order accuracy central difference method
    # First-order derivatives with fourth-order accuracy
    df_dx = (-torch.roll(f, shifts=-2, dims=1) 
             + 8 * torch.roll(f, shifts=-1, dims=1)
             - 8 * torch.roll(f, shifts=1, dims=1) 
             + torch.roll(f, shifts=2, dims=1)) / (12 * dx)
    
    df_dy = (-torch.roll(f, shifts=-2, dims=0) 
             + 8 * torch.roll(f, shifts=-1, dims=0)
             - 8 * torch.roll(f, shifts=1, dims=0) 
             + torch.roll(f, shifts=2, dims=0)) / (12 * dy)
    
    # Second-order derivatives with fourth-order accuracy
    d2f_dx2 = (-torch.roll(f, shifts=-2, dims=1) 
               + 16 * torch.roll(f, shifts=-1, dims=1)
               - 30 * f
               + 16 * torch.roll(f, shifts=1, dims=1) 
               - torch.roll(f, shifts=2, dims=1)) / (12 * dx**2)
    
    d2f_dy2 = (-torch.roll(f, shifts=-2, dims=0) 
               + 16 * torch.roll(f, shifts=-1, dims=0)
               - 30 * f
               + 16 * torch.roll(f, shifts=1, dims=0) 
               - torch.roll(f, shifts=2, dims=0)) / (12 * dy**2)
    
    # Mixed second derivative with fourth-order accuracy
    d2f_dxdy = (torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=-2, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-2, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-2, dims=1)
                - torch.roll(torch.roll(f, shifts=2, dims=0), shifts=-2, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=-1, dims=1)
                + 64 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-1, dims=1)
                - 64 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-1, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=2, dims=0), shifts=-1, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=1, dims=1)
                - 64 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=1, dims=1)
                + 64 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=1, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=2, dims=0), shifts=1, dims=1)
                - torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=2, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=2, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=2, dims=1)
                + torch.roll(torch.roll(f, shifts=2, dims=0), shifts=2, dims=1)) / (144 * dx * dy)
    
    return [df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]

def compute_dynamic_weights(ref_loss,target_loss,lambdaa,model,control):
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    delta_ref_teta = torch.autograd.grad(ref_loss, params_to_update,  retain_graph=True)
    values = [p.reshape(-1,).cpu().tolist() for p in delta_ref_teta if p is not None]
    delta_ref_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

    delta_target_teta = torch.autograd.grad(target_loss, params_to_update,  retain_graph=True)
    values = [p.reshape(-1,).cpu().tolist() for p in delta_target_teta if p is not None]
    delta_target_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

    temp1 = torch.mean(delta_ref_teta_abs) / torch.mean(delta_target_teta_abs)
    if control == 1:
        return (1.0 - lambdaa) * model.alpha + lambdaa * temp1
    else:
        return (1.0 - lambdaa) * model.beta + lambdaa * temp1

lambdaa = 0.1
def compute_dynamic_weight_2(ref_loss,target_loss,model):
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    delta_ref_teta = torch.autograd.grad(ref_loss, params_to_update,  retain_graph=True,allow_unused=True)
    values = [p.reshape(-1,).cpu().tolist() for p in delta_ref_teta if p is not None]
    delta_ref_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

    delta_target_teta = torch.autograd.grad(target_loss, params_to_update,  retain_graph=True,allow_unused=True)
    values = [p.reshape(-1,).cpu().tolist() for p in delta_target_teta if p is not None]
    delta_target_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

    temp1 = torch.mean(delta_ref_teta_abs) / torch.mean(delta_target_teta_abs)
    
    return (1.0 - lambdaa) * model.alpha + lambdaa * temp1

def projectDensity(x,b=16):
    nmr = np.tanh(0.5*b) + torch.tanh(b*(x-0.5))
    x = 0.5*nmr/np.tanh(0.5*b)
    return x

def dynamic_binarize_density(density_field, tolerance=1e-6):
    # Original average density
    avg_density = np.mean(density_field)
    
    # Initialize threshold
    low, high = 0.0, 1.0
    
    while high - low > tolerance:
        threshold = (low + high) / 2
        
        # Binarize using the current threshold
        binary_field = np.where(density_field > threshold, 1, 0)

        # Compute the average density of the binary field
        binary_avg_density = np.mean(binary_field)
        
        # Adjust the threshold
        if binary_avg_density > avg_density:
            low = threshold  # Threshold is too low, increase it
        else:
            high = threshold  # Threshold is too high, decrease it
    
    # Final binarized field
    binary_field = np.where(density_field > threshold, 1, 0)
    
    return binary_field, threshold
