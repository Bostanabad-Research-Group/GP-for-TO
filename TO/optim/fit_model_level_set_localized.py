import torch
import math
import numpy as np
from gpytorch.settings import cholesky_jitter
import matplotlib.pyplot as plt
import torch.optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from TO.utils_lcsmto import  plot_density_and_velocity_fields, save_loss_history_to_csv 
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 15
plt.rcParams['figure.dpi'] = 150



checkpoints=[10,1000,4000,10000,15000,20000]
lambdaa = 0.1
w_1,w_2,w_3, w_4, w_5 =0.01,0.01,1e2,8e3,1e4




def modified_sigmoid(x, alpha=12.0):
    """Compute the modified sigmoid function."""
    return 1 / (1 + torch.exp(-alpha * (x - 0.5)))


def heaviside(phi, epsilon=1e-1):
    # A smooth Heaviside: 0.5*(1 + tanh(phi/epsilon))
    return 0.5 * (1 + torch.tanh(phi / epsilon))



def projectDensity(x,b=16):
    nmr = np.tanh(0.5*b) + torch.tanh(b*(x-0.5))
    x = 0.5*nmr/np.tanh(0.5*b)
    return x

def compute_dynamic_weights_ic(pde_loss, bc_loss, ic_loss, model):
    """
    Compute dynamic weights for loss functions based on gradients.
    """
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    
    def compute_gradients(loss, scaling_factor):
        gradients = torch.autograd.grad(scaling_factor * loss, params_to_update, retain_graph=True, allow_unused=True)
        values = [p.reshape(-1).cpu().tolist() for p in gradients if p is not None]
        return torch.abs(torch.tensor([v for val in values for v in val]))
    
    delta_pde = compute_gradients(pde_loss, 1.0)
    delta_bc = compute_gradients(bc_loss, model.alpha)
    delta_ic = compute_gradients(ic_loss, model.beta)

    temp_bc = torch.max(delta_pde) / torch.mean(delta_bc)
    temp_ic = torch.max(delta_pde) / torch.mean(delta_ic)

    return (
        (1.0 - lambdaa) * model.alpha + lambdaa * temp_bc,
        (1.0 - lambdaa) * model.beta + lambdaa * temp_ic
    )

def compute_fvm_residuals_higher_order(u, v, p, f_x, f_y, nu, dx, dy):
    # Check the device of the input tensors and move other tensors to the same device
    device = u.device
    
    # Extend the fields with ghost cells to handle boundary conditions
    u_ext = torch.zeros((u.shape[0] + 4, u.shape[1] + 4), device=device)
    v_ext = torch.zeros((v.shape[0] + 4, v.shape[1] + 4), device=device)
    p_ext = torch.zeros((p.shape[0] + 4, p.shape[1] + 4), device=device)
    
    # Copy the interior
    u_ext[2:-2, 2:-2] = u
    v_ext[2:-2, 2:-2] = v
    p_ext[2:-2, 2:-2] = p
    
    # Apply ghost cells for no-slip boundary conditions
    u_ext[:2, :] = u_ext[2:4, :]
    u_ext[-2:, :] = u_ext[-4:-2, :]
    u_ext[:, :2] = u_ext[:, 2:4]
    u_ext[:, -2:] = u_ext[:, -4:-2]
    
    v_ext[:2, :] = v_ext[2:4, :]
    v_ext[-2:, :] = v_ext[-4:-2, :]
    v_ext[:, :2] = v_ext[:, 2:4]
    v_ext[:, -2:] = v_ext[:, -4:-2]
    
    # Laplacian of u and v (fourth-order central differences for second derivatives)
    u_xx = (-u_ext[4:, 2:-2] + 16 * u_ext[3:-1, 2:-2] - 30 * u_ext[2:-2, 2:-2] + 16 * u_ext[1:-3, 2:-2] - u_ext[:-4, 2:-2]) / (12 * dx**2)
    u_yy = (-u_ext[2:-2, 4:] + 16 * u_ext[2:-2, 3:-1] - 30 * u_ext[2:-2, 2:-2] + 16 * u_ext[2:-2, 1:-3] - u_ext[2:-2, :-4]) / (12 * dy**2)
    laplacian_u = u_xx + u_yy
    
    v_xx = (-v_ext[4:, 2:-2] + 16 * v_ext[3:-1, 2:-2] - 30 * v_ext[2:-2, 2:-2] + 16 * v_ext[1:-3, 2:-2] - v_ext[:-4, 2:-2]) / (12 * dx**2)
    v_yy = (-v_ext[2:-2, 4:] + 16 * v_ext[2:-2, 3:-1] - 30 * v_ext[2:-2, 2:-2] + 16 * v_ext[2:-2, 1:-3] - v_ext[2:-2, :-4]) / (12 * dy**2)
    laplacian_v = v_xx + v_yy
    
    # Gradients of p using fourth-order central difference
    grad_p_x = (-p_ext[4:, 2:-2] + 8 * p_ext[3:-1, 2:-2] - 8 * p_ext[1:-3, 2:-2] + p_ext[:-4, 2:-2]) / (12 * dx)
    grad_p_y = (-p_ext[2:-2, 4:] + 8 * p_ext[2:-2, 3:-1] - 8 * p_ext[2:-2, 1:-3] + p_ext[2:-2, :-4]) / (12 * dy)
    
    # Momentum equation residuals
    residual_u = -nu * laplacian_u + grad_p_x + f_x
    residual_v = -nu * laplacian_v + grad_p_y + f_y
    
    # Mass conservation (divergence of velocity) using central differences
    div_u = (u_ext[2:-2, 2:-2] - u_ext[1:-3, 2:-2]) / dx
    div_v = (v_ext[2:-2, 2:-2] - v_ext[2:-2, 1:-3]) / dy
    
    u_y = (u_ext[2:-2, 2:-2] - u_ext[2:-2, 1:-3]) / dy
    v_x = (v_ext[2:-2, 2:-2] - v_ext[1:-3, 2:-2]) / dx
    residual_mass = div_u + div_v
    

    
    return residual_u.reshape(-1), residual_v.reshape(-1), residual_mass.reshape(-1), div_u.reshape(-1), div_v.reshape(-1), u_y.reshape(-1), v_x.reshape(-1)



def compute_autograd_derivatives(model, z_column, collocation_x, grad_order=1):
    """
    Compute first and second-order derivatives using PyTorch autograd.
    :param model: Model to train
    :param z_column: Column of `z_all` for differentiation
    :param collocation_x: Input coordinates
    :param grad_order: Order of gradients (1 for first, 2 for second)
    :return: First and second-order derivatives as tensors
    """
    model.train()
    grad_1 = torch.autograd.grad(z_column, collocation_x, torch.ones_like(z_column), create_graph=True)[0]
    if grad_order > 1:
        grad_2_x = torch.autograd.grad(grad_1[:, 0], collocation_x, torch.ones_like(grad_1[:, 0]), create_graph=True)[0][:, 0]
        grad_2_y = torch.autograd.grad(grad_1[:, 1], collocation_x, torch.ones_like(grad_1[:, 1]), create_graph=True)[0][:, 1]
        return grad_1[:, 0], grad_1[:, 1], grad_2_x, grad_2_y
    return grad_1[:, 0], grad_1[:, 1], None, None


def loss_volume(y, gamma=0.5):
    """
    Compute the volume loss as the squared difference between the mean of y and gamma.
    """
    mean_y = torch.mean(y)
    return torch.square(mean_y - gamma)



def dissipated_power_with_body_force(u, u_x, v_y, u_y, v_x,f_masked):
    """
    Compute the total dissipated power based on the input velocity gradients and displacements.
    """
    # First part of the dissipated power
    p1 = (u_x**2 + v_y**2 + u_y**2 + v_x**2)#.sum(dim=1, keepdim=True)
    
    # Second part of the dissipated power
    u2 = (u[:, :2]**2).sum(dim=1, keepdim=True)
    p2 = alpha(u[:, 3:]) * u2
    
    f_d=u[:, 0].reshape(-1,1)*f_masked
    return 0.5 * (p1 + p2)- f_d

def dissipated_power(u, u_x, v_y, u_y, v_x ):
    """
    Compute the total dissipated power based on the input velocity gradients and displacements.
    """
    # First part of the dissipated power
    p1 = (u_x**2 + v_y**2 + u_y**2 + v_x**2)#.sum(dim=1, keepdim=True)
    
    # Second part of the dissipated power
    u2 = (u[:, :2]**2).sum(dim=1, keepdim=True)
    p2 = alpha(u[:, 3:]) * u2
    
    return 0.5 * (p1 + p2)


def alpha(rho):
    """
    Compute the alpha parameter based on rho.
    """
    alpha_max = 2.5e4
    alpha_min = 0#2.5e-4
    q = 0.1
    return alpha_max + (alpha_min - alpha_max) * rho * (1 + q) / (rho + q)


def moving_average(data, window_size):
    # Convert data to a 1D tensor
    data = torch.tensor(data, dtype=torch.float32).flatten()
    
    # Create a zero tensor with the same number of dimensions as data
    zero_tensor = torch.zeros((1,), dtype=data.dtype, device=data.device)
    
    # Concatenate and compute the cumulative sum
    cumsum = torch.cumsum(torch.cat((zero_tensor, data)), dim=0)
    
    # Calculate the moving average
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


threshold_rho = 1e-5
max_stagnation = 100
stagnation_count = 0
rho_prev = None





added_body_force=-1125#1687.5#562.5,

def calculate_loss_multioutput(model_list, iteration, diff_method='Numerical', title="default",rho_prev=None,stagnation_count=0,diff_rho_list=[],smoothed_rho_list=[],localized_weight=True,volume_fraction=1/3,ratio= 1.0):
    """
    Calculate loss for multi-output models based on PDE, BC, and IC residuals.
    """
    
    dx = model_list[0].dx
    dy = model_list[0].dy
    Nx = model_list[0].Nx
    Ny = model_list[0].Ny
    # mask_col = model_list[0].mask_col
    collocation_x = model_list[0].collocation_x.clone()

    # Clone collocation points
    collocation_x = model_list[0].collocation_x.clone()

    # Compute mean values at collocation points
    m_col = model_list[0].mean_module_NN_All(collocation_x)

    gamma_end = volume_fraction # next((val for key, val in gamma_values.items() if key in title), 0.5)

    # m_col[:,3]=torch.tanh(m_col[:,3])    
    
    # Evaluate g_uvp and perform Cholesky decomposition if not already done
    # if model_list[0].g_uvp is None:
    for i, model in enumerate(model_list):
        model.g_uvp = model.covar_module(model.train_inputs[0], collocation_x).evaluate()

    with cholesky_jitter(1e-3):  # Ensure numerical stability with jitter
        for model in model_list:
            model.chol_decomp = model.covar_module(model.train_inputs[0]).cholesky()

    # Calculate for K_inv offsets
    K_inv_offsets = []
    for i, model in enumerate(model_list):
        # if i==3:
        #     target_offset = model.train_targets.unsqueeze(-1) - torch.tanh(model.mean_module_NN_All(model.train_inputs[0])[:, i]).unsqueeze(-1)
        # else:
        target_offset = model.train_targets.unsqueeze(-1) - model.mean_module_NN_All(model.train_inputs[0])[:, i].unsqueeze(-1)
        K_inv_offsets.append(model.chol_decomp._cholesky_solve(target_offset))

    # Compute values for density, velocity, and pressure

    u = (m_col[:, 0].unsqueeze(-1) + model_list[0].g_uvp.t() @ K_inv_offsets[0]).squeeze(-1)
    v = (m_col[:, 1].unsqueeze(-1) + model_list[1].g_uvp.t() @ K_inv_offsets[1]).squeeze(-1)
    p = (m_col[:, 2].unsqueeze(-1) + model_list[2].g_uvp.t() @ K_inv_offsets[2]).squeeze(-1)

    ro_tensor = (m_col[:, 3].unsqueeze(-1) + model_list[3].g_uvp.t() @ K_inv_offsets[3]).squeeze(-1)
    

    ro = heaviside(ro_tensor)
    
    # --- Existing Code ---

    if localized_weight and iteration >= 15000:
        if iteration in checkpoints: #

            # Define boundary detection threshold
            boundary_threshold = 0.01#0.1  
            weight_on_boundary = 2  # User-defined weight for points exactly on the boundary

            # **Step 1: Extract Boundary Points (Zero Level-Set)**
            collocation_x_NPP = model_list[0].collocation_x.clone().detach().cpu().numpy()  # (N, 2)
            phi_values = ro_tensor.clone().detach().cpu().numpy().flatten()  # φ values

            # Extract x and y coordinates
            x_coords = collocation_x_NPP[:, 0]
            y_coords = collocation_x_NPP[:, 1]

            # Create grid for contour extraction
            xi = np.linspace(min(x_coords), max(x_coords), 200)
            yi = np.linspace(min(y_coords), max(y_coords), 200)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x_coords, y_coords), phi_values, (X, Y), method='cubic')  # Interpolated φ

            # Extract Zero Level-Set (Boundary Contour)
            contour_lines = plt.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
            plt.close()  # Close to prevent duplicate plots

            # **Extract boundary points from contour**
            boundary_points = []
            for collection in contour_lines.collections:
                for path in collection.get_paths():
                    boundary_points.extend(path.vertices)
            boundary_points = np.array(boundary_points) if boundary_points else np.empty((0, 2))

            # **Initialize boundary_x and boundary_y to avoid UnboundLocalError**

            # **Find collocation points nearest to the boundary**
            if boundary_points.shape[0] > 0:
                tree = KDTree(boundary_points)  # Use KDTree for efficient nearest neighbor search
                distances, _ = tree.query(collocation_x_NPP)  # Find nearest distances

                # Select points within the boundary threshold (for plotting, if desired)

            # **Step 2: Compute Normal Vectors at Boundary**
            dy_phi, dx_phi = np.gradient(Z, yi, xi)  # Compute ∇φ (order: dy, dx)
            grad_magnitude = np.sqrt(dx_phi**2 + dy_phi**2) + 1e-9  # Avoid division by zero
            normal_x = dx_phi / grad_magnitude
            normal_y = dy_phi / grad_magnitude

            # **Interpolate normal vectors at boundary points**
            boundary_normal_x = griddata(
                (X.flatten(), Y.flatten()), normal_x.flatten(), (boundary_points[:, 0], boundary_points[:, 1]), method='linear'
            )
            boundary_normal_y = griddata(
                (X.flatten(), Y.flatten()), normal_y.flatten(), (boundary_points[:, 0], boundary_points[:, 1]), method='linear'
            )
            boundary_normal_x = np.nan_to_num(boundary_normal_x, nan=0.0)
            boundary_normal_y = np.nan_to_num(boundary_normal_y, nan=0.0)

            # **Step 4: Create Weighted Mask Based on Distance to the Boundary**
            # For far away points, we assign a weight of 1; for points exactly on the boundary, the weight is set to weight_on_boundary.
            weighted_mask = 1*np.ones_like(phi_values, dtype=float)  # Default weight = 1

            if boundary_points.shape[0] > 0:
                tree = KDTree(boundary_points)  # Use KDTree for fast nearest neighbor search
                distances, _ = tree.query(collocation_x_NPP)  # Compute distances from each collocation point to the boundary

                # For points within the boundary_threshold, linearly interpolate the weight:
                near_boundary = distances < boundary_threshold
                weighted_mask[near_boundary] = weight_on_boundary - (weight_on_boundary - 1) * (distances[near_boundary] / boundary_threshold)

            model_list[0].mask_tensor= torch.tensor(weighted_mask, dtype=torch.float32, device=ro_tensor.device)

    # ######################################################

    
    z_all = torch.cat((u.unsqueeze(1), v.unsqueeze(1), p.unsqueeze(1), ro.unsqueeze(1)), dim=1)
    f = alpha(z_all[:, 3:]) * z_all[:, :2]
    fx, fy = f[:, :1], f[:, 1:]

        
    rho_prev = ro.clone()
    
    ## Body Force
    if 'pipe_with_force_term' in title:
        mask_temp=collocation_x.clone()#-torch.tensor([0.5,1/3]).to("cuda")
        center=torch.tensor([0.5,1/3]).to("cuda")
        mask=torch.where(((mask_temp[:,0]-center[0])**2 + (mask_temp[:,1]-center[1])**2)<(1/12)**2 )[0]        
        f_masked = torch.zeros_like(fx)
        f_masked[mask]=added_body_force
        fx-=f_masked

    # Compute residuals
    if diff_method == 'Numerical':
        dx, dy = 0.01, 0.01
        nx, ny = int(ratio*100), 100
        u, v, p = [arr.reshape(nx, ny) for arr in (u, v, p)]
        fx, fy = [arr.reshape(nx, ny) for arr in (fx, fy)]
        residuals = compute_fvm_residuals_higher_order(u, v, p, fx, fy, nu=1, dx=dx, dy=dy)
        residual_pde1, residual_pde2, residual_pde3, u_x, v_y, u_y, v_x = residuals
        residual_pde1, residual_pde2, residual_pde3 = [res * w for res, w in zip(residuals[:3], (w_1, w_2, w_3))]
        u, v, p = [arr.reshape(-1) for arr in (u, v, p)]
        fx, fy = [arr.reshape(-1) for arr in (fx, fy)]
    else:
        u_x, u_y, u_xx, u_yy = compute_autograd_derivatives(model_list[0], z_all[:, 0], collocation_x, grad_order=2)
        v_x, v_y, v_xx, v_yy = compute_autograd_derivatives(model_list[1], z_all[:, 1], collocation_x, grad_order=2)
        p_x, p_y, _, _ = compute_autograd_derivatives(model_list[2], z_all[:, 2], collocation_x, grad_order=1)
        residual_pde1 = (- (u_xx + u_yy) + p_x + fx[:, 0]) * w_1
        residual_pde2 = (- (v_xx + v_yy) + p_y + fy[:, 0]) * w_2
        residual_pde3 = (u_x + v_y) * w_3

    # Plot fields at checkpoints
    if iteration in checkpoints:
        plot_density_and_velocity_fields(u, v, p, ro , collocation_x, iteration)



    losses = [torch.mean(res**2) for res in (residual_pde1, residual_pde2, residual_pde3)]

    loss_pde1, loss_pde2, loss_pde3 = losses

    dp_loss = dissipated_power(z_all, u_x.reshape(-1, 1), v_y.reshape(-1, 1), u_y.reshape(-1, 1), v_x.reshape(-1, 1))
    dp_loss_val=torch.mean(dp_loss)*w_5



    # Hyperparameters
    gamma_start = 1.0
    target_iteration = 4000
    p_c = 1         # exponent for polynomial (increase for more abrupt drop)
    block_size = 200

    # Compute how many blocks in total from 0 to target_iteration
    max_block = target_iteration // block_size

    # Determine current block index
    current_block = iteration // block_size

    # Clamp the current_block so it doesn't go beyond max_block
    if current_block > max_block:
        current_block = max_block

    # Now compute the fraction of progress (in terms of blocks, not individual iterations)
    block_frac = current_block / max_block  # goes from 0.0 to 1.0

    # Apply polynomial schedule using the block-based fraction
    gamma = gamma_end + (gamma_start - gamma_end) * ((1.0 - block_frac) ** p_c)

    # Use gamma
    volfrac = torch.mean(z_all[:, 3:])
    vol_loss = torch.square((volfrac / gamma) - 1.0) * w_4

    if iteration % 1000 == 0:
        print(f"Iteration {iteration}, Block {current_block}, gamma={gamma:.4f}, vol_loss={vol_loss/w_4:.6f}, volfrac={volfrac:.4f}, dp_loss={ratio*dp_loss_val/w_5:.4f}")

    ### From Alex code ###z_all_for_dp
    volfrac = torch.mean(z_all[:, 3:]); # get the current volume fraction
    vol_loss  = torch.square((volfrac / gamma) - 1.0)* w_4 #+ 1e-5
    
    # Return final losses
    return loss_pde1, loss_pde2, loss_pde3, dp_loss_val, vol_loss,rho_prev,stagnation_count,diff_rho_list,smoothed_rho_list,residual_pde1,residual_pde2,residual_pde3




def find_TO_level_set_localized(
    model_list,
    lr_default: float = 0.01,
    num_iter: int = 500,
    title: str = 'default',
    X_col_all: dict= {},
    diff_method: str = 'Numerical',
    localized_weight=True,
    volume_fraction=1/3,
    ratio: float =1
) -> float:
    """
    Train models to minimize total loss using dynamic weights and record progress.

    Args:
        model_list: List of models to be trained.
        lr_default: Default learning rate for the optimizer.
        num_iter: Number of training iterations.
        title: Title for specific settings.
        diff_method: Differentiation method ('Numerical' or 'Autograd').

    Returns:
        loss history
    """
    # Use the first model as a reference for shared NN
    model_ref = model_list[0]
    for model in model_list:
        model.mean_module_NN_All = model_ref.mean_module_NN_All
        model.train()

    # Initialize variables
    loss_total, loss_hist, loss_hist_total = [], [], []
    loss_pde1_hist, loss_pde2_hist, loss_pde3_hist = [], [], []
    loss_dp_hist, vol_loss_hist = [], []
    scaled_loss_pde1_hist, scaled_loss_pde2_hist = [], []
    scaled_loss_pde3_hist, scaled_loss_dp_hist, scaled_vol_loss_hist = [], [], []
    weights = {'alpha': [], 'beta': [], 'mu_p': []}
    result_loss_thetas = {'largest_eigval_hist': [], 'condition_number_hist': [], 'Hessian': [], 'grads': []}
    GP_NN_hist, NN_hist, time_hist = [], [], []
    dynamic_weights = True
    sigma_1, beta_1, alph_1 = 10, 10, 0.1
    f_inc, mu_F = math.inf, 1

    # Initialize model-specific parameters
    model1 = model_list[0]
    model1.alpha, model1.beta, model1.theta = 1, 1, 1

    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model_ref.parameters(), lr=lr_default)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=np.linspace(0, num_iter, 4).tolist(),
        gamma=0.75
    )
    rho_prev=None
    stagnation_count=0
    diff_rho_list=[]
    smoothed_rho_list=[]
    
    
    
    lagrang_pde1,lagrang_pde2,lagrang_pde3=[torch.zeros_like(model_list[0].collocation_x[:,0]).clone().detach()],[torch.zeros_like(model_list[0].collocation_x[:,0].clone().detach())],[torch.zeros_like(model_list[0].collocation_x[:,0].clone().detach())]
    lagrang_vf=[torch.zeros_like(model_list[0].collocation_x[0,0]).clone().detach()]
    # Generate random integers
    index_CP = np.random.randint(0, 50, size=num_iter)

    # Training loop
    for j in tqdm(range(num_iter), desc='Epoch', position=0, leave=True):
        optimizer.zero_grad()




        index = index_CP[j]
        model_list[0].dx = X_col_all[index]['dx']
        model_list[0].dy = X_col_all[index]['dy']
        model_list[0].Nx = X_col_all[index]['Nx']
        model_list[0].Ny = X_col_all[index]['Ny']
        model_list[0].mask_col = X_col_all[index]['mask_col']
        model_list[0].collocation_x = X_col_all[index]['X_col']
        
        # Calculate losses
        loss_pde1, loss_pde2, loss_pde3, loss_dp, vol_loss,rho_prev,stagnation_count,diff_rho_list,smoothed_rho_list,residual_pde1,residual_pde2,residual_pde3 = calculate_loss_multioutput(
            model_list, j, diff_method=diff_method, title=title,rho_prev=rho_prev,stagnation_count=stagnation_count,diff_rho_list=diff_rho_list,smoothed_rho_list=smoothed_rho_list,localized_weight=localized_weight,volume_fraction=volume_fraction,ratio= ratio)
        loss_pde = loss_pde1 + loss_pde3

        # Dynamic weight adjustments
        if dynamic_weights:
            alpha, beta = compute_dynamic_weights_ic(
                loss_pde1 + loss_pde2, loss_pde3, vol_loss, model1
            )
            if all(torch.is_tensor(val) and not (torch.isnan(val).any() or torch.isinf(val).any()) for val in [alpha, beta]):
                model1.alpha, model1.beta = alpha, beta

            weights['alpha'].append(alpha.detach().cpu().item())
            weights['beta'].append(beta.detach().cpu().item())
            weights['mu_p'].append(mu_F)
            
            
            
            loss = 1*loss_dp + mu_F * (
                loss_pde1 + loss_pde2 + model1.alpha * loss_pde3 + model1.beta * vol_loss
            )#+ lagrang_loss_pde1 + lagrang_loss_pde1 
            if (j + 1) % 50 == 0:
                mu_F = min(mu_F * 1.05, 5e2)
        else:
            alph_1 = 1e-6 / (j + 1)
            beta_1, sigma_1 = 2 * (j + 1), 5 * (j + 1)
            loss = sigma_1 * loss_pde + alph_1 * loss_dp + beta_1 * vol_loss

        # Record loss history
        loss_total.append(loss.item())
        loss_pde1_hist.append((1/(w_1**2)) * loss_pde1.item())
        loss_pde2_hist.append((1/(w_2**2)) * loss_pde2.item())
        loss_pde3_hist.append((1/w_3**2) * loss_pde3.item())
        loss_dp_hist.append((1/w_5) * loss_dp.item()*3.) #remove_3
        vol_loss_hist.append((1/w_4) * vol_loss.item())
        scaled_loss_pde1_hist.append(loss_pde1.item())
        scaled_loss_pde2_hist.append(loss_pde2.item())
        scaled_loss_pde3_hist.append(model1.alpha.item() * loss_pde3.item())
        scaled_loss_dp_hist.append(loss_dp.item())
        scaled_vol_loss_hist.append(model1.beta.item() * vol_loss.item())

        # Plot loss history at checkpoints
        if j in checkpoints:
            # plot_lafgrangian(model_list[0].collocation_x.detach(), j,lagrang_pde1_temp, lagrang_pde2_temp, lagrang_pde3_temp)
            save_loss_history_to_csv(
                loss_total,
                [loss_pde1_hist, scaled_loss_pde1_hist],
                [loss_pde2_hist, scaled_loss_pde2_hist],
                [loss_pde3_hist, scaled_loss_pde3_hist],
                [loss_dp_hist, scaled_loss_dp_hist],
                [vol_loss_hist, scaled_vol_loss_hist],
                j , fname =title,
            )

        # Backpropagation and optimizer step
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # Update progress description
        # desc = f'Epoch {j} - loss {loss.item():.6f}'
        # tqdm.write(desc)
        loss_hist.append(loss.item())

    # Store final loss history
    loss_hist_total = loss_hist
    #plt.show(block=True)

    return loss_hist_total[0]

