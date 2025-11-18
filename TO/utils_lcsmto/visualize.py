import matplotlib.pyplot as plt
import os
from matplotlib import ticker
import numpy as np
from scipy.interpolate import griddata
from matplotlib.patches import Polygon as MplPolygon
import torch
import pandas as pd
from datetime import datetime

def format_axis_ticks(ax):
    """Format axis ticks for consistent appearance."""
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='major', labelsize=10)

def dump_predictions_and_residuals(u, v, p, ro, collocation_x, iteration,
                                  residual_pde1, residual_pde2, residual_pde3, 
                                  w_pde1, w_pde2, w_pde3, output_dir="results_csv" , fname = 'default'):
    """
    Dump the predicted fields (U, V, P, ro) and residuals of the PDEs to CSV files.
    
    :param u, v, p, ro: torch.Tensors, predicted fields
    :param collocation_x: torch.Tensor, collocation points in 2D space (x, y)
    :param iteration: int, current iteration
    :param residual_pde1, residual_pde2, residual_pde3: torch.Tensors, residuals
    :param w_pde1, w_pde2, w_pde3: float, PDE weights
    :param output_dir: str, directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    ro_np = ro.detach().cpu().numpy()
    collocation_x_np = collocation_x.detach().cpu().numpy()
    
    # Compute normalized residuals
    res_pde1_np = (1 / w_pde1) * residual_pde1.detach().cpu().numpy()
    res_pde2_np = (1 / w_pde2) * residual_pde2.detach().cpu().numpy()
    res_pde3_np = (1 / w_pde3) * residual_pde3.detach().cpu().numpy()
    
    # Calculate mean squared residuals (for reference)
    mean_sq_res1 = np.mean(res_pde1_np**2)
    mean_sq_res2 = np.mean(res_pde2_np**2)
    mean_sq_res3 = np.mean(res_pde3_np**2)
    
    # Create DataFrames
    # Predictions DataFrame
    predictions_df = pd.DataFrame({
        'x': collocation_x_np[:, 0],
        'y': collocation_x_np[:, 1],
        'u': u_np,
        'v': v_np,
        'p': p_np,
        'density': ro_np
    })
    
    # Residuals DataFrame
    residuals_df = pd.DataFrame({
        'x': collocation_x_np[:, 0],
        'y': collocation_x_np[:, 1],
        'residual_pde1': res_pde1_np,
        'residual_pde2': res_pde2_np,
        'residual_pde3': res_pde3_np
    })
    
    # Summary DataFrame (contains statistics only)
    summary_df = pd.DataFrame({
        'iteration': [iteration],
        'mean_squared_residual_pde1': [mean_sq_res1],
        'mean_squared_residual_pde2': [mean_sq_res2],
        'mean_squared_residual_pde3': [mean_sq_res3],
        'weight_pde1': [w_pde1],
        'weight_pde2': [w_pde2],
        'weight_pde3': [w_pde3]
    })
    
    # Save to CSV files
    predictions_file = os.path.join(output_dir, f"{fname}_predictions_iteration_{iteration}.csv")
    residuals_file = os.path.join(output_dir, f"{fname}_residuals_iteration_{iteration}.csv")
    summary_file = os.path.join(output_dir, f"{fname}_summary_iteration_{iteration}.csv")
    
    predictions_df.to_csv(predictions_file, index=False)
    residuals_df.to_csv(residuals_file, index=False)
    summary_df.to_csv(summary_file, index=False)
    
    # Also create a single combined file for easier reference
    combined_df = predictions_df.copy()
    combined_df['residual_pde1'] = res_pde1_np
    combined_df['residual_pde2'] = res_pde2_np
    combined_df['residual_pde3'] = res_pde3_np
    
    combined_file = os.path.join(output_dir, f"{fname}_combined_data_iteration_{iteration}.csv")
    combined_df.to_csv(combined_file, index=False)
    
    print(f"Data saved to CSV files in {output_dir}:")
    print(f"  - {predictions_file}")
    print(f"  - {residuals_file}")
    print(f"  - {summary_file}")
    print(f"  - {combined_file}")
    
    # Return the filenames for reference
    return {
        'predictions': predictions_file,
        'residuals': residuals_file,
        'summary': summary_file,
        'combined': combined_file
    }

def plot_predictions_and_residuals(u, v, p, ro, collocation_x, iteration,
                                   residual_pde1, residual_pde2, residual_pde3, 
                                   w_pde1, w_pde2, w_pde3):
    """
    Plot the predicted fields (U, V, P, ro) and residuals of the PDEs.
    """
    def plot_field(ax, x, y, field, title, levels):
        contour = ax.tricontourf(x, y, field, levels=levels, cmap='jet')
        plt.colorbar(contour, ax=ax)
        ax.set_title(title, usetex=True, fontsize=14, pad=10)
        ax.set_xlabel(r"$x$", usetex=True, fontsize=12)
        ax.set_ylabel(r"$y$", usetex=True, fontsize=12)
        ax.grid(True)
        format_axis_ticks(ax)

    # Convert tensors to numpy arrays
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    ro_np = ro.detach().cpu().numpy()
    collocation_x_np = collocation_x.detach().cpu().numpy()

    # Plot predicted fields
    # fig, axs = plt.subplots(1, 4, figsize=(18, 6))
    # plot_field(axs[0], collocation_x_np[:, 0], collocation_x_np[:, 1], u_np, 
    #            rf"$U$: Predicted Mean (Iteration {iteration})", 
    #            np.linspace(np.min(u_np), np.max(u_np), 2000))
    # plot_field(axs[1], collocation_x_np[:, 0], collocation_x_np[:, 1], v_np, 
    #            rf"$V$: Predicted Mean (Iteration {iteration})", 
    #            np.linspace(np.min(v_np), np.max(v_np), 2000))
    # plot_field(axs[2], collocation_x_np[:, 0], collocation_x_np[:, 1], p_np, 
    #            rf"$P$: Predicted Mean (Iteration {iteration})", 
    #            np.linspace(np.min(p_np), np.max(p_np), 2000))
    # plot_field(axs[3], collocation_x_np[:, 0], collocation_x_np[:, 1], ro_np, 
    #            rf"$\rho$: Predicted Mean (Iteration {iteration})", 
    #            np.linspace(np.min(ro_np), np.max(ro_np), 2000))
    # plt.tight_layout()
    # plt.pause(0.05)

    # Compute normalized residuals
    res_pde1_np = (1 / w_pde1) * residual_pde1.detach().cpu().numpy()
    res_pde2_np = (1 / w_pde2) * residual_pde2.detach().cpu().numpy()
    res_pde3_np = (1 / w_pde3) * residual_pde3.detach().cpu().numpy()

    # Plot residuals
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plot_field(axs[0], collocation_x_np[:, 0], collocation_x_np[:, 1], res_pde1_np, 
               rf"Loss PDE 1 at Iteration {iteration} is {np.mean(res_pde1_np**2)}", 
               np.linspace(np.min(res_pde1_np), np.max(res_pde1_np), 2000))
    plot_field(axs[1], collocation_x_np[:, 0], collocation_x_np[:, 1], res_pde2_np, 
               rf"Loss PDE 1 at Iteration {iteration} is {np.mean(res_pde2_np**2)}", 
               np.linspace(np.min(res_pde2_np), np.max(res_pde2_np), 2000))
    plot_field(axs[2], collocation_x_np[:, 0], collocation_x_np[:, 1], res_pde3_np, 
               rf"Loss PDE 1 at Iteration {iteration} is {np.mean(res_pde3_np**2)})", 
               np.linspace(np.min(res_pde3_np), np.max(res_pde3_np), 2000))
    plt.tight_layout(pad=2.0)
    # plt.pause(0.05)
    
    # Specify the file path and save the figure
    file_path = f"results_localized/Residual_iteration_{iteration}.tiff"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, format='tiff', dpi=200)
    # plt.show(block=False)


def plot_lafgrangian(collocation_x, iteration,
                                   residual_pde1, residual_pde2, residual_pde3):
    """
    Plot the predicted fields (U, V, P, ro) and residuals of the PDEs.
    """
    def plot_field(ax, x, y, field, title, levels):
        contour = ax.tricontourf(x, y, field, levels=levels, cmap='jet')
        plt.colorbar(contour, ax=ax)
        ax.set_title(title, usetex=True, fontsize=14, pad=10)
        ax.set_xlabel(r"$x$", usetex=True, fontsize=12)
        ax.set_ylabel(r"$y$", usetex=True, fontsize=12)
        ax.grid(True)
        format_axis_ticks(ax)

    # Convert tensors to numpy arrays
    collocation_x_np = collocation_x.detach().cpu().numpy()

    # Compute normalized residuals
    res_pde1_np = residual_pde1.detach().cpu().numpy()
    res_pde2_np = residual_pde2.detach().cpu().numpy()
    res_pde3_np = residual_pde3.detach().cpu().numpy()

    # Plot residuals
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plot_field(axs[0], collocation_x_np[:, 0], collocation_x_np[:, 1], res_pde1_np, 
               rf"Lagrangian multipliers for PDE 1 (Iteration {iteration})", 
               np.linspace(np.min(res_pde1_np), np.max(res_pde1_np), 2000))
    plot_field(axs[1], collocation_x_np[:, 0], collocation_x_np[:, 1], res_pde2_np, 
               rf"Lagrangian multipliers for PDE 2 (Iteration {iteration})", 
               np.linspace(np.min(res_pde2_np), np.max(res_pde2_np), 2000))
    plot_field(axs[2], collocation_x_np[:, 0], collocation_x_np[:, 1], res_pde3_np, 
               rf"Lagrangian multipliers for PDE 3 (Iteration {iteration})", 
               np.linspace(np.min(res_pde3_np), np.max(res_pde3_np), 2000))
    plt.tight_layout(pad=2.0)
    # plt.pause(0.05)
    
    # Specify the file path and save the figure
    file_path = f"results_localized/Lagrangian_iteration_{iteration}.tiff"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, format='tiff', dpi=200)
    # plt.show(block=False)

def save_loss_history_to_csv(loss_total, loss_pde1_hist_both, loss_pde2_hist_both, 
                           loss_pde3_hist_both, loss_dp_hist_both, vol_loss_hist_both, 
                           iteration, output_dir="results_csv", fname = 'default'):
    """
    Save the loss history for different components during training to CSV files.
    
    :param loss_total: list, total loss history
    :param loss_pde1_hist_both, loss_pde2_hist_both, loss_pde3_hist_both: tuples of lists, PDE loss histories
    :param loss_dp_hist_both, vol_loss_hist_both: tuples of lists, other loss histories
    :param iteration: int, current iteration
    :param output_dir: str, directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack losses
    loss_pde1, loss_pde1_s = loss_pde1_hist_both
    loss_pde2, loss_pde2_s = loss_pde2_hist_both
    loss_pde3, loss_pde3_s = loss_pde3_hist_both
    loss_dp, loss_dp_s = loss_dp_hist_both
    vol_loss, vol_loss_s = vol_loss_hist_both
    
    # Create a DataFrame with all loss components
    epochs = range(len(loss_total))
    
    # Ensure all losses have the same length by padding with NaN if necessary
    max_length = max(len(loss_total), len(loss_pde1), len(loss_pde2), 
                    len(loss_pde3), len(loss_dp), len(vol_loss))
    
    def pad_to_length(loss_list, target_length):
        """Pad a list to the target length with NaN values"""
        if len(loss_list) < target_length:
            return np.pad(loss_list, (0, target_length - len(loss_list)), 
                          'constant', constant_values=np.nan).tolist()
        return loss_list
    
    # Pad all loss lists
    loss_total_padded = pad_to_length(loss_total, max_length)
    loss_pde1_padded = pad_to_length(loss_pde1, max_length)
    loss_pde2_padded = pad_to_length(loss_pde2, max_length)
    loss_pde3_padded = pad_to_length(loss_pde3, max_length)
    loss_dp_padded = pad_to_length(loss_dp, max_length)
    vol_loss_padded = pad_to_length(vol_loss, max_length)
    
    # Create epochs list
    epochs = list(range(max_length))
    
    # Create the DataFrame
    loss_df = pd.DataFrame({
        'epoch': epochs,
        'total_loss': loss_total_padded,
        'pde1_loss': loss_pde1_padded,
        'pde2_loss': loss_pde2_padded,
        'pde3_loss': loss_pde3_padded,
        'dissipated_power_loss': loss_dp_padded,
        'volume_loss': vol_loss_padded
    })
    
    # Create smooth losses DataFrame if available
    if loss_pde1_s:
        # Pad smooth losses
        loss_pde1_s_padded = pad_to_length(loss_pde1_s, max_length)
        loss_pde2_s_padded = pad_to_length(loss_pde2_s, max_length)
        loss_pde3_s_padded = pad_to_length(loss_pde3_s, max_length)
        loss_dp_s_padded = pad_to_length(loss_dp_s, max_length)
        vol_loss_s_padded = pad_to_length(vol_loss_s, max_length)
        
        # Add smooth losses to DataFrame
        loss_df['pde1_loss_smooth'] = loss_pde1_s_padded
        loss_df['pde2_loss_smooth'] = loss_pde2_s_padded
        loss_df['pde3_loss_smooth'] = loss_pde3_s_padded
        loss_df['dissipated_power_loss_smooth'] = loss_dp_s_padded
        loss_df['volume_loss_smooth'] = vol_loss_s_padded
    
    # Save to CSV
    loss_file = os.path.join(output_dir, f"{fname}_loss_history_iteration_{iteration}.csv")
    loss_df.to_csv(loss_file, index=False)
    
    # Create a summary file with final loss values
    final_loss_df = pd.DataFrame({
        'metric': ['iteration', 'final_total_loss', 'final_pde1_loss', 'final_pde2_loss', 
                   'final_pde3_loss', 'final_dissipated_power_loss', 'final_volume_loss'],
        'value': [iteration, 
                 loss_total[-1] if loss_total else np.nan,
                 loss_pde1[-1] if loss_pde1 else np.nan,
                 loss_pde2[-1] if loss_pde2 else np.nan,
                 loss_pde3[-1] if loss_pde3 else np.nan,
                 loss_dp[-1] if loss_dp else np.nan,
                 vol_loss[-1] if vol_loss else np.nan]
    })
    
    summary_file = os.path.join(output_dir, f"{fname}_loss_summary_iteration_{iteration}.csv")
    final_loss_df.to_csv(summary_file, index=False)
    
    print(f"Loss history saved to:")
    print(f"  - {loss_file}")
    print(f"  - {summary_file}")
    
    return {
        'loss_history': loss_file,
        'loss_summary': summary_file
    }
def plot_loss_history(loss_total, loss_pde1_hist_both, loss_pde2_hist_both, 
                      loss_pde3_hist_both, loss_dp_hist_both, vol_loss_hist_both, iteration):
    """
    Plot the loss history for different components during training.
    """
    def plot_log_loss(ax, loss, title, xlabel="Epoch", ylabel="Loss", formatted_value=None):
        ax.semilogy(loss)
        ax.set_title(f"{title}{f' = {formatted_value}' if formatted_value else ''}", fontsize=14, pad=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Unpack losses
    loss_pde1, loss_pde1_s = loss_pde1_hist_both
    loss_pde2, loss_pde2_s = loss_pde2_hist_both
    loss_pde3, loss_pde3_s = loss_pde3_hist_both
    loss_dp, loss_dp_s = loss_dp_hist_both
    vol_loss, vol_loss_s = vol_loss_hist_both

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Plot total loss
    plot_log_loss(axs[0, 0], loss_total, "Total Loss")

    # Plot PDE 1 loss
    plot_log_loss(axs[0, 1], loss_pde1, "Loss PDE 1")

    # Plot PDE 2 loss
    plot_log_loss(axs[0, 2], loss_pde2, "Loss PDE 2")

    # Plot PDE 3 loss
    plot_log_loss(axs[1, 0], loss_pde3, "Loss PDE 3")

    # Plot dissipated power loss
    formatted_loss_dp = f'{loss_dp[-1]:.2e}'
    plot_log_loss(axs[1, 1], loss_dp, "Dissipated Power", formatted_value=formatted_loss_dp)

    # Plot volume loss
    formatted_vol_loss = f'{vol_loss[-1]:.2e}'
    plot_log_loss(axs[1, 2], vol_loss, "Volume Loss", formatted_value=formatted_vol_loss)

    # Adjust layout and display
    plt.tight_layout(pad=2.0)
    # plt.show(block=False)
    # Specify the file path and save the figure
    file_path = f"results_localized/loss_historyn_{iteration}.tiff"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, format='tiff', dpi=200)
    # plt.show(block=False)




def format_axis_ticks(ax):
    """Format axis ticks and labels for consistency."""
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='major', labelsize=10)

def format_colorbar_ticks(cbar):
    """Format colorbar ticks for consistency."""
    cbar.ax.tick_params(labelsize=10)
    cbar.formatter = ticker.FormatStrFormatter('%.2f')
    cbar.update_ticks()




def plot_density_and_velocity_fields(u, v, p, ro, collocation_x, iteration,title="default"):
    """
    Plot density distribution, velocity streamline, and predicted U and V components.

    :param u: torch.Tensor, predicted U values.
    :param v: torch.Tensor, predicted V values.
    :param p: torch.Tensor, predicted P values (not used in this function, reserved for future use).
    :param ro: torch.Tensor, predicted density values.
    :param collocation_x: torch.Tensor, collocation points in 2D space (x, y).
    :param iteration: int, current iteration or epoch count.
    """
    # Convert tensors to numpy arrays
    U_pred_np = u.detach().cpu().numpy()
    V_pred_np = v.detach().cpu().numpy()
    ro_pred_np = ro.detach().cpu().numpy()
    collocation_x_cpu = collocation_x.detach().cpu().numpy()

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(np.min(collocation_x_cpu[:, 0]), np.max(collocation_x_cpu[:, 0]), 100),
        np.linspace(np.min(collocation_x_cpu[:, 1]), np.max(collocation_x_cpu[:, 1]), 100)
    )

    # Interpolate the U and V values onto the grid
    grid_U = griddata(collocation_x_cpu, U_pred_np, (grid_x, grid_y), method='cubic')
    grid_V = griddata(collocation_x_cpu, V_pred_np, (grid_x, grid_y), method='cubic')

    # Compute velocity squared
    velocity_squared = grid_U**2 + grid_V**2

    # Mask velocities where velocity squared >= 0.01
    mask = velocity_squared > 0.001  # >= 0.01
    grid_U_masked = np.where(mask, grid_U, 0)  # Set to 0 or another placeholder for masked values
    grid_V_masked = np.where(mask, grid_V, 0)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Density subplot
    levels_ro = np.linspace(np.min(ro_pred_np), np.max(ro_pred_np + 0.01), 2000)
    contour_ro = axs[0, 0].tricontourf(collocation_x_cpu[:, 0], collocation_x_cpu[:, 1], ro_pred_np,
                                    levels=levels_ro, vmin=0, vmax=1, cmap='jet')
    for v in contour_ro.collections:
        v.set_edgecolor("face")
    cbar_ro = fig.colorbar(contour_ro, ax=axs[0, 0], fraction=0.046, pad=0.04)
    format_colorbar_ticks(cbar_ro)
    axs[0, 0].set_title(rf'Density Distribution ($\rho$) at Iteration {iteration}', usetex=True, pad=7)

    axs[0, 0].scatter(collocation_x_cpu[:, 0], collocation_x_cpu[:, 1], s=.1, alpha=0.5)  # Adjust `s` for size and `alpha` for transparency
    
    axs[0, 0].set_xlabel(r'$x$', usetex=True)
    axs[0, 0].set_ylabel(r'$y$', usetex=True)
    
    
    ###########################
    ###########################
    # Define the obstacle polygons with the provided coordinates
    
    pg_1=[(0.20, 0.30),(0.25, 0.40),(0.35, 0.48),(0.33, 0.28),(0.25, 0.25)]#[(0.20, 0.10),(0.25, 0.20),(0.35, 0.28),(0.33, 0.08),(0.25, 0.05)]
    pg_2=[(0.55, 0.43),(0.63, 0.52),(0.70, 0.50),(0.68, 0.40),(0.58, 0.38)]
    pg_3=[(0.83, 0.68),(0.87, 0.75),(0.95, 0.73),(0.93, 0.65),(0.84, 0.63)]

    obstacle_polygons = [
        pg_1,  # Obstacle 1
        pg_2,  # Obstacle 2
        pg_3   # Obstacle 3
    ]
        
    # Define the thin wall polygon (tweak coordinates as needed)
    ratio=2/3
    H=1
    W=H*ratio
    L_t=0.35*W
    L_a=0.125*W
    w_w=0.025*W
    thin_wall = [
        (W/2-(w_w/2), 0.00),  # Bottom-left corner
        (W/2+(w_w/2), 0.00),  # Bottom-right corner (adjust width for a "thin" wall)
        (W/2+(w_w/2), 0.25),  # Top-right corner
        (W/2-(w_w/2), 0.25)   # Top-left corner
    ]

    # Create a shapely polygon for the wall
    # obstacle_polygons =[thin_wall]
    
    
    # Add each obstacle as a filled polygon
    plot_patch=False#True
    if plot_patch:
        for poly in obstacle_polygons:
            obstacle_patch = MplPolygon(poly, closed=True, color='gray', alpha=0.8)
            axs[0, 0].add_patch(obstacle_patch)
        
    ###########################
    ###########################
    # axs[0, 0].grid(True)
    format_axis_ticks(axs[0, 0])
    axs[0, 0].set_aspect('equal', adjustable='box')

    # Velocity streamline plot
    axs[0, 1].streamplot(grid_x, grid_y, grid_U_masked, grid_V_masked, color='blue', density=2)
    axs[0, 1].set_title(rf'Velocity Streamline at Iteration {iteration}', usetex=True, pad=7)
    axs[0, 1].set_xlabel(r'$x$', usetex=True)
    axs[0, 1].set_ylabel(r'$y$', usetex=True)
    axs[0, 1].grid(True)
    axs[0, 1].set_aspect('equal', adjustable='box')
    format_axis_ticks(axs[0, 1])

    # U_pred subplot
    levels_U = np.linspace(np.min(U_pred_np), np.max(U_pred_np), 2000)
    contour_U = axs[1, 0].tricontourf(collocation_x_cpu[:, 0], collocation_x_cpu[:, 1], U_pred_np,
                                      levels=levels_U, cmap='jet')
    for v in contour_U.collections:
        v.set_edgecolor("face")
    cbar_U = fig.colorbar(contour_U, ax=axs[1, 0], fraction=0.046, pad=0.04)
    format_colorbar_ticks(cbar_U)
    axs[1, 0].set_title(rf'Predicted $u(x,y)$ at Iteration {iteration}', usetex=True, pad=7)
    axs[1, 0].set_xlabel(r'$x$', usetex=True)
    axs[1, 0].set_ylabel(r'$y$', usetex=True)
    axs[1, 0].grid(True)
    format_axis_ticks(axs[1, 0])
    axs[1, 0].set_aspect('equal', adjustable='box')

    # V_pred subplot
    levels_V = np.linspace(np.min(V_pred_np), np.max(V_pred_np), 2000)
    contour_V = axs[1, 1].tricontourf(collocation_x_cpu[:, 0], collocation_x_cpu[:, 1], V_pred_np,
                                      levels=levels_V, cmap='jet')
    for v in contour_V.collections:
        v.set_edgecolor("face")
    cbar_V = fig.colorbar(contour_V, ax=axs[1, 1], fraction=0.046, pad=0.04)
    format_colorbar_ticks(cbar_V)
    axs[1, 1].set_title(rf'Predicted $v(x,y)$ at Iteration {iteration}', usetex=True, pad=7)
    axs[1, 1].set_xlabel(r'$x$', usetex=True)
    axs[1, 1].set_ylabel(r'$y$', usetex=True)
    axs[1, 1].grid(True)
    format_axis_ticks(axs[1, 1])
    axs[1, 1].set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # Ensure output directory exists (absolute path recommended)
    
    base_name = "figure_result"
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")  # e.g., 2025_11_10-16_45_30
    file_name = f"{base_name}_{timestamp}.png"
    file_path = os.path.join("Resuts_LCSMTO", file_name)

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save figure
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    print(f"✅ Figure saved successfully at: {os.path.abspath(file_path)}")



def modified_sigmoid(x, alpha=12.0):
    """Compute the modified sigmoid function."""
    return 1 / (1 + torch.exp(-alpha * (x - 0.5)))

def heaviside(phi, epsilon=1e-2):
    # A common smooth Heaviside: 0.5*(1 + tanh(phi/epsilon))
    return 0.5 * (1 + torch.tanh(phi / epsilon))

def alpha(rho):
    """
    Compute the alpha parameter based on rho.
    """
    alpha_max = 2.5e4
    alpha_min = 2.5e-4
    q = 0.1
    return alpha_max + (alpha_min - alpha_max) * rho * (1 + q) / (rho + q)

def dissipated_power(u, u_x, v_y, u_y, v_x):
    """
    Compute the total dissipated power based on the input velocity gradients and displacements.
    """
    # First part of the dissipated power
    p1 = (u_x**2 + v_y**2 + u_y**2 + v_x**2)#.sum(dim=1, keepdim=True)
    
    # Second part of the dissipated power
    u2 = (u[:, :2]**2).sum(dim=1, keepdim=True)
    p2 = alpha(u[:, 3:]) * u2
    
    return 0.5 * (p1 + p2)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

def plot_density_wirh_deviations(ro_tensor, L_z_posterior, collocation_x, iteration, z_all, u_x, v_y, u_y, v_x,mask_tensor):
    dp_loss = dissipated_power(z_all, u_x.reshape(-1, 1), v_y.reshape(-1, 1), u_y.reshape(-1, 1), v_x.reshape(-1, 1))
    dp_loss_val = torch.mean(dp_loss).item()

    # Compute density deviations
    ro_dev_1 = heaviside(ro_tensor + mask_tensor*1*L_z_posterior[:, 95])
    ro_dev_2 = heaviside(ro_tensor + mask_tensor*1*L_z_posterior[:, 96])
    ro_dev_3 = heaviside(ro_tensor + mask_tensor*1*L_z_posterior[:, 97])

    # Clone z_all for different deviations
    z_all_dev_1, z_all_dev_2, z_all_dev_3 = z_all.clone(), z_all.clone(), z_all.clone()
    z_all_dev_1[:, 3] = ro_dev_1
    z_all_dev_2[:, 3] = ro_dev_2
    z_all_dev_3[:, 3] = ro_dev_3

    dp_loss_dev_1_val = torch.mean(dissipated_power(z_all_dev_1, u_x.reshape(-1, 1), v_y.reshape(-1, 1), u_y.reshape(-1, 1), v_x.reshape(-1, 1))).item()
    dp_loss_dev_2_val = torch.mean(dissipated_power(z_all_dev_2, u_x.reshape(-1, 1), v_y.reshape(-1, 1), u_y.reshape(-1, 1), v_x.reshape(-1, 1))).item()
    dp_loss_dev_3_val = torch.mean(dissipated_power(z_all_dev_3, u_x.reshape(-1, 1), v_y.reshape(-1, 1), u_y.reshape(-1, 1), v_x.reshape(-1, 1))).item()

    # Convert tensors to numpy
    ro_pred_np = heaviside(ro_tensor).detach().cpu().numpy()
    ro_dev_np_1 = ro_dev_1.detach().cpu().numpy()
    ro_dev_np_2 = ro_dev_2.detach().cpu().numpy()
    ro_dev_np_3 = ro_dev_3.detach().cpu().numpy()
    collocation_x_cpu = collocation_x.detach().cpu().numpy()

    # Set up a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25)

    # Plot settings
    plot_data = [
        (ro_pred_np, dp_loss_val, "Original Density", axs[0, 0]),
        (ro_dev_np_1, dp_loss_dev_1_val, "Realization 1", axs[0, 1]),
        (ro_dev_np_2, dp_loss_dev_2_val, "Realization 2", axs[1, 0]),
        (ro_dev_np_3, dp_loss_dev_3_val, "Realization 3", axs[1, 1]),
    ]

    for ro_data, dp_loss_val, title, ax in plot_data:
        levels = np.linspace(np.min(ro_data), np.max(ro_data + 0.01), 2000)
        contour = ax.tricontourf(collocation_x_cpu[:, 0], collocation_x_cpu[:, 1], ro_data,
                                 levels=levels, vmin=0, vmax=1, cmap='jet')
        for v in contour.collections:
            v.set_edgecolor("face")
        
        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        format_colorbar_ticks(cbar)
        ax.set_title(f"{title}\nDP {dp_loss_val:.4f} at Iteration {iteration}", fontsize=12, pad=5)
        ax.scatter(collocation_x_cpu[:, 0], collocation_x_cpu[:, 1], s=.1, alpha=0.5)
        ax.set_xlabel(r'$x$', fontsize=10)
        ax.set_ylabel(r'$y$', fontsize=10)
        ax.set_aspect('equal', adjustable='box')

    # Remove excessive margins
    plt.tight_layout()

    # Save the figure
    file_path = f"FIG_results_localized/Density_and_its_Deviations_at_iteration_{iteration}.tiff"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # plt.savefig(file_path, format='tiff', dpi=200, bbox_inches='tight', pad_inches=0.01)
    png_path = os.path.splitext(file_path)[0] + '.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)


