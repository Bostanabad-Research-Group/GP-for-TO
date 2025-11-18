import matplotlib.pyplot as plt
import os
from matplotlib import ticker
import numpy as np
from scipy.interpolate import griddata
import datetime

def format_axis_ticks(ax):
    """Format axis ticks for consistent appearance."""
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='major', labelsize=10)

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
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))
    plot_field(axs[0], collocation_x_np[:, 0], collocation_x_np[:, 1], u_np, 
               rf"$U$: Predicted Mean (Iteration {iteration})", 
               np.linspace(np.min(u_np), np.max(u_np), 2000))
    plot_field(axs[1], collocation_x_np[:, 0], collocation_x_np[:, 1], v_np, 
               rf"$V$: Predicted Mean (Iteration {iteration})", 
               np.linspace(np.min(v_np), np.max(v_np), 2000))
    plot_field(axs[2], collocation_x_np[:, 0], collocation_x_np[:, 1], p_np, 
               rf"$P$: Predicted Mean (Iteration {iteration})", 
               np.linspace(np.min(p_np), np.max(p_np), 2000))
    plot_field(axs[3], collocation_x_np[:, 0], collocation_x_np[:, 1], ro_np, 
               rf"$\rho$: Predicted Mean (Iteration {iteration})", 
               np.linspace(np.min(ro_np), np.max(ro_np), 2000))
    plt.tight_layout()
    plt.pause(0.05)

    # Compute normalized residuals
    res_pde1_np = (1 / w_pde1) * residual_pde1.detach().cpu().numpy()
    res_pde2_np = (1 / w_pde2) * residual_pde2.detach().cpu().numpy()
    res_pde3_np = (1 / w_pde3) * residual_pde3.detach().cpu().numpy()

    # Plot residuals
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plot_field(axs[0], collocation_x_np[:, 0], collocation_x_np[:, 1], res_pde1_np, 
               rf"Residual PDE 1 (Iteration {iteration})", 
               np.linspace(np.min(res_pde1_np), np.max(res_pde1_np), 2000))
    plot_field(axs[1], collocation_x_np[:, 0], collocation_x_np[:, 1], res_pde2_np, 
               rf"Residual PDE 2 (Iteration {iteration})", 
               np.linspace(np.min(res_pde2_np), np.max(res_pde2_np), 2000))
    plot_field(axs[2], collocation_x_np[:, 0], collocation_x_np[:, 1], res_pde3_np, 
               rf"Residual PDE 3 (Iteration {iteration})", 
               np.linspace(np.min(res_pde3_np), np.max(res_pde3_np), 2000))
    plt.tight_layout(pad=2.0)
    plt.pause(0.05)



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

def plot_density_and_velocity_fields(u, v, p, ro, collocation_x, iteration, title):
    # --- to numpy
    U_pred_np = u.detach().cpu().numpy()
    V_pred_np = v.detach().cpu().numpy()
    ro_pred_np = ro.detach().cpu().numpy()
    collocation_x_cpu = collocation_x.detach().cpu().numpy()

    # --- grid (same extents as data)
    x_min, x_max = collocation_x_cpu[:, 0].min(), collocation_x_cpu[:, 0].max()
    y_min, y_max = collocation_x_cpu[:, 1].min(), collocation_x_cpu[:, 1].max()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # --- interpolate U,V
    grid_U = griddata(collocation_x_cpu, U_pred_np, (grid_x, grid_y), method='cubic')
    grid_V = griddata(collocation_x_cpu, V_pred_np, (grid_x, grid_y), method='cubic')

    # --- mask (optional, keep as you had)
    velocity_squared = grid_U**2 + grid_V**2
    mask = velocity_squared > 0.01
    grid_U_masked = np.where(mask, grid_U, 0)
    grid_V_masked = np.where(mask, grid_V, 0)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # --- density
    levels_ro = np.linspace(np.min(ro_pred_np), np.max(ro_pred_np + 0.01), 2000)
    contour_ro = axs[0, 0].tricontourf(collocation_x_cpu[:, 0], collocation_x_cpu[:, 1], ro_pred_np,
                                       levels=levels_ro, vmin=0, vmax=1, cmap='jet')
    for coll in contour_ro.collections:
        coll.set_edgecolor("face")
    cbar_ro = fig.colorbar(contour_ro, ax=axs[0, 0], fraction=0.046, pad=0.04)
    format_colorbar_ticks(cbar_ro)
    axs[0, 0].set_title(rf'Density Distribution ($\rho$) at Iteration {iteration}', usetex=True, pad=7)
    axs[0, 0].set_xlabel(r'$x$', usetex=True)
    axs[0, 0].set_ylabel(r'$y$', usetex=True)
    axs[0, 0].grid(True)
    format_axis_ticks(axs[0, 0])

    # --- streamlines
    axs[0, 1].streamplot(grid_x, grid_y, grid_U_masked, grid_V_masked, color='blue', density=2)
    axs[0, 1].set_title(rf'Velocity Streamline at Iteration {iteration}', usetex=True, pad=7)
    axs[0, 1].set_xlabel(r'$x$', usetex=True)
    axs[0, 1].set_ylabel(r'$y$', usetex=True)
    axs[0, 1].grid(True)
    format_axis_ticks(axs[0, 1])

    # --- U
    levels_U = np.linspace(np.min(U_pred_np), np.max(U_pred_np), 2000)
    contour_U = axs[1, 0].tricontourf(collocation_x_cpu[:, 0], collocation_x_cpu[:, 1], U_pred_np,
                                      levels=levels_U, cmap='jet')
    for coll in contour_U.collections:
        coll.set_edgecolor("face")
    cbar_U = fig.colorbar(contour_U, ax=axs[1, 0], fraction=0.046, pad=0.04)
    format_colorbar_ticks(cbar_U)
    axs[1, 0].set_title(rf'Predicted $u(x,y)$ at Iteration {iteration}', usetex=True, pad=7)
    axs[1, 0].set_xlabel(r'$x$', usetex=True)
    axs[1, 0].set_ylabel(r'$y$', usetex=True)
    axs[1, 0].grid(True)
    format_axis_ticks(axs[1, 0])

    # --- V
    levels_V = np.linspace(np.min(V_pred_np), np.max(V_pred_np), 2000)
    contour_V = axs[1, 1].tricontourf(collocation_x_cpu[:, 0], collocation_x_cpu[:, 1], V_pred_np,
                                      levels=levels_V, cmap='jet')
    for coll in contour_V.collections:
        coll.set_edgecolor("face")
    cbar_V = fig.colorbar(contour_V, ax=axs[1, 1], fraction=0.046, pad=0.04)
    format_colorbar_ticks(cbar_V)
    axs[1, 1].set_title(rf'Predicted $v(x,y)$ at Iteration {iteration}', usetex=True, pad=7)
    axs[1, 1].set_xlabel(r'$x$', usetex=True)
    axs[1, 1].set_ylabel(r'$y$', usetex=True)
    axs[1, 1].grid(True)
    format_axis_ticks(axs[1, 1])

    # === REAL ASPECT RATIO (key lines) ===
    for ax in axs.ravel():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')  # 1 data unit in x == 1 in y

    plt.tight_layout()



    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M") # Example output: "20250910_2256"

    # --- save
    file_path = f"Resuts_SMTO/density_velocity_iteration_{title}_{iteration}_timestamp_{timestamp}.tiff"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # plt.savefig(file_path, format='tiff', dpi=20)
    png_path = os.path.splitext(file_path)[0] + '.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)

