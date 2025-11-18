import torch
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_all(iteration, save_folder, contour_data, timeHistory):
    Nelx = contour_data['Nelx']
    Nely = contour_data['Nely']
    collocation_x = contour_data['collocation_x']
    u = contour_data['u']
    v = contour_data['v']
    rho = contour_data['rho']
    if iteration < 0:
        rho_static_thres = contour_data['rho_static_thres']
        rho_dynamic_thres = contour_data['rho_dynamic_thres']
    s11 = contour_data['s11']
    s22 = contour_data['s22']
    s12 = contour_data['s12']
    e11 = contour_data['e11']
    e22 = contour_data['e22']
    e12 = contour_data['e12']
    residual_pde1 = contour_data['residual_pde1']
    residual_pde2 = contour_data['residual_pde2']
    compliance = contour_data['comp_vector']
    dC_drho = contour_data['dC_drho']
    x_nodes = collocation_x[:, 0]
    y_nodes = collocation_x[:, 1]
    if iteration < 0: # use negative values to make contours only
         pass
    else: # regular iteration plot has both contour and time history
        loss_total = timeHistory['loss_total']
        loss_compliance = timeHistory['loss_compliance']
        loss_dem = timeHistory['loss_dem']
        loss_volConstraint = timeHistory['loss_volConstraint']
            
        SE = timeHistory['strain_energy']
        EW = timeHistory['external_work']
        loss_pde_1 = timeHistory['loss_pde_1']
        loss_pde_2 = timeHistory['loss_pde_2']
        vol = timeHistory['vol']
        wc = timeHistory['wc'][-1]
        wd = timeHistory['wd'][-1]
        wv = timeHistory['wv'][-1]
        wc_hist = timeHistory['wc']
        wd_hist = timeHistory['wd']
        wv_hist = timeHistory['wv']
        grey = timeHistory['grey']
        if timeHistory['loss_tv']:
            loss_tv = timeHistory['loss_tv']
            wtv_hist = timeHistory['wtv']
            wtv = timeHistory['wtv'][-1]


    # Define number of levels in the contour plot
    num_levels = 150
    if iteration <0:
        num_levels = 500

    # Create a meshgrid for contour plotting of displacements
    xi, yi = np.linspace(x_nodes.min(), x_nodes.max(), 3*Nelx), np.linspace(y_nodes.min(), y_nodes.max(), 3*Nely)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate rho values on the grid using scipy's griddata
    zi_u = griddata((x_nodes, y_nodes), u, (xi, yi), method='linear')
    zi_v = griddata((x_nodes, y_nodes), v, (xi, yi), method='linear')
    zi_rho = griddata((x_nodes, y_nodes), rho, (xi, yi), method='linear')
    if iteration < 0:
        zi_rho_static = griddata((x_nodes, y_nodes), rho_static_thres, (xi, yi), method='linear')
        zi_rho_dynamic = griddata((x_nodes, y_nodes), rho_dynamic_thres, (xi, yi), method='linear')
    zi_s11 = griddata((x_nodes, y_nodes), s11, (xi, yi), method='linear')
    zi_s22 = griddata((x_nodes, y_nodes), s22, (xi, yi), method='linear')
    zi_s12 = griddata((x_nodes, y_nodes), s12, (xi, yi), method='linear')
    zi_e11 = griddata((x_nodes, y_nodes), e11, (xi, yi), method='linear')
    zi_e22 = griddata((x_nodes, y_nodes), e22, (xi, yi), method='linear')
    zi_e12 = griddata((x_nodes, y_nodes), e12, (xi, yi), method='linear')
    zi_pde1 = griddata((x_nodes, y_nodes), residual_pde1, (xi, yi), method='linear')
    zi_pde2 = griddata((x_nodes, y_nodes), residual_pde2, (xi, yi), method='linear')
    zi_C = griddata((x_nodes, y_nodes), compliance, (xi, yi), method='linear')
    zi_dC_drho = griddata((x_nodes, y_nodes), dC_drho, (xi, yi), method='linear')
    
    # Create a figure with 2 subplots for displacements
    if iteration < 0: # contour only for prediction
        fig, axs = plt.subplots(3,5, figsize=(40, 15)) # horizontal interval 8 and vertical interval 5
        # Contour plot for rho
        contour = axs[0,4].contourf(xi, yi, zi_rho_static, levels=num_levels, cmap='Greys', vmin=0, vmax=1)
        axs[0,4].set_title('rho (static binarized)')
        axs[0,4].set_xlabel('X coordinate')
        axs[0,4].set_ylabel('Y coordinate')
        axs[0,4].set_aspect('equal')
        fig.colorbar(contour, ax=axs[0,4])

        contour = axs[1,4].contourf(xi, yi, zi_rho_dynamic, levels=num_levels, cmap='Greys', vmin=0, vmax=1)
        axs[1,4].set_title('rho (dynamic binarized)')
        axs[1,4].set_xlabel('X coordinate')
        axs[1,4].set_ylabel('Y coordinate')
        axs[1,4].set_aspect('equal')
        fig.colorbar(contour, ax=axs[1,4])
    else:
        fig, axs = plt.subplots(4,5, figsize=(40, 20)) # horizontal interval 8 and vertical interval 5
        axs[0, 4].semilogy(loss_pde_1, label='loss pde 1', linewidth=3)
        axs[0, 4].semilogy(loss_pde_2, label='loss pde 2', linewidth=2)
        axs[0, 4].set_xlabel('Epoch')
        axs[0, 4].set_ylabel('Loss')
        axs[0, 4].legend()

        axs[1, 4].semilogy(wc_hist, label='wc', linewidth=3)
        axs[1, 4].semilogy(wd_hist, label='wd', linewidth=2)
        axs[1, 4].semilogy(wv_hist, label='wv', linewidth=1.5)
        if timeHistory['loss_tv']:
            axs[1, 4].semilogy(wtv_hist, label='wtv', linewidth=1)
        axs[1, 4].set_xlabel('Epoch')
        axs[1, 4].set_ylabel('weights')
        axs[1, 4].legend()

        axs[3, 0].plot(np.array(loss_total),label=f'Loss_total', linewidth=3)
        axs[3, 0].plot(np.array(loss_compliance)*wc,label=f'{wc}*Loss_compliance', linewidth=2)
        axs[3, 0].plot(np.array(loss_dem)*wd,label=f'{wd}*Loss_DEM', linewidth=1.5)
        axs[3, 0].plot(np.array(loss_volConstraint)*wv,label=f'{wv}*Loss_volconstr', linewidth=1)
        if timeHistory['loss_tv']:
            axs[3, 0].plot(np.array(loss_tv)*wtv,label=f'{wtv}*Loss_tv', linewidth=1)
        axs[3, 0].set_xlabel('Epoch')
        axs[3, 0].set_ylabel('Loss')
        axs[3, 0].legend()

        axs[3, 1].semilogy(np.array(loss_total),label=f'Loss_total', linewidth=3)
        axs[3, 1].semilogy(np.array(loss_compliance)*wc,label=f'{wc}*Loss_compliance', linewidth=2)
        axs[3, 1].semilogy(np.array(loss_dem)*wd,label=f'{wd}*Loss_DEM', linewidth=1.5)
        axs[3, 1].semilogy(np.array(loss_volConstraint)*wv,label=f'{wv}*Loss_volconstr', linewidth=1)
        if timeHistory['loss_tv']:
            axs[3, 1].semilogy(np.array(loss_tv)*wtv,label=f'{wtv}*Loss_tv', linewidth=1)
        axs[3, 1].set_xlabel('Epoch')
        axs[3, 1].set_ylabel('Loss')
        axs[3, 1].legend()

        axs[3, 2].plot(np.array(SE)*2,label=f'2*SE', linewidth=3)
        axs[3, 2].plot(np.array(EW),label=f'EW', linewidth=2)
        axs[3, 2].set_xlabel('Epoch')
        axs[3, 2].set_ylabel('Energies')
        axs[3, 2].legend()

        axs[3, 3].semilogy(np.array(SE)*2,label=f'2*SE', linewidth=3)
        axs[3, 3].semilogy(np.array(EW),label=f'EW', linewidth=2)
        axs[3, 3].set_xlabel('Epoch')
        axs[3, 3].set_ylabel('Energies')
        axs[3, 3].legend()

        axs[3, 4].semilogy(vol, label = 'Volume fraction')
        axs[3, 4].semilogy(grey, label = 'grey element fraction')
        axs[3, 4].set_xlabel('Epoch')
        axs[3, 4].set_ylabel('Fraction')
        axs[3, 4].legend()
    
    # Contour plot for u
    contour = axs[0,0].contourf(xi, yi, zi_u, levels=num_levels, cmap='rainbow')
    axs[0,0].set_title('horizontal displacement u')
    axs[0,0].set_xlabel('X coordinate')
    axs[0,0].set_ylabel('Y coordinate')
    axs[0,0].set_aspect('equal')
    fig.colorbar(contour, ax=axs[0,0])

    # Contour plot for v
    contour = axs[1,0].contourf(xi, yi, zi_v, levels=num_levels, cmap='rainbow')
    axs[1,0].set_title('vertical displacement v')
    axs[1,0].set_xlabel('X coordinate')
    axs[1,0].set_ylabel('Y coordinate')
    axs[1,0].set_aspect('equal')
    fig.colorbar(contour, ax=axs[1,0])

    # Contour plot for rho
    contour = axs[2,0].contourf(xi, yi, zi_rho, levels=num_levels, cmap='Greys', vmin=0, vmax=1)
    axs[2,0].set_title('rho (before binarize)')
    axs[2,0].set_xlabel('X coordinate')
    axs[2,0].set_ylabel('Y coordinate')
    axs[2,0].set_aspect('equal')
    fig.colorbar(contour, ax=axs[2,0])

    # Contour plot for s11
    contour = axs[0,1].contourf(xi, yi, zi_s11, levels=num_levels, cmap='rainbow')
    axs[0,1].set_title('s11')
    axs[0,1].set_xlabel('X coordinate')
    axs[0,1].set_ylabel('Y coordinate')
    axs[0,1].set_aspect('equal')
    fig.colorbar(contour, ax=axs[0,1])

    # Contour plot for s22
    contour = axs[1,1].contourf(xi, yi, zi_s22, levels=num_levels, cmap='rainbow')
    axs[1,1].set_title('s22')
    axs[1,1].set_xlabel('X coordinate')
    axs[1,1].set_ylabel('Y coordinate')
    axs[1,1].set_aspect('equal')
    fig.colorbar(contour, ax=axs[1,1])

    # Contour plot for s12
    contour = axs[2,1].contourf(xi, yi, zi_s12, levels=num_levels, cmap='rainbow')
    axs[2,1].set_title('s12')
    axs[2,1].set_xlabel('X coordinate')
    axs[2,1].set_ylabel('Y coordinate')
    axs[2,1].set_aspect('equal')
    fig.colorbar(contour, ax=axs[2,1])

    # Contour plot for e11
    contour = axs[0,2].contourf(xi, yi, zi_e11, levels=num_levels, cmap='rainbow')
    axs[0,2].set_title('e11')
    axs[0,2].set_xlabel('X coordinate')
    axs[0,2].set_ylabel('Y coordinate')
    axs[0,2].set_aspect('equal')
    fig.colorbar(contour, ax=axs[0,2])

    # Contour plot for e22
    contour = axs[1,2].contourf(xi, yi, zi_e22, levels=num_levels, cmap='rainbow')
    axs[1,2].set_title('e22')
    axs[1,2].set_xlabel('X coordinate')
    axs[1,2].set_ylabel('Y coordinate')
    axs[1,2].set_aspect('equal')
    fig.colorbar(contour, ax=axs[1,2])

    # Contour plot for e12
    contour = axs[2,2].contourf(xi, yi, zi_e12, levels=num_levels, cmap='rainbow')
    axs[2,2].set_title('e12')
    axs[2,2].set_xlabel('X coordinate')
    axs[2,2].set_ylabel('Y coordinate')
    axs[2,2].set_aspect('equal')
    fig.colorbar(contour, ax=axs[2,2])

    # Contour plot for compliance
    contour = axs[2,3].contourf(xi, yi, zi_C, levels=num_levels, cmap='rainbow')
    axs[2,3].set_title('Compliance')
    axs[2,3].set_xlabel('X coordinate')
    axs[2,3].set_ylabel('Y coordinate')
    axs[2,3].set_aspect('equal')
    fig.colorbar(contour, ax=axs[2,3])

    # Contour plot for compliance
    contour = axs[2,4].contourf(xi, yi, zi_dC_drho, levels=num_levels, cmap='rainbow')
    axs[2,4].set_title('dC_drho')
    axs[2,4].set_xlabel('X coordinate')
    axs[2,4].set_ylabel('Y coordinate')
    axs[2,4].set_aspect('equal')
    fig.colorbar(contour, ax=axs[2,4])

    # Contour plot for Residual of PDE1
    contour = axs[0,3].contourf(xi, yi, zi_pde1, levels=num_levels, cmap='rainbow')
    axs[0,3].set_title('Residual of PDE1')
    axs[0,3].set_xlabel('X coordinate')
    axs[0,3].set_ylabel('Y coordinate')
    axs[0,3].set_aspect('equal')
    fig.colorbar(contour, ax=axs[0,3])

    # Contour plot for Residual of PDE2
    contour = axs[1,3].contourf(xi, yi, zi_pde2, levels=num_levels, cmap='rainbow')
    axs[1,3].set_title('Residual of PDE2')
    axs[1,3].set_xlabel('X coordinate')
    axs[1,3].set_ylabel('Y coordinate')
    axs[1,3].set_aspect('equal')
    fig.colorbar(contour, ax=axs[1,3])

    if iteration < 0:
        file_name = f"Trained_prediction.jpeg"
    else:
        file_name = f"TO_iter_{iteration}.jpeg"
    file_path = f"{save_folder}{file_name}"
    plt.savefig(file_path, format='jpeg', dpi=600) 
    plt.clf()
