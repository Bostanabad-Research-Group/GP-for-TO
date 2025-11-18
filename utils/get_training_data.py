import torch
import numpy as np
from utils.utils_general import get_tkwargs
import matplotlib.pyplot as plt

tkwargs = get_tkwargs()

def get_data_EX2D1(MP):
    # get collocation points
    domain = MP['domain']
    pad = MP['pad']
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    Nelx_max = MP['Nelx_max']
    Nelx_min = MP['Nelx_min']
    Nely_max = MP['Nely_max']
    Nely_min = MP['Nely_min']
    num_CP = MP['num_CP']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    xi = np.pad(xi, pad_width=pad, mode='linear_ramp', end_values=(xi[0] - pad * dx, xi[-1] + pad * dx))
    yi = np.pad(yi, pad_width=pad, mode='linear_ramp', end_values=(yi[0] - pad * dy, yi[-1] + pad * dy))
    xi, yi = np.meshgrid(xi, yi)
    
    # delete points in unwanted area:
    distance = ((xi) ** 2 + (yi) ** 2) ** 0.5
    mask_domain = distance >= -1 # no holes, so all True
    mask_padded = np.ones_like(xi, dtype=bool)
    mask_padded[pad:-pad, pad:-pad] = False  # set the padding to True
    mask_col = mask_domain & (~mask_padded)
    Nx = mask_col.shape[1]
    Ny = mask_col.shape[0]
    mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)

    index = (X_col[:, 0] == 0.0) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax)
    LE = X_col[index] # left edge coordinates
    index = (X_col[:, 0] == xmax) & (X_col[:, 1] == 0)
    Bottom_support = X_col[index] 

    # define the traction BC:
    traction_coord = torch.tensor([xmin, ymax])
    traction_magnitude = torch.tensor(-1e-1)
    traction_indices = (X_col[:, 0] == xmin) & (X_col[:, 1] == ymax)

    # masked_domain = X_col[mask_col]
    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2], marker='o', alpha=0.3, s=3, color='purple', label = 'masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left edge')
    # ax.scatter(traction_coord[0], traction_coord[1], marker='o', alpha=0.9, s=2, color='green', label = 'external force')
    # ax.scatter(Bottom_support[0,0], Bottom_support[0,1], marker='o', alpha=0.9, s=2, color='black', label = 'bottom support')
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    traction_indices = traction_indices.requires_grad_(False).to(tkwargs['device'])

    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    LE = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    u_X_train = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(Bottom_support).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    rho_X_train = torch.vstack([traction_coord,Bottom_support]).detach()
    rho_X_train = rho_X_train.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(rho_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_coord':traction_coord,
                'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'LE':LE,'X_col':X_col,'mask_col':mask_col,'xi':xi,'yi':yi,'dx':dx, 'dy':dy,'Nx':Nx,'Ny':Ny}
    
    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nelx = Nelx_list[i]
        Nely = Nely_list[i]
    
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        xi = np.pad(xi, pad_width=pad, mode='linear_ramp', end_values=(xi[0] - pad*dx, xi[-1] + pad*dx))
        yi = np.pad(yi, pad_width=pad, mode='linear_ramp', end_values=(yi[0] - pad*dy, yi[-1] + pad*dy))
        xi, yi = np.meshgrid(xi, yi)

        # delete points in unwanted area:
        distance = ((xi) ** 2 + (yi) ** 2) ** 0.5
        mask_domain = distance >= -1 # no holes, so all True
        mask_padded = np.ones_like(xi, dtype=bool)
        mask_padded[pad:-pad, pad:-pad] = False  # set the padding to True
        mask_col = mask_domain & (~mask_padded)
        Nx = mask_col.shape[1]
        Ny = mask_col.shape[0]
        mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
        X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # define the traction BC:
        traction_indices = (X_col[:, 0] == xmin) & (X_col[:, 1] == ymax)
        traction_indices = traction_indices.requires_grad_(False).to(tkwargs['device'])

        # save in X_col_all[i]
        # Save all data in a dictionary
        X_col_data = {'traction_indices': traction_indices,'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'dx':dx,'dy':dy}
        
        # Append the dictionary to the list
        X_col_all.append(X_col_data)
    return Training, X_col_all

def get_data_EX2D2(MP):
    # get collocation points
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    domain = MP['domain']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    xi = np.pad(xi, pad_width=2, mode='linear_ramp', end_values=(xi[0] - 2*dx, xi[-1] + 2*dx))
    yi = np.pad(yi, pad_width=2, mode='linear_ramp', end_values=(yi[0] - 2*dy, yi[-1] + 2*dy))
    xi, yi = np.meshgrid(xi, yi)
    
    # delete points in unwanted area:
    distance = ((xi) ** 2 + (yi) ** 2) ** 0.5
    mask_domain = distance >= -1 # no holes, so all True
    mask_padded = np.ones_like(xi, dtype=bool)
    mask_padded[2:-2, 2:-2] = False  # set the padding to True
    mask_col = mask_domain & (~mask_padded)
    Nx = mask_col.shape[1]
    Ny = mask_col.shape[0]
    mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)

    index = (X_col[:, 0] == 0.0) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax)
    LE = X_col[index] # left edge coordinates
    index = (X_col[:, 0] == xmax) & (X_col[:, 1] == 0)
    #Bottom_support = X_col[index] 

    # define the traction BC:
    traction_coord = torch.tensor([[xmax, ymin]])
    traction_magnitude = torch.tensor(-1e-1)
    traction_indices = (X_col[:, 0] == xmax) & (X_col[:, 1] == ymin)
    #traction_indices = torch.nonzero((X_col[:, 0] == xmax) & (X_col[:, 1] == ymin), as_tuple=False).squeeze().tolist()
    
    # Visualize ALL the training points
    # masked_domain = X_col[mask_col]
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2], marker='o', alpha=0.3, s=3, color='purple', label = 'masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left edge')
    # ax.scatter(traction_coord[0,0], traction_coord[0,1], marker='o', alpha=0.9, s=2, color='green', label = 'external force')
    # #ax.scatter(Bottom_support[0,0], Bottom_support[0,1], marker='o', alpha=0.9, s=2, color='black', label = 'bottom support')
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    traction_indices = traction_indices.requires_grad_(False).to(tkwargs['device'])

    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    LE = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    u_X_train = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    rho_X_train = traction_coord
    rho_X_train = rho_X_train.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(rho_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_coord':traction_coord,
                'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'LE':LE,'X_col':X_col,'mask_col':mask_col,'xi':xi,'yi':yi,'dx':dx, 'dy':dy,'Nx':Nx,'Ny':Ny}
    return Training



