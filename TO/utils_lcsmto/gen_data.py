import torch
import numpy as np
import torch
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from torch.quasirandom import SobolEngine
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import torch
import matplotlib.pyplot as plt

def gen_data(design_domain, steps_x,steps_y):
    x_values = torch.linspace(design_domain['x'][0], design_domain['x'][1], steps=steps_x)
    y_values = torch.linspace(design_domain['y'][0], design_domain['y'][1], steps=steps_y)
    X, Y = torch.meshgrid(x_values, y_values, indexing='ij')

    X_col_domain = torch.stack([X.flatten(), Y.flatten()], dim=1)
    return X_col_domain

def get_data_fluid(problem = 'rugby', N_col_domain = 10000, N_train = 25.0, ratio=1.0,MP={},tkwargs={}):

    domain = {'x':[0.0, ratio*1.0], 'y':[0.0, 1.0]}
    # points_x = torch.linspace(domain['x'][0], domain['x'][1], 5*N_train+2)[1:-1]
    points_x = torch.linspace(domain['x'][0], domain['x'][1], int((ratio*N_train)+2))[1:-1]

    points_y = torch.linspace(domain['y'][0], domain['y'][1], N_train+2)[1:-1]
    
    
    # if problem == 'four_terminals':
        
        
    #     l=1/3
    #     L=1
    #     vmax=10
    #     """
    #     Generate boundary-condition data for a domain consisting of:
    #     - A central rectangle of width L and height l.
    #     - Two pipes on the left (inlets) of length 2*l (top and bottom).
    #     - Two pipes on the right (outlets) of length 2*l (top and bottom).
        
    #     Inlet boundaries are on the left edges at x = -2*l.
    #     Outlet boundaries are on the right edges at x = L + 2*l.
    #     The top and bottom boundaries span the entire external domain.
    #     """
    #     # --- Helper functions ---
    #     def boundary_line_x(x_val, y_start, y_end, n_points):
    #         """Create a vertical line at x = x_val, with y in [y_start, y_end]."""
    #         y = torch.linspace(y_start, y_end, n_points)
    #         return torch.stack((x_val * torch.ones_like(y), y), dim=1)

    #     def boundary_line_y(y_val, x_start, x_end, n_points):
    #         """Create a horizontal line at y = y_val, with x in [x_start, x_end]."""
    #         x = torch.linspace(x_start, x_end, n_points)
    #         return torch.stack((x, y_val * torch.ones_like(x)), dim=1)

    #     def parabolic_velocity(y, y1, y2, vmax):
    #         """
    #         Compute a simple parabolic profile:
    #         - Maximum velocity vmax at the midpoint,
    #         - Zero velocity at y = y1 and y = y2.
    #         """
    #         mid = 0.5 * (y1 + y2)
    #         half = 0.5 * (y2 - y1)
    #         return vmax * (1.0 - ((y - mid) / half) ** 2)

    #     # Store boundary data in lists, then concatenate
    #     X_bc, U_bc, V_bc, ro_bc = [], [], [], []

    #     # ------------------------------------------------------------
    #     # 1) LEFT PIPE INLETS (parabolic velocity profile)
    #     # ------------------------------------------------------------
    #     # a) Top-left pipe inlet: x = -2*l, y in [l/2, l]
    #     x_tl_in = -2.0 * l
    #     y_tl_low, y_tl_high = 0.5 * l, l
    #     inlet_tl = boundary_line_x(x_tl_in, y_tl_low, y_tl_high, N_train)
    #     u_tl = parabolic_velocity(inlet_tl[:, 1], y_tl_low, y_tl_high, vmax)
    #     v_tl = torch.zeros_like(u_tl)
    #     ro_tl = torch.ones_like(u_tl)  # Example: constant density

    #     X_bc.append(inlet_tl)
    #     U_bc.append(u_tl)
    #     V_bc.append(v_tl)
    #     ro_bc.append(ro_tl)

    #     # b) Bottom-left pipe inlet: x = -2*l, y in [0, l/2]
    #     x_bl_in = -2.0 * l
    #     y_bl_low, y_bl_high = 0.0, 0.5 * l
    #     inlet_bl = boundary_line_x(x_bl_in, y_bl_low, y_bl_high, N_train)
    #     u_bl = parabolic_velocity(inlet_bl[:, 1], y_bl_low, y_bl_high, vmax)
    #     v_bl = torch.zeros_like(u_bl)
    #     ro_bl = torch.ones_like(u_bl)

    #     X_bc.append(inlet_bl)
    #     U_bc.append(u_bl)
    #     V_bc.append(v_bl)
    #     ro_bc.append(ro_bl)

    #     # ------------------------------------------------------------
    #     # 2) RIGHT PIPE OUTLETS (often zero velocity or a specified outflow)
    #     # ------------------------------------------------------------
    #     # a) Top-right pipe outlet: x = L + 2*l, y in [l/2, l]
    #     x_tr_out = L + 2.0 * l
    #     outlet_tr = boundary_line_x(x_tr_out, y_tl_low, y_tl_high, N_train)
    #     u_tr = torch.zeros(N_train)  # Example: zero velocity at outlet
    #     v_tr = torch.zeros(N_train)
    #     ro_tr = torch.ones(N_train)

    #     X_bc.append(outlet_tr)
    #     U_bc.append(u_tr)
    #     V_bc.append(v_tr)
    #     ro_bc.append(ro_tr)

    #     # b) Bottom-right pipe outlet: x = L + 2*l, y in [0, l/2]
    #     x_br_out = L + 2.0 * l
    #     outlet_br = boundary_line_x(x_br_out, y_bl_low, y_bl_high, N_train)
    #     u_br = torch.zeros(N_train)
    #     v_br = torch.zeros(N_train)
    #     ro_br = torch.ones(N_train)

    #     X_bc.append(outlet_br)
    #     U_bc.append(u_br)
    #     V_bc.append(v_br)
    #     ro_bc.append(ro_br)

    #     # ------------------------------------------------------------
    #     # 3) TOP & BOTTOM BOUNDARIES AROUND THE ENTIRE SHAPE
    #     #    (x from -2*l to L + 2*l)
    #     # ------------------------------------------------------------
    #     # a) Top boundary: y = l
    #     top_boundary = boundary_line_y(l, -2.0 * l, L + 2.0 * l, 2 * N_train)
    #     u_top = torch.zeros(top_boundary.shape[0])
    #     v_top = torch.zeros(top_boundary.shape[0])
    #     ro_top = torch.ones(top_boundary.shape[0])

    #     X_bc.append(top_boundary)
    #     U_bc.append(u_top)
    #     V_bc.append(v_top)
    #     ro_bc.append(ro_top)

    #     # b) Bottom boundary: y = 0
    #     bottom_boundary = boundary_line_y(0.0, -2.0 * l, L + 2.0 * l, 2 * N_train)
    #     u_bot = torch.zeros(bottom_boundary.shape[0])
    #     v_bot = torch.zeros(bottom_boundary.shape[0])
    #     ro_bot = torch.ones(bottom_boundary.shape[0])

    #     X_bc.append(bottom_boundary)
    #     U_bc.append(u_bot)
    #     V_bc.append(v_bot)
    #     ro_bc.append(ro_bot)

    #     # ------------------------------------------------------------
    #     # CONCATENATE ALL BOUNDARY POINTS
    #     # ------------------------------------------------------------
    #     X_train = torch.cat(X_bc, dim=0)
    #     U_train = torch.cat(U_bc, dim=0)
    #     V_train = torch.cat(V_bc, dim=0)
    #     train_ro = torch.cat(ro_bc, dim=0)

    #     # If you also want a pressure boundary or special corner points, you can define them here.
    #     X_train_P = torch.tensor([[0.0, 0.0]], dtype=torch.float)
    #     train_P = torch.tensor([0.0], dtype=torch.float)


    if problem == 'four_terminals':
        
        
        g_bar = 10.0 
        # Define domains
        domain = {'x': [0, 1*ratio], 'y': [0.0, 1]}
        # Define the number of samples
        N_train_r =  int(N_train / 3) # For the new domain

        # Define the exclusion ranges
        exclude_range1_min = 1/8 - 1/12
        exclude_range1_max = 1/8 + 1/12
        exclude_range2_min = (1-1/8)- 1/12
        exclude_range2_max = (1-1/8)+ 1/12
        
        domain_r1 = {'x': [0.0, ratio], 'y': [exclude_range1_min , exclude_range1_max ]}
        domain_r2 = {'x': [0.0, ratio], 'y': [exclude_range2_min , exclude_range2_max ]}

        # Generate points excluding the overlapping region for the left boundary

        # Create boolean masks to exclude ranges
        mask1 = (points_y < exclude_range1_min) | (points_y > exclude_range1_max)
        mask2 = (points_y < exclude_range2_min) | (points_y > exclude_range2_max)
        mask = mask1 & mask2

        # Apply mask to points_y
        points_y_filtered = points_y[mask]

        # Create x_left and related tensors
        x_left = torch.stack([domain['x'][0] * torch.ones(len(points_y_filtered)), points_y_filtered.squeeze()], dim=1)
        u_left = torch.zeros_like(x_left[:, 0])
        v_left = torch.zeros_like(x_left[:, 0])
        ro_left = torch.zeros_like(x_left[:, 0])

        # Calculate the length of each exclusion range
        length_range1 = exclude_range1_max - exclude_range1_min
        length_range2 = exclude_range2_max - exclude_range2_min
        total_length = length_range1 + length_range2

        # Calculate the number of samples needed for each range
        num_samples_range1 = int((length_range1 / total_length) * N_train_r)
        num_samples_range2 = N_train_r - num_samples_range1  # Ensure total is exactly 100

        # Generate the required number of samples within each range
        points_y_r1 = torch.linspace(exclude_range1_min, exclude_range1_max, num_samples_range1 + 2)[1:-1]
        points_y_r2 = torch.linspace(exclude_range2_min, exclude_range2_max, num_samples_range2 + 2)[1:-1]

        # Stack the points to create x_left_r
        x_left_r1 = torch.stack([domain_r1['x'][0] * torch.ones(num_samples_range1), points_y_r1], dim=1)

        # Parameters for the velocity profile for the left boundar

        l = domain_r1['y'][1] - domain_r1['y'][0]
        t = points_y_r1 - (domain_r1['y'][0] + l / 2)

        # Compute the velocity profile for the left boundary
        u_left_r1 =g_bar * (1 - (2 * t / l) ** 2)
        v_left_r1 = torch.zeros_like(u_left_r1)
        ro_left_r1 = 1+ torch.zeros_like(u_left_r1)
        
        # Stack the points to create x_left_r
        x_left_r2 = torch.stack([domain_r2['x'][0] * torch.ones(num_samples_range2), points_y_r2], dim=1)
        # Parameters for the velocity profile for the left boundary
        l = domain_r2['y'][1] - domain_r2['y'][0]
        t = points_y_r2 - (domain_r2['y'][0] + l / 2)

        # Compute the velocity profile for the left boundary
        u_left_r2 =-g_bar * (1 - (2 * t / l) ** 2)
        v_left_r2 = torch.zeros_like(u_left_r2)
        ro_left_r2 = 1+ torch.zeros_like(u_left_r2)

        # Concatenate the new samples for the left boundary
        x_left_combined = torch.cat((x_left, x_left_r1, x_left_r2), dim=0)
        u_left_combined = torch.cat((u_left, u_left_r1, u_left_r2), dim=0)
        v_left_combined = torch.cat((v_left, v_left_r1, v_left_r2), dim=0)
        ro_left_combined = torch.cat((ro_left, ro_left_r1, ro_left_r2), dim=0)

        # Original domain and points for the bottom boundary
        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(len(points_x))], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])
        ro_bottom = torch.zeros_like(x_bottom[:,0])


        # The combined tensors now contain the samples from both domains without overlap
        x_top = torch.stack([points_x.squeeze(), torch.ones(len(points_x))], dim=1)
        u_top =torch.zeros_like(x_top[:,0])#5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:,0])
        ro_top = torch.zeros_like(x_top[:,0])


        # Create x_left and related tensors
        x_right = torch.stack([domain['x'][1] * torch.ones(len(points_y_filtered)), points_y_filtered.squeeze()], dim=1)
        u_right = torch.zeros_like(x_right[:, 0])
        v_right = torch.zeros_like(x_right[:, 0])
        ro_right = torch.zeros_like(x_right[:, 0])

        # Calculate the length of each exclusion range
        length_range1 = exclude_range1_max - exclude_range1_min
        length_range2 = exclude_range2_max - exclude_range2_min
        total_length = length_range1 + length_range2

        # Calculate the number of samples needed for each range
        num_samples_range1 = int((length_range1 / total_length) * N_train_r)
        num_samples_range2 = N_train_r - num_samples_range1  # Ensure total is exactly 100

        # Generate the required number of samples within each range
        points_y_r1 = torch.linspace(exclude_range1_min, exclude_range1_max, num_samples_range1 + 2)[1:-1]
        points_y_r2 = torch.linspace(exclude_range2_min, exclude_range2_max, num_samples_range2 + 2)[1:-1]

        # Stack the points to create x_right_r
        x_right_r1 = torch.stack([domain_r1['x'][1] * torch.ones(num_samples_range1), points_y_r1], dim=1)

        # Parameters for the velocity profile for the right boundar

        l = domain_r1['y'][1] - domain_r1['y'][0]
        t = points_y_r1 - (domain_r1['y'][0] + l / 2)

        # Compute the velocity profile for the right boundary
        u_right_r1 =g_bar * (1 - (2 * t / l) ** 2)
        v_right_r1 = torch.zeros_like(u_right_r1)
        ro_right_r1 = 1+ torch.zeros_like(u_right_r1)
        
        # Stack the points to create x_right_r
        x_right_r2 = torch.stack([domain_r2['x'][1] * torch.ones(num_samples_range2), points_y_r2], dim=1)
        # Parameters for the velocity profile for the right boundary
        l = domain_r2['y'][1] - domain_r2['y'][0]
        t = points_y_r2 - (domain_r2['y'][0] + l / 2)

        # Compute the velocity profile for the right boundary
        u_right_r2 = -g_bar * (1 - (2 * t / l) ** 2)
        v_right_r2 = torch.zeros_like(u_right_r2)
        ro_right_r2 = 1+ torch.zeros_like(u_right_r2)

        # Concatenate the new samples for the right boundary
        x_right_combined = torch.cat((x_right, x_right_r1, x_right_r2), dim=0)
        u_right_combined = torch.cat((u_right, u_right_r1, u_right_r2), dim=0)
        v_right_combined = torch.cat((v_right, v_right_r1, v_right_r2), dim=0)
        ro_right_combined = torch.cat((ro_right, ro_right_r1, ro_right_r2), dim=0)
        
        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, 1.0],[ratio, 0.0],[ratio, 1.0]])
        u_corners = torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])
        ro_corners = torch.zeros_like(x_corners[:,0])
    
        X_train = torch.cat([x_top, x_left_combined, x_bottom, x_right_combined, x_corners ], dim=0)
        U_train = torch.cat([u_top, u_left_combined, u_bottom, u_right_combined, u_corners], dim=0)
        V_train = torch.cat([v_top, v_left_combined, v_bottom, v_right_combined, v_corners], dim=0)
        train_ro = torch.cat([ro_top, ro_left_combined, ro_bottom, ro_right_combined, ro_corners ], dim=0)#, ro_center
        
        
        X_train_P = torch.tensor([[0, 0]])
        train_P =torch.tensor([0])
        
        
    if problem == 'pipebend':
        N_train_r = int(torch.floor(torch.tensor(N_train / 5)).item())  # 
        N_train_b = int(torch.floor(torch.tensor(N_train / 5)).item()) # 

        # Points excluding the overlapping region [0.7, 0.9] for the left boundary
        points_y_filtered = points_y[(points_y < 0.7) | (points_y > 0.9)]
        x_left = torch.stack([domain['x'][0]*torch.ones(len(points_y_filtered)), points_y_filtered.squeeze()], dim=1)
        u_left = torch.zeros_like(x_left[:,0])
        v_left = torch.zeros_like(x_left[:,0])
        ro_left = torch.zeros_like(x_left[:,0])

        # New domain and points for the left boundary
        domain_r = {'x':[0.0, 0.0], 'y':[0.7, 0.9]}
        points_y_r = torch.linspace(domain_r['y'][0], domain_r['y'][1], N_train_r+2)[1:-1]
        x_left_r = torch.stack([domain_r['x'][1]*torch.ones(N_train_r), points_y_r.squeeze()], dim=1)

        # Parameters for the velocity profile for the left boundary
        l = domain_r['y'][1] - domain_r['y'][0]
        t = points_y_r - (domain_r['y'][0] + l / 2)
        g_bar = 10.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the left boundary
        u_left_r = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r = torch.zeros_like(u_left_r)
        ro_left_r = 1+ torch.zeros_like(u_left_r)

        # Concatenate the new samples for the left boundary
        x_left_combined = torch.cat((x_left, x_left_r), dim=0)
        u_left_combined = torch.cat((u_left, u_left_r), dim=0)
        v_left_combined = torch.cat((v_left, v_left_r), dim=0)
        ro_left_combined = torch.cat((ro_left, ro_left_r), dim=0)

        # Original domain and points for the bottom boundary
        points_x_filtered = points_x[(points_x < 0.7) | (points_x > 0.9)]
        x_bottom = torch.stack([points_x_filtered.squeeze(), torch.zeros(len(points_x_filtered))], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])
        ro_bottom = torch.zeros_like(x_bottom[:,0])

        # New domain and points for the bottom boundary
        domain_b = {'x':[0.7, 0.9], 'y':[0, 0]}
        points_x_b = torch.linspace(domain_b['x'][0], domain_b['x'][1], N_train_b+2)[1:-1]
        x_bottom_b = torch.stack([points_x_b.squeeze(), torch.zeros(N_train_b)], dim=1)

        # Parameters for the velocity profile for the bottom boundary
        l_b = domain_b['x'][1] - domain_b['x'][0]
        t_b = points_x_b - (domain_b['x'][0] + l_b / 2)

        # Compute the velocity profile for the bottom boundary
        v_bar = 100.0  # Example value for the magnitude of the flow velocity at the center
        v_bottom_b = -v_bar * (1 - (2 * t_b / l_b) ** 2)
        u_bottom_b = torch.zeros_like(v_bottom_b)
        ro_bottom_b =1+ torch.zeros_like(v_bottom_b)

        # Concatenate the new samples for the bottom boundary
        x_bottom_combined = torch.cat((x_bottom, x_bottom_b), dim=0)
        u_bottom_combined = torch.cat((u_bottom, u_bottom_b), dim=0)
        v_bottom_combined = torch.cat((v_bottom, v_bottom_b), dim=0)
        ro_bottom_combined = torch.cat((ro_bottom, ro_bottom_b), dim=0)

        # The combined tensors now contain the samples from both domains without overlap

        x_top = torch.stack([points_x.squeeze(), torch.ones(N_train)], dim=1)
        u_top =torch.zeros_like(x_top[:,0])#5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:,0])
        ro_top = torch.zeros_like(x_top[:,0])

        points_y = torch.linspace(domain['y'][0], domain['y'][1], N_train+2)[1:-1]
        x_right = torch.stack([torch.ones(N_train), points_y.squeeze()], dim=1)
        u_right = torch.zeros_like(x_right[:,0])
        v_right = torch.zeros_like(x_right[:,0])
        ro_right = torch.zeros_like(x_right[:,0])

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
        u_corners = torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])
        ro_corners = torch.zeros_like(x_corners[:,0])
        

        X_train = torch.cat([x_top, x_left_combined, x_bottom_combined, x_right, x_corners], dim=0)
        U_train = torch.cat([u_top, u_left_combined, u_bottom_combined, u_right, u_corners], dim=0)
        V_train = torch.cat([v_top, v_left_combined, v_bottom_combined, v_right, v_corners], dim=0)
        train_ro = torch.cat([ro_top, ro_left_combined, ro_bottom_combined, ro_right, ro_corners], dim=0)
        
                
        X_train_P = torch.tensor([[0, 0]])
        train_P =torch.tensor([0])


    elif problem == 'diffuser':
        N_train_r = N_train//3  # Adjust N_train_r as needed for the new domain
        N_train_l = N_train  # Adjust N_train_b as needed for the new domain
                

        # New domain and points for the left boundary
        domain_l = {'x':[0.0, 0.0], 'y':[0, 1]}
        points_y_r = torch.linspace(domain_l['y'][0], domain_l['y'][1], N_train_l+2)[1:-1]
        x_left_r = torch.stack([domain_l['x'][1]*torch.ones(N_train_l), points_y_r.squeeze()], dim=1)

        # Parameters for the velocity profile for the left boundary
        l = domain_l['y'][1] - domain_l['y'][0]
        t = points_y_r - (domain_l['y'][0] + l / 2)
        g_bar = 1.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the left boundary
        u_left_r = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r = torch.zeros_like(u_left_r)
        ro_left_r = 1+ torch.zeros_like(u_left_r)

        # Original domain and points for the bottom boundary
        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(len(points_x))], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])
        ro_bottom = torch.zeros_like(x_bottom[:,0])

        # The combined tensors now contain the samples from both domains without overlap

        x_top = torch.stack([points_x.squeeze(), torch.ones(N_train)], dim=1)
        u_top =torch.zeros_like(x_top[:,0])#5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:,0])
        ro_top = torch.zeros_like(x_top[:,0])


        points_y_filtered = points_y[(points_y < 0.333) | (points_y > 0.666)]
        x_right = torch.stack([domain['x'][1]*torch.ones(len(points_y_filtered)), points_y_filtered.squeeze()], dim=1)
        u_right = torch.zeros_like(x_right[:,0])
        v_right = torch.zeros_like(x_right[:,0])
        ro_right = torch.zeros_like(x_right[:,0])

        # New domain and points for the left boundary
        domain_r = {'x':[1, 1], 'y':[0.333, 0.666]}
        points_y_l = torch.linspace(domain_r['y'][0], domain_r['y'][1], N_train_r+2)[1:-1]
        x_right_l = torch.stack([domain_r['x'][1]*torch.ones(N_train_r), points_y_l.squeeze()], dim=1)

        # Parameters for the velocity profile for the left boundary
        l = domain_r['y'][1] - domain_r['y'][0]
        t = points_y_l - (domain_r['y'][0] + l / 2)
        g_bar = 3.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the left boundary
        u_right_l = g_bar * (1 - (2 * t / l) ** 2)
        v_right_l = torch.zeros_like(u_right_l)
        ro_right_l = 1+ torch.zeros_like(u_right_l)

        # Concatenate the new samples for the left boundary
        x_right_combined = torch.cat((x_right, x_right_l), dim=0)
        u_right_combined = torch.cat((u_right, u_right_l), dim=0)
        v_right_combined = torch.cat((v_right, v_right_l), dim=0)
        ro_right_combined = torch.cat((ro_right, ro_right_l), dim=0)
        
        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
        u_corners = torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])
        ro_corners = torch.zeros_like(x_corners[:,0])
        

        X_train = torch.cat([x_top, x_left_r, x_bottom, x_right_combined, x_corners], dim=0)
        U_train = torch.cat([u_top, u_left_r, u_bottom, u_right_combined, u_corners], dim=0)
        V_train = torch.cat([v_top, v_left_r, v_bottom, v_right_combined, v_corners], dim=0)
        train_ro = torch.cat([ro_top, ro_left_r, ro_bottom, ro_right_combined, ro_corners], dim=0)
        
        X_train_P = torch.tensor([[0, 0]])
        train_P =torch.tensor([0])

    elif problem == 'rugby_large':
        
        
        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(int(ratio*N_train))], dim=1)
        u_bottom =1+ torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])

        x_top = torch.stack([points_x.squeeze(), torch.ones(int(ratio*N_train))], dim=1)
        u_top =1+ torch.zeros_like(x_top[:,0])#5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:,0])

        x_right = torch.stack([domain['x'][1]*torch.ones(N_train), points_y.squeeze()], dim=1)
        u_right =1+ torch.zeros_like(x_right[:,0])
        v_right = torch.zeros_like(x_right[:,0])

        x_left = torch.stack([torch.zeros(N_train), points_y.squeeze()], dim=1)
        u_left = 1+torch.zeros_like(x_left[:,0])
        v_left = torch.zeros_like(x_left[:,0])

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, ratio*1.0],[ratio*1.0, 0.0],[ratio*1.0, ratio*1.0]])
        u_corners = 1+torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])

        X_train = torch.cat([x_top, x_right, x_bottom, x_left, x_corners], dim=0)
        U_train = torch.cat([u_top, u_right, u_bottom, u_left, u_corners], dim=0)
        V_train = torch.cat([v_top, v_right, v_bottom, v_left, v_corners], dim=0)
        
        
        middel_info=False
        if middel_info:
        
            X_train=torch.cat((X_train, torch.tensor([[ratio*0.5, 0.5]])), dim=0)
            # X_train=torch.cat((X_train, torch.tensor([[0.5051, 0.5051]])), dim=0)

            X_train_U = X_train
            X_train_V = X_train
            train_ro_1=V_train.clone()
            train_ro = torch.cat((1 + 0 * train_ro_1, torch.tensor([0])), dim=0)
            U_train = torch.cat((U_train, torch.tensor([0])), dim=0)
            V_train = torch.cat((V_train, torch.tensor([0])), dim=0)
        else:

            X_train_U = X_train
            X_train_V = X_train
            train_ro_1=V_train.clone()
            train_ro =1 + 0 * train_ro_1
        
        X_train_P = torch.tensor([[0, 0]])
        train_P =torch.tensor([0])


    elif problem == 'rugby':
        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(int(ratio*N_train))], dim=1)
        u_bottom =1+ torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])

        x_top = torch.stack([points_x.squeeze(), torch.ones(int(ratio*N_train))], dim=1)
        u_top =1+ torch.zeros_like(x_top[:,0])#5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:,0])

        x_right = torch.stack([domain['x'][1]*torch.ones(N_train), points_y.squeeze()], dim=1)
        u_right =1+ torch.zeros_like(x_right[:,0])
        v_right = torch.zeros_like(x_right[:,0])

        x_left = torch.stack([torch.zeros(N_train), points_y.squeeze()], dim=1)
        u_left = 1+torch.zeros_like(x_left[:,0])
        v_left = torch.zeros_like(x_left[:,0])

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, ratio*1.0],[ratio*1.0, 0.0],[ratio*1.0, ratio*1.0]])
        u_corners = 1+torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])

        X_train = torch.cat([x_top, x_right, x_bottom, x_left, x_corners], dim=0)
        U_train = torch.cat([u_top, u_right, u_bottom, u_left, u_corners], dim=0)
        V_train = torch.cat([v_top, v_right, v_bottom, v_left, v_corners], dim=0)
        
        
        middel_info=True
        if middel_info:
            X_train=torch.cat((X_train, torch.tensor([[ratio*0.5, 0.5]])), dim=0)
            # X_train=torch.cat((X_train, torch.tensor([[0.5051, 0.5051]])), dim=0)

            X_train_U = X_train
            X_train_V = X_train
            train_ro_1=V_train.clone()
            train_ro = torch.cat((1 + 0 * train_ro_1, torch.tensor([0])), dim=0)
            U_train = torch.cat((U_train, torch.tensor([0])), dim=0)
            V_train = torch.cat((V_train, torch.tensor([0])), dim=0)
        else:

            X_train_U = X_train
            X_train_V = X_train
            train_ro_1=V_train.clone()
            train_ro =1 + 0 * train_ro_1
        
        X_train_P = torch.tensor([[0, 0]])
        train_P =torch.tensor([0])
    elif problem == 'doublepipe':


        # Define domains
        domain = {'x': [0, 1*ratio], 'y': [0.0, 1]}
        # Define the number of samples
        N_train_r =  int(N_train / 3) # For the new domain

        # Define the exclusion ranges
        exclude_range1_min = 1/4 - 1/12
        exclude_range1_max = 1/4 + 1/12
        exclude_range2_min = 0.666
        exclude_range2_max = 0.666 + 1/6
        
        domain_r1 = {'x': [0.0, ratio], 'y': [exclude_range1_min , exclude_range1_max ]}
        domain_r2 = {'x': [0.0, ratio], 'y': [exclude_range2_min , exclude_range2_max ]}

        # Generate points excluding the overlapping region for the left boundary

        # Create boolean masks to exclude ranges
        mask1 = (points_y < exclude_range1_min) | (points_y > exclude_range1_max)
        mask2 = (points_y < exclude_range2_min) | (points_y > exclude_range2_max)
        mask = mask1 & mask2

        # Apply mask to points_y
        points_y_filtered = points_y[mask]

        # Create x_left and related tensors
        x_left = torch.stack([domain['x'][0] * torch.ones(len(points_y_filtered)), points_y_filtered.squeeze()], dim=1)
        u_left = torch.zeros_like(x_left[:, 0])
        v_left = torch.zeros_like(x_left[:, 0])
        ro_left = torch.zeros_like(x_left[:, 0])

        # Calculate the length of each exclusion range
        length_range1 = exclude_range1_max - exclude_range1_min
        length_range2 = exclude_range2_max - exclude_range2_min
        total_length = length_range1 + length_range2

        # Calculate the number of samples needed for each range
        num_samples_range1 = int((length_range1 / total_length) * N_train_r)
        num_samples_range2 = N_train_r - num_samples_range1  # Ensure total is exactly 100

        # Generate the required number of samples within each range
        points_y_r1 = torch.linspace(exclude_range1_min, exclude_range1_max, num_samples_range1 + 2)[1:-1]
        points_y_r2 = torch.linspace(exclude_range2_min, exclude_range2_max, num_samples_range2 + 2)[1:-1]

        # Stack the points to create x_left_r
        x_left_r1 = torch.stack([domain_r1['x'][0] * torch.ones(num_samples_range1), points_y_r1], dim=1)

        # Parameters for the velocity profile for the left boundar

        l = domain_r1['y'][1] - domain_r1['y'][0]
        t = points_y_r1 - (domain_r1['y'][0] + l / 2)
        g_bar = 1.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the left boundary
        u_left_r1 = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r1 = torch.zeros_like(u_left_r1)
        ro_left_r1 = 1+ torch.zeros_like(u_left_r1)
        
        # Stack the points to create x_left_r
        x_left_r2 = torch.stack([domain_r2['x'][0] * torch.ones(num_samples_range2), points_y_r2], dim=1)
        # Parameters for the velocity profile for the left boundary
        l = domain_r2['y'][1] - domain_r2['y'][0]
        t = points_y_r2 - (domain_r2['y'][0] + l / 2)
        g_bar = 1.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the left boundary
        u_left_r2 = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r2 = torch.zeros_like(u_left_r2)
        ro_left_r2 = 1+ torch.zeros_like(u_left_r2)

        # Concatenate the new samples for the left boundary
        x_left_combined = torch.cat((x_left, x_left_r1, x_left_r2), dim=0)
        u_left_combined = torch.cat((u_left, u_left_r1, u_left_r2), dim=0)
        v_left_combined = torch.cat((v_left, v_left_r1, v_left_r2), dim=0)
        ro_left_combined = torch.cat((ro_left, ro_left_r1, ro_left_r2), dim=0)

        # Original domain and points for the bottom boundary
        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(len(points_x))], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])
        ro_bottom = torch.zeros_like(x_bottom[:,0])


        # The combined tensors now contain the samples from both domains without overlap
        x_top = torch.stack([points_x.squeeze(), torch.ones(len(points_x))], dim=1)
        u_top =torch.zeros_like(x_top[:,0])#5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:,0])
        ro_top = torch.zeros_like(x_top[:,0])


        # Create x_left and related tensors
        x_right = torch.stack([domain['x'][1] * torch.ones(len(points_y_filtered)), points_y_filtered.squeeze()], dim=1)
        u_right = torch.zeros_like(x_right[:, 0])
        v_right = torch.zeros_like(x_right[:, 0])
        ro_right = torch.zeros_like(x_right[:, 0])

        # Calculate the length of each exclusion range
        length_range1 = exclude_range1_max - exclude_range1_min
        length_range2 = exclude_range2_max - exclude_range2_min
        total_length = length_range1 + length_range2

        # Calculate the number of samples needed for each range
        num_samples_range1 = int((length_range1 / total_length) * N_train_r)
        num_samples_range2 = N_train_r - num_samples_range1  # Ensure total is exactly 100

        # Generate the required number of samples within each range
        points_y_r1 = torch.linspace(exclude_range1_min, exclude_range1_max, num_samples_range1 + 2)[1:-1]
        points_y_r2 = torch.linspace(exclude_range2_min, exclude_range2_max, num_samples_range2 + 2)[1:-1]

        # Stack the points to create x_right_r
        x_right_r1 = torch.stack([domain_r1['x'][1] * torch.ones(num_samples_range1), points_y_r1], dim=1)

        # Parameters for the velocity profile for the right boundar

        l = domain_r1['y'][1] - domain_r1['y'][0]
        t = points_y_r1 - (domain_r1['y'][0] + l / 2)
        g_bar = 1.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the right boundary
        u_right_r1 = g_bar * (1 - (2 * t / l) ** 2)
        v_right_r1 = torch.zeros_like(u_right_r1)
        ro_right_r1 = 1+ torch.zeros_like(u_right_r1)
        
        # Stack the points to create x_right_r
        x_right_r2 = torch.stack([domain_r2['x'][1] * torch.ones(num_samples_range2), points_y_r2], dim=1)
        # Parameters for the velocity profile for the right boundary
        l = domain_r2['y'][1] - domain_r2['y'][0]
        t = points_y_r2 - (domain_r2['y'][0] + l / 2)
        g_bar = 1.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the right boundary
        u_right_r2 = g_bar * (1 - (2 * t / l) ** 2)
        v_right_r2 = torch.zeros_like(u_right_r2)
        ro_right_r2 = 1+ torch.zeros_like(u_right_r2)

        # Concatenate the new samples for the right boundary
        x_right_combined = torch.cat((x_right, x_right_r1, x_right_r2), dim=0)
        u_right_combined = torch.cat((u_right, u_right_r1, u_right_r2), dim=0)
        v_right_combined = torch.cat((v_right, v_right_r1, v_right_r2), dim=0)
        ro_right_combined = torch.cat((ro_right, ro_right_r1, ro_right_r2), dim=0)
        
        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, 1.0],[ratio, 0.0],[ratio, 1.0]])
        u_corners = torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])
        ro_corners = torch.zeros_like(x_corners[:,0])
    
        X_train = torch.cat([x_top, x_left_combined, x_bottom, x_right_combined, x_corners ], dim=0)
        U_train = torch.cat([u_top, u_left_combined, u_bottom, u_right_combined, u_corners], dim=0)
        V_train = torch.cat([v_top, v_left_combined, v_bottom, v_right_combined, v_corners], dim=0)
        train_ro = torch.cat([ro_top, ro_left_combined, ro_bottom, ro_right_combined, ro_corners ], dim=0)#, ro_center
        
        
        X_train_P = torch.tensor([[0, 0]])
        train_P =torch.tensor([0])
        

    
    elif problem == 'pipe_with_force_term':
        N_train_r = N_train//6  # Adjust N_train_r as needed for the new domain
        N_train_l = N_train//6  # Adjust N_train_b as needed for the new domain
                
        # New domain and points for the left boundary
        domain_r = {'x':[0.0, 0.0], 'y':[0.58333, 0.75]}
        points_y_r = torch.linspace(domain_r['y'][0], domain_r['y'][1], N_train_r+2)[1:-1]
        x_left_r = torch.stack([domain_r['x'][1]*torch.ones(N_train_r), points_y_r.squeeze()], dim=1)

        # Parameters for the velocity profile for the left boundary
        l = domain_r['y'][1] - domain_r['y'][0]
        t = points_y_r - (domain_r['y'][0] + l / 2)
        g_bar = 1.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the left boundary
        u_left_r = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r = torch.zeros_like(u_left_r)
        ro_left_r = 1+ torch.zeros_like(u_left_r)

        points_y_filtered = points_y[(points_y < 0.58333) | (points_y > 0.75)]
        x_left = torch.stack([domain['x'][0]*torch.ones(len(points_y_filtered)), points_y_filtered.squeeze()], dim=1)
        u_left = torch.zeros_like(x_left[:,0])
        v_left = torch.zeros_like(x_left[:,0])
        ro_left = torch.zeros_like(x_left[:,0])
        
        # Concatenate the new samples for the left boundary
        x_left_combined = torch.cat((x_left, x_left_r), dim=0)
        u_left_combined = torch.cat((u_left, u_left_r), dim=0)
        v_left_combined = torch.cat((v_left, v_left_r), dim=0)
        ro_left_combined = torch.cat((ro_left, ro_left_r), dim=0)
        
        ####################################################

        # New domain and points for the left boundary
        domain_r = {'x':[1.0, 1.0], 'y':[0.58333, 0.75]}
        points_y_l = torch.linspace(domain_r['y'][0], domain_r['y'][1], N_train_r+2)[1:-1]
        x_right_l = torch.stack([domain_r['x'][1]*torch.ones(N_train_r), points_y_l.squeeze()], dim=1)

        # Parameters for the velocity profile for the left boundary
        l = domain_r['y'][1] - domain_r['y'][0]
        t = points_y_l - (domain_r['y'][0] + l / 2)
        g_bar = 1.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the left boundary
        u_right_l = g_bar * (1 - (2 * t / l) ** 2)
        v_right_l = torch.zeros_like(u_right_l)
        ro_right_l = 1+ torch.zeros_like(u_right_l)


        x_right = torch.stack([domain['x'][1]*torch.ones(len(points_y_filtered)), points_y_filtered.squeeze()], dim=1)
        u_right = torch.zeros_like(x_right[:,0])
        v_right = torch.zeros_like(x_right[:,0])
        ro_right = torch.zeros_like(x_right[:,0])
        
        # Concatenate the new samples for the right boundary
        x_right_combined = torch.cat((x_right, x_right_l), dim=0)
        u_right_combined = torch.cat((u_right, u_right_l), dim=0)
        v_right_combined = torch.cat((v_right, v_right_l), dim=0)
        ro_right_combined = torch.cat((ro_right, ro_right_l), dim=0)

        # Original domain and points for the bottom boundary
        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(len(points_x))], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])
        ro_bottom = torch.zeros_like(x_bottom[:,0])

        # The combined tensors now contain the samples from both domains without overlap

        x_top = torch.stack([points_x.squeeze(), torch.ones(N_train)], dim=1)
        u_top =torch.zeros_like(x_top[:,0])#5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:,0])
        ro_top = torch.zeros_like(x_top[:,0])

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
        u_corners = torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])
        ro_corners = torch.zeros_like(x_corners[:,0])
        

        X_train = torch.cat([x_top, x_left_combined, x_bottom, x_right_combined, x_corners], dim=0)
        U_train = torch.cat([u_top, u_left_combined, u_bottom, u_right_combined, u_corners], dim=0)
        V_train = torch.cat([v_top, v_left_combined, v_bottom, v_right_combined, v_corners], dim=0)
        train_ro = torch.cat([ro_top, ro_left_combined, ro_bottom, ro_right_combined, ro_corners], dim=0)

        
    if problem == 'pip_with_obstacle':
        N_train_r = int(torch.floor(torch.tensor(N_train / 5)).item())  # 
        N_train_b = int(torch.floor(torch.tensor(N_train / 5)).item()) # 

        # Points excluding the overlapping region [0.7, 0.9] for the left boundary
        points_y_filtered = points_y[(points_y < 0.4) | (points_y > 0.6)]
        x_left = torch.stack([domain['x'][0]*torch.ones(len(points_y_filtered)), points_y_filtered.squeeze()], dim=1)
        u_left = torch.zeros_like(x_left[:,0])
        v_left = torch.zeros_like(x_left[:,0])
        ro_left = torch.zeros_like(x_left[:,0])

        # New domain and points for the left boundary
        domain_r = {'x':[0.0, 0.0], 'y':[0.4, 0.6]}
        points_y_r = torch.linspace(domain_r['y'][0], domain_r['y'][1], N_train_r+2)[1:-1]
        x_left_r = torch.stack([domain_r['x'][1]*torch.ones(N_train_r), points_y_r.squeeze()], dim=1)

        # Parameters for the velocity profile for the left boundary
        l = domain_r['y'][1] - domain_r['y'][0]
        t = points_y_r - (domain_r['y'][0] + l / 2)
        g_bar = 1.0  # Example value for the magnitude of the flow velocity at the center

        # Compute the velocity profile for the left boundary
        u_left_r = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r = torch.zeros_like(u_left_r)
        ro_left_r = 1+ torch.zeros_like(u_left_r)

        # Concatenate the new samples for the left boundary
        x_left_combined = torch.cat((x_left, x_left_r), dim=0)
        u_left_combined = torch.cat((u_left, u_left_r), dim=0)
        v_left_combined = torch.cat((v_left, v_left_r), dim=0)
        ro_left_combined = torch.cat((ro_left, ro_left_r), dim=0)

        # Original domain and points for the bottom boundary
        points_x_filtered = points_x[(points_x < 0.6) | (points_x > 0.8)]
        x_bottom = torch.stack([points_x_filtered.squeeze(), torch.zeros(len(points_x_filtered))], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])
        ro_bottom = torch.zeros_like(x_bottom[:,0])

        # New domain and points for the bottom boundary
        domain_b = {'x':[0.6, 0.8], 'y':[0, 0]}
        points_x_b = torch.linspace(domain_b['x'][0], domain_b['x'][1], N_train_b+2)[1:-1]
        x_bottom_b = torch.stack([points_x_b.squeeze(), torch.zeros(N_train_b)], dim=1)

        # Parameters for the velocity profile for the bottom boundary
        l_b = domain_b['x'][1] - domain_b['x'][0]
        t_b = points_x_b - (domain_b['x'][0] + l_b / 2)

        # Compute the velocity profile for the bottom boundary
        v_bar = g_bar/2 # Example value for the magnitude of the flow velocity at the center
        v_bottom_b = -v_bar * (1 - (2 * t_b / l_b) ** 2)
        u_bottom_b = torch.zeros_like(v_bottom_b)
        ro_bottom_b =1+ torch.zeros_like(v_bottom_b)

        # Concatenate the new samples for the bottom boundary
        x_bottom_combined = torch.cat((x_bottom, x_bottom_b), dim=0)
        u_bottom_combined = torch.cat((u_bottom, u_bottom_b), dim=0)
        v_bottom_combined = torch.cat((v_bottom, v_bottom_b), dim=0)
        ro_bottom_combined = torch.cat((ro_bottom, ro_bottom_b), dim=0)

        # The combined tensors now contain the samples from both domains without overlap

        # New domain and points for the bottom boundary
        domain_t = {'x':[0.6, 0.8], 'y':[1, 1]}
        points_x_t = torch.linspace(domain_t['x'][0], domain_t['x'][1], N_train_b+2)[1:-1]
        x_top_t = torch.stack([points_x_t.squeeze(), 1+torch.zeros(N_train_b)], dim=1)

        # Parameters for the velocity profile for the bottom boundary
        l_t = domain_t['x'][1] - domain_t['x'][0]
        t_t = points_x_b - (domain_t['x'][0] + l_b / 2)

        # Compute the velocity profile for the bottom boundary
        v_bar = g_bar/2  # Example value for the magnitude of the flow velocity at the center
        v_top_t = v_bar * (1 - (2 * t_t / l_t) ** 2)
        u_top_t = torch.zeros_like(v_top_t)
        ro_top_t =1+ torch.zeros_like(v_top_t)
        
        points_x_filtered = points_x[(points_x < 0.6) | (points_x > 0.8)]
        x_top = torch.stack([points_x_filtered.squeeze(), torch.ones(len(points_x_filtered))], dim=1)
        u_top =torch.zeros_like(x_top[:,0])#5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:,0])
        ro_top = torch.zeros_like(x_top[:,0])
        
        # Concatenate the new samples for the bottom boundary
        x_top_combined = torch.cat((x_top, x_top_t), dim=0)
        u_top_combined = torch.cat((u_top, u_top_t), dim=0)
        v_top_combined = torch.cat((v_top, v_top_t), dim=0)
        ro_top_combined = torch.cat((ro_top, ro_top_t), dim=0)
        

        points_y = torch.linspace(domain['y'][0], domain['y'][1], N_train+2)[1:-1]
        x_right = torch.stack([torch.ones(N_train), points_y.squeeze()], dim=1)
        u_right = torch.zeros_like(x_right[:,0])
        v_right = torch.zeros_like(x_right[:,0])
        ro_right = torch.zeros_like(x_right[:,0])

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
        u_corners = torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])
        ro_corners = torch.zeros_like(x_corners[:,0])
        
        
        
                # Define the obstacle polygons with the provided coordinates
        pg_1 = [(0.20, 0.30), (0.25, 0.40), (0.35, 0.48), (0.33, 0.28), (0.25, 0.25)]
        pg_2 = [(0.55, 0.43), (0.63, 0.52), (0.70, 0.50), (0.68, 0.40), (0.58, 0.38)]
        pg_3 = [(0.83, 0.68), (0.87, 0.75), (0.95, 0.73), (0.93, 0.65), (0.84, 0.63)]
        obstacle_polygons = [pg_1, pg_2, pg_3]

        # Convert the coordinate lists into Shapely Polygon objects
        shapely_polygons = [Polygon(pg) for pg in obstacle_polygons]

        def in_polygon(points, polygon: Polygon):
            """
            Returns a boolean mask for which points in 'points' lie inside (or on) 'polygon'.
            """
            mask = []
            for xy in points:
                p = Point(float(xy[0]), float(xy[1]))
                mask.append(polygon.contains(p) or polygon.touches(p))
            return torch.tensor(mask, dtype=torch.bool)

        # Generate a set of points over the domain using Sobol sampling
        n = 2*torch.floor(torch.sqrt(torch.tensor(N_col_domain) / ratio)).int()
        N_sobol = int(n * n * ratio)  # Total number of points in 2D

        sobol = SobolEngine(dimension=2, scramble=True)
        X_unit = sobol.draw(N_sobol)  # Points in [0,1]^2

        # Scale points to the actual domain
        x_min, x_max = domain['x']
        y_min, y_max = domain['y']
        X_for_obst = torch.zeros_like(X_unit)
        X_for_obst[:, 0] = x_min + (x_max - x_min) * X_unit[:, 0]
        X_for_obst[:, 1] = y_min + (y_max - y_min) * X_unit[:, 1]

        # Create a combined mask for points that lie in any of the obstacles
        mask_obstacles = torch.zeros(X_for_obst.shape[0], dtype=torch.bool)
        for poly in shapely_polygons:
            mask_obstacles |= in_polygon(X_for_obst, poly)

        # Extract points that are inside any obstacle
        x_in_obstacle = X_for_obst[mask_obstacles]

        # Optionally, you might want to select a subset (e.g., first 100 points)
        if x_in_obstacle.shape[0] > 120:
            x_in_obstacle = x_in_obstacle[:120, :]

        # --- New Section: Sampling Border Points ---

        # Define how many points you want to sample along each obstacle's border
        points_per_border = 20  # Adjust this value as needed

        x_on_obstacle = []  # List to store border points

        for poly in shapely_polygons:
            # poly.exterior is a LinearRing representing the boundary of the polygon
            for i in range(points_per_border):
                # Compute a normalized parameter between 0 and 1
                t = i / points_per_border
                # Get a point along the boundary using interpolation
                point = poly.exterior.interpolate(t, normalized=True)
                x_on_obstacle.append((point.x, point.y))

        # Convert the list of border points to a torch tensor
        x_on_obstacle = torch.tensor(x_on_obstacle)

        x_in_obstacle = torch.cat((x_in_obstacle, x_on_obstacle), dim=0)

        # Zero out state variables (u, v, ρ) for points inside obstacles
        u_in_obstacle = torch.zeros_like(x_in_obstacle[:, 0])
        v_in_obstacle = torch.zeros_like(x_in_obstacle[:, 0])
        train_ro_in_obstacle = torch.zeros_like(x_in_obstacle[:, 0])


        X_train = torch.cat([x_top_combined, x_left_combined, x_bottom_combined, x_right, x_corners,x_in_obstacle], dim=0)
        U_train = torch.cat([u_top_combined, u_left_combined, u_bottom_combined, u_right, u_corners,u_in_obstacle], dim=0)
        V_train = torch.cat([v_top_combined, v_left_combined, v_bottom_combined, v_right, v_corners,v_in_obstacle], dim=0)
        train_ro = torch.cat([ro_top_combined, ro_left_combined, ro_bottom_combined, ro_right, ro_corners,train_ro_in_obstacle], dim=0)
        
        import matplotlib.pyplot as plt
        plt.scatter(X_train[:, 0], X_train[:, 1], c=V_train, cmap='viridis', s=5)
        plt.show()
                    
        X_train_P = torch.tensor([[0, 0]])
        train_P =torch.tensor([0])
        
        
        
    if problem == 'pip_with_thin_wall':
        H=1
        W=H*ratio
        L_t=0.35*W
        L_a=0.125*W
        w_w=0.025*W

        N_train_r = int(torch.floor(torch.tensor(N_train / 5)).item())  # 
        N_train_b = int(torch.floor(torch.tensor(N_train / 5)).item()) # 



        points_x = torch.linspace(domain['x'][0], domain['x'][1], N_train+2)[1:-1]
        # Points excluding the overlapping region [0.7, 0.9] for the left boundary
        x_left = torch.stack([domain['x'][0]*torch.ones(len(points_y)), points_y.squeeze()], dim=1)
        u_left = torch.zeros_like(x_left[:,0])
        v_left = torch.zeros_like(x_left[:,0])
        ro_left = torch.zeros_like(x_left[:,0])


        # Original domain and points for the bottom boundary
        
        # New domain and points for the bottom boundary
        domain_b = {'x':[L_a, L_a+L_t], 'y':[0, 0]}
        points_x_b = torch.linspace(domain_b['x'][0], domain_b['x'][1], N_train_b+2)[1:-1]
        x_bottom_b = torch.stack([points_x_b.squeeze(), torch.zeros(N_train_b)], dim=1)

        # Parameters for the velocity profile for the bottom boundary
        l_bb = points_x_b - L_a
        # Compute the velocity profile for the bottom boundary
        v_bottom_b = (l_bb/L_t)*(L_t-l_bb)
        u_bottom_b = torch.zeros_like(v_bottom_b)
        ro_bottom_b =1+ torch.zeros_like(v_bottom_b)


        # Original domain and points for the bottom boundary
        points_x_filtered = points_x[(points_x < L_a) | (points_x > (W-L_a))]
        x_bottom = torch.stack([points_x_filtered.squeeze(), torch.zeros(len(points_x_filtered))], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])
        ro_bottom = torch.zeros_like(x_bottom[:,0])

        # Concatenate the new samples for the bottom boundary
        x_bottom_combined = torch.cat((x_bottom, x_bottom_b), dim=0)
        u_bottom_combined = torch.cat((u_bottom, u_bottom_b), dim=0)
        v_bottom_combined = torch.cat((v_bottom, v_bottom_b), dim=0)
        ro_bottom_combined = torch.cat((ro_bottom, ro_bottom_b), dim=0)
        # The combined tensors now contain the samples from both domains without overla

        points_y = torch.linspace(domain['y'][0], domain['y'][1], N_train+2)[1:-1]
        x_right = torch.stack([W*torch.ones(N_train), points_y.squeeze()], dim=1)
        u_right = torch.zeros_like(x_right[:,0])
        v_right = torch.zeros_like(x_right[:,0])
        ro_right = torch.zeros_like(x_right[:,0])



        # Original domain and points for the top boundary
        points_x_filtered = points_x[(points_x < L_a) | (points_x > (L_a+L_t))]
        x_top = torch.stack([points_x_filtered.squeeze(), H*torch.ones(len(points_x_filtered))], dim=1)
        u_top = torch.zeros_like(x_top[:,0])
        v_top = torch.zeros_like(x_top[:,0])
        ro_top = torch.zeros_like(x_top[:,0])

        # The combined tensors now contain the samples from both domains without overla
        
        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, H],[W, 0.0],[W, H]])
        u_corners = torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])
        ro_corners = torch.zeros_like(x_corners[:,0])
        
        
    # Suppose your final training points are in X_train (N x 2) again

        def in_polygon(points, polygon: Polygon):
            """
            Returns a boolean mask for which points in 'points' lie inside (or on) 'polygon'.
            """
            # Shapely expects float coordinates, so we loop over each point
            mask = []
            for xy in points:
                p = Point(float(xy[0]), float(xy[1]))
                mask.append(polygon.contains(p) or polygon.touches(p))
            return torch.tensor(mask, dtype=torch.bool)
        
        # poly_obstacle3 = Polygon(pg_3)
        
        # Define the thin wall polygon (tweak coordinates as needed)
        thin_wall = [
            (W/2-(w_w/2), 0.00),  # Bottom-left corner
            (W/2+(w_w/2), 0.00),  # Bottom-right corner (adjust width for a "thin" wall)
            (W/2+(w_w/2), 0.25),  # Top-right corner
            (W/2-(w_w/2), 0.25)   # Top-left corner
        ]

        # Create a shapely polygon for the wall
        poly_thin_wall = Polygon(thin_wall)

        # n = torch.floor(torch.sqrt(torch.div(torch.tensor(N_col_domain), torch.tensor(ratio).int(), rounding_mode='trunc'))).int()
        # nx, ny = int(1*n*ratio),int(1*n)   # Change these values to your desired grid size
        # X_for_obst=gen_data(design_domain=domain, steps_x=nx,steps_y=ny)       
        n = torch.floor(torch.sqrt(torch.tensor(N_col_domain) / ratio)).int()
        N_sobol = n * n * ratio  # or any suitable logic to get the total number of 2D points
        N_sobol = int(N_sobol)

        # Initialize a Sobol sampler for 2D
        sobol = SobolEngine(dimension=2, scramble=True)

        # Draw points in [0,1]^2
        X_unit = sobol.draw(N_sobol)  # shape: (N_sobol, 2)

        # Scale to the actual domain
        x_min, x_max = domain['x']
        y_min, y_max = domain['y']

        # Create an empty tensor to hold scaled coordinates
        X_for_obst = torch.zeros_like(X_unit)

        X_for_obst[:, 0] = x_min + (x_max - x_min) * X_unit[:, 0]
        X_for_obst[:, 1] = y_min + (y_max - y_min) * X_unit[:, 1]
        
        # Create a mask for each polygon and combine them
        mask_obstacles = in_polygon(X_for_obst, poly_thin_wall)

        
        # x_in_obstacle = X_for_obst[mask_obstacles][0:100,:]
        
        # # Zero out (u, v, rho) inside obstacles
        # u_in_obstacle = torch.zeros_like(x_in_obstacle[:,0])
        # v_in_obstacle = torch.zeros_like(x_in_obstacle[:,0])
        # train_ro_in_obstacle= torch.zeros_like(x_in_obstacle[:,0])
        
        # Wall bounding box
        x_min_wall = W/2 - w_w/2
        x_max_wall = W/2 + w_w/2
        y_min_wall = 0.0
        y_max_wall =0.25
        # Choose grid resolution on the wall
        n_x = 3
        n_y = 7

        # Regular grid in the wall bounding box
        x_lin = torch.linspace(x_min_wall, x_max_wall, n_x)
        y_lin = torch.linspace(y_min_wall, y_max_wall, n_y)
        Xv, Yv = torch.meshgrid(x_lin, y_lin, indexing='xy')
        x_in_obstacle = torch.stack([Xv.flatten(), Yv.flatten()], dim=-1)


        # Zero out (u, v, rho) inside obstacles
        u_in_obstacle = torch.zeros_like(x_in_obstacle[:,0])
        v_in_obstacle = torch.zeros_like(x_in_obstacle[:,0])
        train_ro_in_obstacle= torch.zeros_like(x_in_obstacle[:,0])


        X_train = torch.cat([x_top, x_left, x_bottom_combined, x_right, x_corners,x_in_obstacle], dim=0)
        U_train = torch.cat([u_top, u_left, u_bottom_combined, u_right, u_corners,u_in_obstacle], dim=0)
        V_train = torch.cat([v_top, v_left, v_bottom_combined, v_right, v_corners,v_in_obstacle], dim=0)
        train_ro = torch.cat([ro_top, ro_left, ro_bottom_combined, ro_right, ro_corners,train_ro_in_obstacle], dim=0)
        
        import matplotlib.pyplot as plt
        plt.scatter(X_train[:, 0], X_train[:, 1], c=V_train, cmap='viridis', s=5)
        plt.show()
            

        

        
        ############# train_P ##############
        domain_b_p = {'x':[W-(L_a+L_t), W-(L_a)], 'y':[0, 0]}
        points_x_b = torch.linspace(domain_b_p['x'][0], domain_b_p['x'][1], N_train_b+2)[1:-1]
        x_bottom_b = torch.stack([points_x_b.squeeze(), torch.zeros(N_train_b)], dim=1)
        p_bottem=torch.zeros_like(x_bottom_b[:,0])
        
        
        domain_t_p = {'x':[L_a, L_a+L_t], 'y':[1, 1]}
        points_x_t = torch.linspace(domain_t_p['x'][0], domain_t_p['x'][1], N_train_b+2)[1:-1]
        x_top_b = torch.stack([points_x_t.squeeze(), H*torch.ones(N_train_b)], dim=1)
        p_top=torch.zeros_like(x_top_b[:,0])
        
        
        X_train_P = torch.cat([ x_bottom_b, x_top_b], dim=0)
        train_P =torch.cat([p_bottem, p_top], dim=0)
    ######################### COLOCATION POINTS #########################
    

    n = torch.floor(torch.sqrt(torch.div(torch.tensor(N_col_domain), torch.tensor(ratio), rounding_mode='trunc'))).int()
    nx, ny = int(n*ratio),int(n)   # Change these values to your desired grid size
    X_col_domain=gen_data(design_domain=domain, steps_x=nx,steps_y=ny)        
    
    ######################### DYNAMIC COLOCATION POINTS
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
    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nely = Nelx_list[i]
        Nelx = Nely_list[i]
    
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
        Nx = mask_col.shape[0]
        Ny = mask_col.shape[1]
        mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
        X_col =gen_data(design_domain=domain, steps_x=nx,steps_y=ny)# torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # define the traction BC:
        traction_indices = (X_col[:, 0] == xmin) & (X_col[:, 1] == ymax)
        traction_indices = traction_indices.requires_grad_(False).to(tkwargs['device'])

        # save in X_col_all[i]
        # Save all data in a dictionary
        
        X_col_data = {'traction_indices': traction_indices,'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'dx':dx,'dy':dy}


        # import matplotlib.pyplot as plt

        # # --- domain bounds ---
        # xmin, xmax = -0.06, 3.06
        # ymin, ymax = -0.0067, 1.0067

        # # --- number of random collocation points ---
        # N_col = 30000   # choose how many you want

        # # --- generate random uniform points inside the box ---
        # x_rand = np.random.uniform(xmin, xmax, N_col)
        # y_rand = np.random.uniform(ymin, ymax, N_col)

        # # stack into (N_col, 2)
        # X_col = np.vstack([x_rand, y_rand]).T

        # # convert to torch tensor
        # X_col = torch.tensor(X_col, dtype=torch.float32, requires_grad=True).to("cuda:0")

        # # ✅ convert back to numpy for plotting
        # X = X_col.detach().cpu().numpy()

        # # LaTeX rendering
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rcParams.update({'font.size': 22})

        # # plot
        # plt.figure(figsize=(8, 4))
        # plt.scatter(X[:, 0], X[:, 1],
        #             s=2,        # small dots
        #             c="black",
        #             marker=".",
        #             linewidths=0)

        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.gca().set_aspect("equal", adjustable="box")

        # # save high-quality PNG
        # plt.savefig("collocation_points.png", dpi=300, bbox_inches="tight")
        # plt.show()

                
        
        # Append the dictionary to the list
        X_col_all.append(X_col_data)
    
    
        X_train_U = X_train
        X_train_V = X_train
        X_train_ro = X_train
    return X_col_all, [X_train_U,X_train_V,X_train_P,X_train_ro],[U_train,V_train,train_P,train_ro]