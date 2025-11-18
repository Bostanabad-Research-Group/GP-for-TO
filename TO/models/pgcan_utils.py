import torch
from torch import nn
from utils.utils_general import get_tkwargs

tkwargs = get_tkwargs()
dtype = tkwargs['dtype']
device = tkwargs['device']

class Model_PGCAN(nn.Module):
    
    def __init__(self, network, encoder = None  , data_params = None) -> None:
        super().__init__()
        self.network = network
        self.encoder = encoder
        self.data_params = data_params

    def forward(self,X) :
        x = X[: , :1]
        y = X[: , 1:]

        out = self.encoder(x, y)

        out = self.network(input = X , features = out)

        return out

class Encoder(nn.Module):

    def __init__(self, n_features = 128, res = [9], n_cells = 2, domain_size = [0, 1, 0, 1], mode = 'cosine') -> None:
        super().__init__()
        self.mode = mode
        self.n_features = n_features 
        self.res = res
        self.n_cells = n_cells
        a = [torch.rand(size=(self.n_cells, self.n_features, self.res[0], self.res[1]),
             dtype=dtype).to(device).data.uniform_(-1e-5,1e-5)]
        
        self.F_active = nn.ParameterList(a)
        self.domain_size = domain_size

        self.conv_layer = nn.Conv2d(in_channels=self.n_features , out_channels=self.n_features,
                                               groups = self.n_features ,kernel_size=3, padding=1, bias=False).to(device)   ### groups=1 ### 
        
        #self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2).to(device)

    def forward(self, x, y):
        features  = []
        x_min, x_max, y_min, y_max = self.domain_size
        x = (x - x_min)/(x_max - x_min)
        y = (y - y_min)/(y_max - y_min)
        x = x*2-1
        y = y*2-1

        x = torch.cat([x, y], dim=-1).unsqueeze(0).unsqueeze(0)
        x = x.repeat([self.n_cells,1,1,1])

        for idx , alpha in enumerate(self.F_active):

            alpha = self.conv_layer(alpha)
            self.beta = alpha
            alpha = nn.Tanh()(alpha)
           

            F = grid_sample_2d(alpha, x, step=self.mode, offset=True) # this is for the fixed angle
            #print(f"pixel shape {F.shape}")
            dim = alpha.shape[1]
            features.append(F.sum(0).view(dim,-1).t())

        
        F = torch.cat(tuple(features) , 1)
        #F_ = torch.cat([F  , boundary_f] , 0)
        #print(F_.shape)
        return F#_
    
def grid_sample_2d(input, grid, step='cosine', offset=True):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW).
        grid: A torch.Tensor of dimension (N, H, W, 2).
    Return:
        torch.Tensor: The bilinearly interpolated values (N, H, W, 2).
    '''
    N, C, IH, IW = input.shape
    _, H, W, _ = grid.shape

    if step=='bilinear':
        step_f = lambda x: x
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError

    ''' (iy,ix) will be the indices of the input
            1. normalize coordinates 0 to 1 (from -1 to 1)
            2. scaling to input size
            3. adding offset to make non-zero derivative interpolation '''
    ix = grid[..., 0]
    iy = grid[..., 1]
    if offset:
        offset = torch.linspace(0,1-(1/N),N).reshape(N,1,1).to(device)
        iy = ((iy+1)/2)*(IH-2) + offset
        ix = ((ix+1)/2)*(IW-2) + offset

    else:
        iy = ((iy+1)/2)*(IH-1)
        ix = ((ix+1)/2)*(IW-1)
    
    # compute corner indices
    with torch.no_grad():
        ix_left = torch.floor(ix)
        ix_right = ix_left + 1
        iy_top = torch.floor(iy)
        iy_bottom = iy_top + 1

    # compute weights
    dx_right = step_f(ix_right-ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom-iy)
    dy_top = 1 - dy_bottom

    nw = dx_right*dy_bottom
    ne = dx_left*dy_bottom
    sw = dx_right*dy_top
    se = dx_left*dy_top

    # sanity checking
    with torch.no_grad():
        torch.clamp(ix_left, 0, IW-1, out=ix_left)
        torch.clamp(ix_right, 0, IW-1, out=ix_right)
        torch.clamp(iy_top, 0, IH-1, out=iy_top)
        torch.clamp(iy_bottom, 0, IH-1, out=iy_bottom)

    # look up values
    input = input.view(N, C, IH*IW)
    nw_val = torch.gather(input, 2, (iy_top * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    ne_val = torch.gather(input, 2, (iy_top * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))
    sw_val = torch.gather(input, 2, (iy_bottom * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    se_val = torch.gather(input, 2, (iy_bottom * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))

    # 2d_cosine/bilinear interpolation
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

class NetworkM4_fused(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 1, layers = [40 , 40 , 40 , 40 ], activation = 'tanh') -> None:
        super(NetworkM4_fused, self).__init__()
        activation_list = {'tanh':nn.Tanh(), 'relu':nn.ReLU(), 'leaky_relu':nn.LeakyReLU(negative_slope=0.01), 'swish':nn.SiLU(), 'sigmoid':nn.Sigmoid(),'softplus':nn.Softplus(),'sin':torch.sin}
        activation = activation_list[activation]


        self.H1 = nn.Linear(input_dim, layers[0]).to(device)
        self.last= nn.Linear(layers[0], output_dim).to(device)

        l = nn.ModuleList()
        for i in range(len(layers)):
            l.append(nn.Linear(layers[i], layers[i]))
            l.append(activation )
        #l.append(nn.Linear(layers[-1], output_dim, bias=True))
        self.layers = nn.Sequential(*l).to(device)

    def forward(self, input , features):

        #features (N,256)
        F = int(features.shape[1]/2)

        U = features[:,:F]

        V = features[:,F:]
        
        H = nn.Tanh()(self.H1(input))

        #out = self.layers[0](input)
        for layer in self.layers:
            Z = layer(H)
            H = (1-Z)*U + Z*V

        out = self.last(H)
        # out = (nn.Tanh()(out) + 1) / 2   ### Amin: added nn.Tanh()
        return out