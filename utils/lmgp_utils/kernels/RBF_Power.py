#!/usr/bin/env python3

from gpytorch.settings import trace_mode
from gpytorch.kernels import Kernel

def postprocess_rbf(dist_mat):
    return dist_mat.div_(-1).exp_()


def default_postprocess_script(x):
    return x

def pairwise_diff_power(x1, x2, p):
    n = x1.shape[0]
    m = x1.shape[1]
    x1_expanded = x1.unsqueeze(1).expand(n, n, m)
    x2_expanded = x2.unsqueeze(0).expand(n, n, m)
    diff = (x1_expanded - x2_expanded).pow(p)
    return diff.norm(dim=-1)

class PowerExpKernel(Kernel):
    has_lengthscale = True
    has_power = True
    #power_prior: Optional[Prior] = None
    #power_constraint: Optional[Interval] = None

    def __init__(self, **kwargs):
        super(PowerExpKernel, self).__init__(**kwargs)
        # def _lengthscale_param(self, m):
        # return m.lengthscale

        # def _lengthscale_closure(self, m, v):
        # return m._set_lengthscale(v)
        # if self.has_power:

        # lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
        # self.register_parameter(
        #     name="power",
        #     parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        #     ) 
        # self.register_prior(name = "power_prior", prior=MollifiedUniformPrior(0.5,1), param_or_closure="power")
        #self.register_constraint("power", constraint=Positive())

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 0) # should be 1. Use 0 when only one numerical input
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            ten_power_omega_sqrt = self.lengthscale.sqrt()
            x1_ = x1.mul(ten_power_omega_sqrt)
            x2_ = x2.mul(ten_power_omega_sqrt)
            #print(self.power)
            return self.covar_dist(
                x1_, x2_,p=2.0/2.0, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
        # return RBFCovariance.apply(
        #     x1,
        #     x2,
        #     self.lengthscale,
        #     lambda x1, x2: self.covar_dist(
        #         x1, x2, p=self.power, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
        #     ),
        # )


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-1).exp_()

