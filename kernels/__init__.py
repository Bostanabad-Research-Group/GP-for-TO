from .matern import Matern32Kernel,Matern52Kernel,Matern12Kernel
from gpytorch.kernels import ScaleKernel,RBFKernel,MultitaskKernel
from .Rough_RBF import Rough_RBF
#### Amin Added  To Test wighted_RBF ####
from .wighted_RBF import wighted_RBF
from .wighted_RBF_Z import wighted_RBF_Z
from .RBF_Power import PowerExpKernel
