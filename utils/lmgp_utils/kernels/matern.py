from gpytorch.kernels import MaternKernel
from functools import partialmethod

class Matern12Kernel(MaternKernel):
    __init__ = partialmethod(MaternKernel.__init__,nu=0.5)

class Matern32Kernel(MaternKernel):
    __init__ = partialmethod(MaternKernel.__init__,nu=1.5)

class Matern52Kernel(MaternKernel):
    __init__ = partialmethod(MaternKernel.__init__,nu=2.5)