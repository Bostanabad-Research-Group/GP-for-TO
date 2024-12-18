# Copyright © 2021 by Northwestern University.
# 
# LVGP-PyTorch is copyrighted by Northwestern University. It may be freely used 
# for educational and research purposes by  non-profit institutions and US government 
# agencies only. All other organizations may use LVGP-PyTorch for evaluation purposes 
# only, and any further uses will require prior written approval. This software may 
# not be sold or redistributed without prior written approval. Copies of the software 
# may be made by a user provided that copies are not sold or distributed, and provided 
# that copies are used under the same terms and conditions as agreed to in this 
# paragraph.
# 
# As research software, this code is provided on an "as is'' basis without warranty of 
# any kind, either expressed or implied. The downloading, or executing any part of this 
# software constitutes an implicit agreement to these terms. These terms and conditions 
# are subject to change at any time without prior notice.

import torch
import gpytorch
import math
from gpytorch.models import ExactGP
from gpytorch import settings as gptsettings
from gpytorch.priors import NormalPrior,LogNormalPrior
from gpytorch.constraints import GreaterThan,Positive
from gpytorch.distributions import MultivariateNormal
from .. import kernels
from ..priors import LogHalfHorseshoePrior,MollifiedUniformPrior
from ..utils.transforms import softplus,inv_softplus
from typing import List,Tuple,Union

import TO

from TO.likelihoods_noise.multifidelity import Multifidelity_likelihood

import botorch
from botorch.models.utils import gpt_posterior_settings
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel

from botorch import settings
from botorch.models.utils import fantasize as fantasize_flag, validate_input_scaling
from botorch.sampling.samplers import MCSampler
from torch import Tensor
from typing import Any, Dict, List, Optional, Union


class GPR(ExactGP, GPyTorchModel):
    """Standard GP regression module for numerical inputs

    :param train_x: The training inputs (size N x d). All input variables are expected
        to be numerical. For best performance, scale the variables to the unit hypercube.
    :type train_x: torch.Tensor
    :param train_y: The training targets (size N)
    :type train_y: torch.Tensor
    :param correlation_kernel: Either a `gpytorch.kernels.Kernel` instance or one of the 
        following strings - 'RBFKernel' (radial basis kernel), 'Matern52Kernel' (twice 
        differentiable Matern kernel), 'Matern32Kernel' (first order differentiable Matern
        kernel). If the former is specified, any hyperparameters to be estimated need to have 
        associated priors for multi-start optimization. If the latter is specified, then 
        the kernel uses a separate lengthscale for each input variable.
    :type correlation_kernel: Union[gpytorch.kernels.Kernel,str]
    :param noise: The (initial) noise variance.
    :type noise: float, optional
    :param fix_noise: Fixes the noise variance at the current level if `True` is specifed.
        Defaults to `False`
    :type fix_noise: bool, optional
    :param lb_noise: Lower bound on the noise variance. Setting a higher value results in
        more stable computations, when optimizing noise variance, but might reduce 
        prediction quality. Defaults to 1e-6
    :type lb_noise: float, optional
    """
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        correlation_kernel,
        noise_indices:List[int],
        noise:float=1e-4,
        fix_noise:bool=False,
        lb_noise:float=1e-12,
    ) -> None:
        # check inputs
        if not torch.is_tensor(train_x):
            raise RuntimeError("'train_x' must be a tensor")
        if not torch.is_tensor(train_y):
            raise RuntimeError("'train_y' must be a tensor")

        if train_x.shape[0] != train_y.shape[0]:
            raise RuntimeError("Inputs and output have different number of observations")
        
        # initializing likelihood
        noise_constraint=GreaterThan(lb_noise,transform=torch.exp,inv_transform=torch.log)
        
        if len(noise_indices) == 0:

            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
        else:

            likelihood = Multifidelity_likelihood(noise_constraint=noise_constraint, noise_indices=noise_indices, fidel_indices=train_x[:,-1])

        # Normilizing the response variable  ### orginlly was this 
        # y_mean,y_std = train_y.mean(),train_y.std()
        # train_y_sc = (train_y-y_mean)/y_std


        # standardizing the response variable
        ################################## ADED for biasis
        # y_h=train_y[torch.where(train_x[:,-1]==0)]
        # y_mean= y_h.mean()
        # y_std=y_h.max()-y_h.min()
        
        # Normalizing for PDEs
        y_mean = torch.tensor(0.0)
        y_std = torch.tensor(1.0)
        train_y_sc = (train_y-y_mean)/y_std

        # y_mean= train_y.mean()
        # y_std=train_y.max()-train_y.min()
        # train_y_sc = (train_y-y_mean)/y_std
        ####################################################
        # y_h=train_y[torch.where(train_x[:,-1]==0)].clone()
        # y_mean_h= y_h.mean()
        # y_std_h=y_h.max()-y_h.min()

        # y_l=train_y[torch.where(train_x[:,-1]==1)].clone()
        # y_mean_l= y_l.mean()
        # y_std_l=y_l.max()-y_l.min()

        # y_mean=train_y.clone()
        # y_mean[torch.where(train_x[:,-1]==0)]=y_mean_h
        # y_mean[torch.where(train_x[:,-1]==1)]=y_mean_l

        # y_std=train_y.clone()
        # y_std[torch.where(train_x[:,-1]==0)]=y_std_h
        # y_std[torch.where(train_x[:,-1]==1)]=y_std_l

        # train_y_sc = (train_y-y_mean)/y_std






        ################################################
        # y_mean= train_y.min()
        # y_std=train_y.max()-train_y.min()
        # train_y_sc = (train_y-y_mean)/y_std
        # y_mean= 0*train_y.min()
        # y_std=(train_y.max()-train_y.min())/(train_y.max()-train_y.min())
        # train_y_sc = (train_y-y_mean)/y_std
        # initializing ExactGP
        #super().__init__(train_x,train_y_sc,likelihood)
        ExactGP.__init__(self, train_x,train_y_sc, likelihood)
        
        # registering mean and std of the raw response
        self.register_buffer('y_mean',y_mean)
        self.register_buffer('y_std',y_std)
        self.register_buffer('y_scaled',train_y_sc)

        ##added for seprate normalizing
        # self.register_buffer('y_mean_h',y_mean_h)
        # self.register_buffer('y_std_h',y_std_h)
        # self.register_buffer('y_mean_l',y_mean_l)
        # self.register_buffer('y_std_l',y_std_l)
        # self.register_buffer('y_scaled',train_y_sc)


        # self.register_buffer('y_mean_h',y_mean)
        # self.register_buffer('y_std_h',y_std)
        # self.register_buffer('y_mean_l',y_mean)
        # self.register_buffer('y_std_l',y_std)
        # self.register_buffer('y_scaled',train_y_sc)

        self._num_outputs = 1

        # initializing and fixing noise
        if noise is not None:
            self.likelihood.initialize(noise=noise)
        
        self.likelihood.register_prior('noise_prior',LogHalfHorseshoePrior(0.01,lb_noise),'raw_noise')
        if fix_noise:
            self.likelihood.raw_noise.requires_grad_(False)
        
        # Modules
        # self.mean_module = gpytorch.means.ConstantMean(prior=NormalPrior(0.,1.))
        # self.mean_module_1 = gpytorch.means.ConstantMean(prior=NormalPrior(0.,1.))
        # self.mean_module_2 = gpytorch.means.ConstantMean(prior=NormalPrior(0.,1.))
        # self.mean_module_3 = gpytorch.means.ConstantMean(prior=NormalPrior(0.,1.))
        # self.mean_module_4 = gpytorch.means.ConstantMean(prior=NormalPrior(0.,1.))
        # self.mean_module_5 =gpytorch.means.LinearMean(input_size=train_x.shape[1], batch_shape=torch.Size([]), bias=True)
        # self.mean_module_5=LinearMean_with_prior(input_size=train_x.shape[1]-1, batch_shape=torch.Size([]), bias=True)



        if isinstance(correlation_kernel,str):
            try:
                correlation_kernel_class = getattr(kernels,correlation_kernel)
                correlation_kernel = correlation_kernel_class(
                    ard_num_dims = self.train_inputs[0].size(1),
                    lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
                )
                correlation_kernel.register_prior(
                    'lengthscale_prior',MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                )
            except:
                raise RuntimeError(
                    "%s not an allowed kernel" % correlation_kernel
                )
        elif not isinstance(correlation_kernel,gpytorch.kernels.Kernel):
            raise RuntimeError(
                "specified correlation kernel is not a `gpytorch.kernels.Kernel` instance"
            )

        self.covar_module = kernels.ScaleKernel(
            base_kernel = correlation_kernel,
            outputscale_constraint=Positive(transform=softplus,inv_transform=inv_softplus),
        )
        # register priors
        self.covar_module.register_prior(
            'outputscale_prior',LogNormalPrior(1e-6,1.),'outputscale'
        )
    
    def forward(self,x:torch.Tensor)->MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x,covar_x)
    
    def predict(
        self,x:torch.Tensor,return_std:bool=False,include_noise:bool=False
    )-> Union[torch.Tensor,Tuple[torch.Tensor]]:
        """Returns the predictive mean, and optionally the standard deviation at the given points

        :param x: The input variables at which the predictions are sought. 
        :type x: torch.Tensor
        :param return_std: Standard deviation is returned along the predictions  if `True`. 
            Defaults to `False`.
        :type return_std: bool, optional
        :param include_noise: Noise variance is included in the standard deviation if `True`. 
            Defaults to `False`.
        :type include_noise: bool
        """
        self.eval()
        with gptsettings.fast_computations(log_prob=False):
            # determine if batched or not
            ndim = self.train_targets.ndim
            if ndim == 1:
                output = self(x)
            else:
                # for batched GPs 
                num_samples = self.train_targets.shape[0]
                output = self(x.unsqueeze(0).repeat(num_samples,1,1))
            
            if return_std and include_noise:
                output = self.likelihood(output)
            j=0
            # out_mean=self.y_mean_l + self.y_std_l*output.mean.clone()
            # for i in output.mean:
            #     if x[j,-1]==0:
            #         out_mean[j] = self.y_mean_h + self.y_std_h*i
            #     elif x[j,-1]==1:
            #         out_mean[j] = self.y_mean_l + self.y_std_l*i
            #     j+=1

            out_mean = self.y_mean + self.y_std*output.mean
            # standard deviation may not always be needed
            if return_std:
                # out_std = output.variance.sqrt()*self.y_std_h
                out_std = output.variance.sqrt()*self.y_std

                return out_mean,out_std

            return out_mean

    
    def posterior(
        self,
        X,
        output_indices = None,
        observation_noise= True,
        posterior_transform= None,
        **kwargs,
    ):

        self.eval()
        with gpt_posterior_settings() and gptsettings.fast_computations(log_prob=False):
    
            if observation_noise:
                return GPyTorchPosterior(mvn = self.likelihood(self(X.double())))
            else:
                return GPyTorchPosterior(mvn = self(X.double()))
    
    def reset_parameters(self) -> None:
        """Reset parameters by sampling from prior
        """
        for _,module,prior,closure,setting_closure in self.named_priors():
            if not closure(module).requires_grad:
                continue
            setting_closure(module,prior.expand(closure(module).shape).sample().cuda())   ##############  .cuda() ???????????????????????


    def fantasize(
            self,
            X: Tensor,
            sampler: MCSampler,
            observation_noise: Union[bool, Tensor] = True,
            **kwargs: Any,
        ):
            r"""Construct a fantasy model.

            Constructs a fantasy model in the following fashion:
            (1) compute the model posterior at `X` (if `observation_noise=True`,
            this includes observation noise taken as the mean across the observation
            noise in the training data. If `observation_noise` is a Tensor, use
            it directly as the observation noise to add).
            (2) sample from this posterior (using `sampler`) to generate "fake"
            observations.
            (3) condition the model on the new fake observations.

            Args:
                X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                    the feature space, `n'` is the number of points per batch, and
                    `batch_shape` is the batch shape (must be compatible with the
                    batch shape of the model).
                sampler: The sampler used for sampling from the posterior at `X`.
                observation_noise: If True, include the mean across the observation
                    noise in the training data as observation noise in the posterior
                    from which the samples are drawn. If a Tensor, use it directly
                    as the specified measurement noise.

            Returns:
                The constructed fantasy model.
            """
            propagate_grads = kwargs.pop("propagate_grads", False)
            with fantasize_flag():
                with settings.propagate_grads(propagate_grads):
                    post_X = self.posterior(
                        X, observation_noise=observation_noise, **kwargs
                    )
                Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
                # Use the mean of the previous noise values (TODO: be smarter here).
                # noise should be batch_shape x q x m when X is batch_shape x q x d, and
                # Y_fantasized is num_fantasies x batch_shape x q x m.
                noise_shape = Y_fantasized.shape[1:]
                noise = self.likelihood.noise.mean().expand(noise_shape)
                return self.condition_on_observations(
                    X=self.transform_inputs(X), Y=Y_fantasized, noise=noise
                )

    '''
    def transform_inputs(
        self,
        X: Tensor,
        input_transform= None,
    ) -> Tensor:
        r"""Transform inputs.

        Args:
            X: A tensor of inputs
            input_transform: A Module that performs the input transformation.

        Returns:
            A tensor of transformed inputs
        """
        if input_transform is not None:
            input_transform.to(X)
            return input_transform(X)
        try:
            return self.input_transform(X)
        except AttributeError:
            return X

    '''

    '''

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> BatchedMultiOutputGPyTorchModel:
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `m` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, its is assumed that the missing batch dimensions are
                the same for all `Y`.

        Returns:
            A `BatchedMultiOutputGPyTorchModel` object of the same type with
            `n + n'` training examples, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.cat(
            >>>     [torch.sin(train_X[:, 0]), torch.cos(train_X[:, 1])], -1
            >>> )
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> new_X = torch.rand(5, 2)
            >>> new_Y = torch.cat([torch.sin(new_X[:, 0]), torch.cos(new_X[:, 1])], -1)
            >>> model = model.condition_on_observations(X=new_X, Y=new_Y)
        """
        noise = kwargs.get("noise")
        if hasattr(self, "outcome_transform"):
            # we need to apply transforms before shifting batch indices around
            Y, noise = self.outcome_transform(Y, noise)
        self._validate_tensor_args(X=X, Y=Y, Yvar=noise, strict=False)
        inputs = X

        inputs = X
        targets = Y

        if noise is not None:
            kwargs.update({"noise": noise})
        fantasy_model = self.condition_on_observations_super(X=inputs, Y=targets, **kwargs)
        fantasy_model._input_batch_shape = fantasy_model.train_targets.shape[
            : (-1 if self._num_outputs == 1 else -2)
        ]
        fantasy_model._aug_batch_shape = fantasy_model.train_targets.shape[:-1]
        return fantasy_model
    

    def condition_on_observations_super(self, X: Tensor, Y: Tensor, **kwargs: Any):
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, its is assumed that the missing batch dimensions are
                the same for all `Y`.

        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X[:, 0]) + torch.cos(train_X[:, 1])
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> new_X = torch.rand(5, 2)
            >>> new_Y = torch.sin(new_X[:, 0]) + torch.cos(new_X[:, 1])
            >>> model = model.condition_on_observations(X=new_X, Y=new_Y)
        """
        Yvar = kwargs.get("noise", None)
        if hasattr(self, "outcome_transform"):
            # pass the transformed data to get_fantasy_model below
            # (unless we've already trasnformed if BatchedMultiOutputGPyTorchModel)
            if not isinstance(self, BatchedMultiOutputGPyTorchModel):
                Y, Yvar = self.outcome_transform(Y, Yvar)
        # validate using strict=False, since we cannot tell if Y has an explicit
        # output dimension
        self._validate_tensor_args(X=X, Y=Y, Yvar=Yvar, strict=False)
        if Y.size(-1) == 1:
            Y = Y.squeeze(-1)
            if Yvar is not None:
                kwargs.update({"noise": Yvar.squeeze(-1)})
        # get_fantasy_model will properly copy any existing outcome transforms
        # (since it deepcopies the original model)
    
        return self.get_fantasy_model(inputs=X, targets=Y, **kwargs)


#######################################
import torch
from gpytorch.means import Mean
class LinearMean_with_prior(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        self.register_prior(name = 'weights_prior', prior=gpytorch.priors.NormalPrior(0.,5.), param_or_closure='weights')
        # lmgp.register_parameter('fci', self.fci.weight)
        # lmgp.register_prior(name = 'latent_prior_fci', prior=gpytorch.priors.NormalPrior(0.,3.), param_or_closure='fci')
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
            self.register_prior(name = 'bias_prior', prior=gpytorch.priors.NormalPrior(0.,5.), param_or_closure='bias')

        else:
            self.bias = None

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res


###################################################
    '''
