import torch
import time
from datetime import datetime
import pandas as pd
from TO.models import GPPLUS
from TO.utils import set_seed, get_data_fluid
from TO.optim import find_TO

tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Timestamp for this run
NOW = datetime.now().strftime("%B%d_%H-%M")

vf = .95
seed = 17
problem ='rugby'# "doublepipe"
ratio = 1.0#  3.0
N_train_per_BC = 25
N_col_domain = 10000 ## 30000
mean_function = "neural_network"
NN_archi = [64, 64, 64, 64, 64]
output_names = ["u", "v", "p", "ro"]

# Store results
records = []

set_seed(seed)

# Data generation
X_col, X_train, Sol_train = get_data_fluid(
    problem=problem, N_col_domain=N_col_domain,
    N_train=N_train_per_BC, ratio=ratio
)

# Define GPPLUS models
models = [
    GPPLUS(
        train_x=X_train[i].type(tkwargs["dtype"]),
        train_y=Sol_train[i].type(tkwargs["dtype"]),
        collocation_x=X_col.type(tkwargs["dtype"]).clone().requires_grad_(True).to(tkwargs["device"]),
        basis=mean_function,
        NN_layers_base=NN_archi,
        name_output=name,
        device=tkwargs["device"],
        dtype=tkwargs["dtype"]
    ).to(**tkwargs)
    for i, name in enumerate(output_names)
]

# Optimization
dissipated_power = find_TO(
    model_list=models,
    num_iter=50001,
    lr_default=0.001,
    ratio=ratio,
    title=f"seed{seed}_{problem}_{NOW}_vf_{vf:.2f}",
    volume_fraction=vf,
    diff_method="Numerical"
)

