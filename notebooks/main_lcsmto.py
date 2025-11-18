"""
Run topology optimization experiments with LC-SMTO models
and collect dissipated power statistics across seeds and volume fractions.

This script:
1. Generates collocation data.
2. Builds LC-SMTO surrogate models (U, V, P, density).
3. Runs optimization with find_TO_level_set_localized.

"""

from datetime import datetime
from pathlib import Path
import numpy as np
import torch

# --- Import project modules ---
from TO.models import LCSMTO      # Renamed here for clarity
from TO.utils_lcsmto import get_data_fluid, set_seed
from TO.optim import find_TO_level_set_localized


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
CONFIG = {
    "problem": 'rugby',           # ['doublepipe', 'pip_with_obstacle', 'rugby']
    "ratio": 1,
    "N_train_per_BC": 40,
    "N_col_domain": 10000,
    "volume_fractions": 0.9, # 1/3, # , 1/4, 1/5
    "seeds": 100, # 17, 18, 20, 100, 103
    "basis": "PGCAN",                  # ['PGCAN', 'neural_network']
    "localized_weight": True,
    "num_iter": 20010,
    "lr_default": 0.001,
    "cool_down_time": 100,              # seconds between runs
    "output_dir": "results_csv"
}

TDEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TDTYPE = torch.float
TARGS = {"dtype": TDTYPE, "device": TDEVICE}


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    problem = CONFIG["problem"]
    ratio = CONFIG["ratio"]
    N_train_per_BC = CONFIG["N_train_per_BC"]
    N_col_domain = CONFIG["N_col_domain"] * ratio
    basis = CONFIG["basis"]

    # timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    results = []

    vf= CONFIG["volume_fractions"]
    seed = CONFIG["seeds"]
    set_seed(seed)

    # ---------------- Domain setup ----------------
    ymin, ymax = 0.0, 1.0
    xmin, xmax = 0.0, 1.0 * ratio
    domain = {"x": [xmin, xmax], "y": [ymin, ymax]}
    Nely = 100
    Nelx = ratio * Nely
    MP = {
        "pad": 2,
        "Nelx": Nelx, "Nely": Nely,
        "Nelx_max": Nelx, "Nely_max": Nely,
        "Nelx_min": Nely, "Nely_min": Nelx,
        "num_CP": 50,
        "domain": domain,
    }

    # ---------------- Data generation ----------------
    X_col_all, X_train, Sol_train = get_data_fluid(
        problem=problem,
        N_col_domain=N_col_domain,
        N_train=N_train_per_BC,
        ratio=ratio,
        MP=MP,
        tkwargs=TARGS
    )
    X_col = X_col_all[0]["X_col"]

    # ---------------- Network config ----------------
    if basis == "PGCAN":
        n_features = 128
        n_neurons = n_features // 2
        n_layers = int(3 * ratio)
        NN_archi = [n_neurons] * n_layers
        NN_config = {
            "n_features": n_features,
            "n_cells": 3,
            "res": [9, int(9 * ratio)],
            "NN_arch": NN_archi,
            "save_folder": [],
            "activation": "tanh"
        }
    else:
        NN_archi = [64] * 5
        NN_config = {
            "n_features": [], "n_cells": [], "res": [],
            "NN_arch": NN_archi, "save_folder": [], "activation": []
        }

    # ---------------- Build LC-SMTO models ----------------
    output_specs = [
        ("u", X_train[0], Sol_train[0]),
        ("v", X_train[1], Sol_train[1]),
        ("p", torch.tensor([[0, 0]]), torch.tensor([0])),  # dummy for pressure
        ("ro", X_train[3], Sol_train[3] - 0.5)
    ]

    models = []
    for name, x, y in output_specs:
        model = LCSMTO(
            train_x=x.type(TDTYPE),
            collocation_x=X_col.type(TDTYPE).clone().requires_grad_(True).to(TDEVICE),
            train_y=y.type(TDTYPE),
            basis=basis,
            NN_config=NN_config,
            MP=MP,
            NN_layers_base=NN_archi,
            name_output=name,
            device=TDEVICE,
            dtype=TDTYPE
        ).to(**TARGS)
        models.append(model)

    # ---------------- Optimization ----------------
    dp = find_TO_level_set_localized(
        model_list=models,
        num_iter=CONFIG["num_iter"],
        lr_default=CONFIG["lr_default"],
        title=f"{timestamp}_seed{seed}_{problem}_vf_{vf:.2f}",
        diff_method="Numerical",
        X_col_all=X_col_all,
        localized_weight=CONFIG["localized_weight"],
        volume_fraction=vf,
        ratio=ratio,
    )

    dissipated_power = float(dp) if dp is not None else np.nan
