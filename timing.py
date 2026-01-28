import pandas as pd
from egret.parsers.matpower_parser import create_ModelData
import egret.models.dcopf as dcopf
from egret.models.acopf import solve_acopf, create_psv_acopf_model
import pyomo.environ as pyo
import egret.model_library.transmission.tx_utils as tx_utils
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
import time

class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
model = FFNN(42,72)  # Initialize the model
model.load_state_dict(torch.load('trained_models/ffnn_pglib_opf_case30_ieee_best.pt'))  # Load the saved model state
model.eval()

input_data = torch.randn(1, 42)  
tensor_from_list_float = torch.tensor(input_data, dtype=torch.float32).reshape(1,-1)

start_time = time.time()
with torch.no_grad():  
    predictions = model(tensor_from_list_float)
end_time = time.time()
NN_time = end_time - start_time
print('NN Time',NN_time)
files=['pglib-opf-master/pglib_opf_case14_ieee.m','pglib-opf-master/pglib_opf_case30_ieee.m']

for f in files:
    matpower_fname = f
    md = create_ModelData(matpower_fname)

    start_time = time.time()

    md_nominal, nominal_acopf, status = dcopf.solve_dcopf(md, "gurobi", solver_tee=True, return_model=True,
                                                        return_results=True)

    end_time = time.time()
    DC_time = end_time - start_time
    print('DC Time',DC_time)

    start_time = time.time()
    md_,n,s=solve_acopf(md, "ipopt", solver_tee=True, return_model=True, return_results=True)
    end_time = time.time()
    AC_time = end_time - start_time
    print('AC Time',AC_time)