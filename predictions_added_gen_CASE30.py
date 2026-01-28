import os
import sys
import math
import random
import pandas as pd
from egret.parsers.matpower_parser import create_ModelData
import egret.models.dcopf as dcopf
from egret.models.acopf import solve_acopf, create_psv_acopf_model
import pyomo.environ as pyo
import egret.model_library.transmission.tx_utils as tx_utils
import numpy as np
from pyDOE import lhs
from IPython import embed
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
import random
from IPython import embed

np.random.seed(2)

case_name = 'pglib_opf_case30_ieee'

train_ds_full = OPFDataset(
    root='data',
    case_name=case_name,
    split='train'
)
val_ds_full = OPFDataset(
    root='data',
    case_name=case_name,
    split='val'
)

test_ds_full = OPFDataset(
    root='data',
    case_name=case_name,
    split='test'
)

i = len(val_ds_full) // 100
indices = random.sample(range(len(val_ds_full)), i)
val_ds = torch.utils.data.Subset(val_ds_full, indices)

j = len(test_ds_full) // 100
indices = random.sample(range(len(test_ds_full)), j)
test_ds = torch.utils.data.Subset(test_ds_full, indices)

k = len(train_ds_full) // 100
indices = random.sample(range(len(train_ds_full)), k)

train_ds = torch.utils.data.Subset(train_ds_full, indices)

training_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

def flatten_inputs(data):
    xs = []
    for node_type in ['load']:
        x = data.x_dict[node_type]
        xs.append(x.flatten())
    return torch.cat(xs, dim=0)

def flatten_targets(data):
    ys = []
    for node_type in ['generator', 'bus']:
        y = data.y_dict[node_type]
        ys.append(y.flatten())
    return torch.cat(ys, dim=0)

sample = train_ds[0]
input_dim = flatten_inputs(sample).numel()
output_dim = flatten_targets(sample).numel()

print(f"Input dim: {input_dim}")
print(f"Output dim: {output_dim}")

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

files = [
    'bus/pglib_opf_case30_ieee_modified_bus_1.m',
    'bus/pglib_opf_case30_ieee_modified_bus_2.m',
    'bus/pglib_opf_case30_ieee_modified_bus_5.m',
    'bus/pglib_opf_case30_ieee_modified_bus_11.m',
    'bus/pglib_opf_case30_ieee_modified_bus_13.m'
]


all_pg_MSE = []
all_qg_MSE = []
all_va_MSE = []
all_vm_MSE = []

model = FFNN(input_dim, output_dim)  
model.load_state_dict(torch.load('trained_models/ffnn_pglib_opf_case30_ieee_best.pt')) 
model.eval()  

for f in files:
    matpower_fname = f
    md = create_ModelData(matpower_fname)

    md_, n, s = solve_acopf(md, "ipopt", solver_tee=False, return_model=True, return_results=True)

    inputs = []
    outputs = []
    skipper = 0
    for l in md.data['elements']['load'].keys():
        inputs.append(md.data['elements']['load'][l]['p_load'] / 100)
        inputs.append(md.data['elements']['load'][l]['q_load'] / 100)
    for g in md.data['elements']['generator'].keys():
        if skipper != 0:
            outputs.append(pyo.value(n.pg[g]))
            outputs.append(pyo.value(n.qg[g]))
        skipper += 1
    for b in md.data['elements']['bus'].keys():
        outputs.append(pyo.value(n.va[b]))
        outputs.append(pyo.value(n.vm[b]))

    print(f"Inputs length for {f}: {len(inputs)}")
    print(f"Outputs length for {f}: {len(outputs)}")

    tensor_from_list_float = torch.tensor(inputs, dtype=torch.float32).reshape(1, -1)

    with torch.no_grad():  
        predictions = model(tensor_from_list_float)

    print("Predictions:", predictions)
    print("True:", outputs)

    # Calculate MSE for each output
    mse_per_output = []
    for i in range(len(outputs)):
        mse = F.mse_loss(predictions[0][i], torch.tensor(outputs[i], dtype=torch.float32))
        mse_per_output.append(mse)

    # Store MSE values in respective lists
    pg_MSE = []
    qg_MSE = []
    va_MSE = []
    vm_MSE = []

    print("\nMSE per output for file:", f)
    for i, mse in enumerate(mse_per_output):
        if i < 12 and i % 2 == 0:
            pg_MSE.append(mse.item())
        elif i < 12 and i % 2 != 0:
            qg_MSE.append(mse.item())
        elif i >= 12 and i % 2 == 0:
            va_MSE.append(mse.item())
        elif i >= 12 and i % 2 != 0:
            vm_MSE.append(mse.item())

    all_pg_MSE.append(pg_MSE)
    all_qg_MSE.append(qg_MSE)
    all_va_MSE.append(va_MSE)
    all_vm_MSE.append(vm_MSE)

print("\nAll MSEs for each file:")
for idx, f in enumerate(files):
    print(f"File: {f}")
    print(f"  PG MSE: {np.mean(all_pg_MSE[idx])}")
    print(f"  QG MSE: {np.mean(all_qg_MSE[idx])}")
    print(f"  VA MSE: {np.mean(all_va_MSE[idx])}")
    print(f"  VM MSE: {np.mean(all_vm_MSE[idx])}")


print(f"OVERALL MEAN")
print(f"  PG MSE: {np.mean(all_pg_MSE)}")
print(f"  QG MSE: {np.mean(all_qg_MSE)}")
print(f"  VA MSE: {np.mean(all_va_MSE)}")
print(f"  VM MSE: {np.mean(all_vm_MSE)}")

#SAVE ARRAYS
all_pg_MSE_array=np.array(all_pg_MSE)
all_qg_MSE_array=np.array(all_qg_MSE)
all_va_MSE_array=np.array(all_va_MSE)
all_vm_MSE_array=np.array(all_vm_MSE)

np.save('all_pg_MSE_array_30_Gplus1.npy', all_pg_MSE_array)
np.save('all_qg_MSE_array_30_Gplus1.npy', all_qg_MSE_array)
np.save('all_va_MSE_array_30_Gplus1.npy', all_va_MSE_array)
np.save('all_vm_MSE_array_30_Gplus1.npy', all_vm_MSE_array)