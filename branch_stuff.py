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


matpower_fname = "pglib-opf-master/pglib_opf_case24_ieee_rts.m"
# matpower_fname = 'pglib-opf-master/pglib_opf_case30_ieee.m'
md = create_ModelData(matpower_fname)
md.data["system"]["load_mismatch_cost"] = 100000

md_, n, s = dcopf.solve_dcopf(
    md,
    "gurobi",
    solver_tee=False,
    return_model=True,
    return_results=True,
    include_feasibility_slack=True,
)
n.pg.pprint()
n.pf.pprint()
# n.pg['1'].fix(0.4)
# n.pg['2'].fix(0.5868421052631578)
# n.pg['3'].fix(3.0131578947368425)
# n.pg['5'].fix(6.0)
# n.pg['4'].fix(0.0)
n.pf["5"].fix(0.0)
n.pf["19"].fix(0.0)
pyo.SolverFactory("gurobi").solve(n)
n.obj.pprint()
n.p_load_shed.pprint()
print("----------Nominal Case Objective---------")
nominal_objective = pyo.value(n.obj)
print(nominal_objective)
# n.pg.pprint()
# n.p_load_shed.pprint()
# n.p_over_generation.pprint()
objective_values = [("Nominal Case", nominal_objective)]


for _, branch_name in enumerate(md.data["elements"]["branch"]):
    print(branch_name)
    md = create_ModelData(matpower_fname)
    md.data["system"]["load_mismatch_cost"] = 1000000000000
    # if md.data['elements']['branch'][f"{branch_name}"]['branch_type'] == 'transformer':
    #    md.data['elements']['branch'][f"{branch_name}"]['rating_long_term'] *= 1
    # elif md.data['elements']['branch'][f"{branch_name}"]['branch_type'] == 'line':
    # md.data['elements']['branch'][f"{branch_name}"]['in_service'] = True
    try:
        md_, n, s = dcopf.solve_dcopf(
            md,
            "gurobi",
            solver_tee=False,
            return_model=True,
            return_results=True,
            include_feasibility_slack=True,
        )
        n.pf[f"{branch_name}"].fix(0.0)
        # n.pf[f"{branch_name}"].pprint()
        pyo.SolverFactory("gurobi").solve(n)
        n1_objective = pyo.value(n.obj)
        # n.obj.pprint()
        # n.pg.pprint()
        # n.p_load_shed.pprint()
        # n.p_over_generation.pprint()
        print("N-1 Objective")
        print(n1_objective)
        objective_values.append((branch_name, n1_objective))

    except:
        print("The problem is infeasible. Continuing with the rest of the code.")
        objective_values.append((branch_name, None))

df = pd.DataFrame(objective_values, columns=["Branch Name", "Objective Value"])

print(df)


embed()
