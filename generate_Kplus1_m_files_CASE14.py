import random
import itertools

def add_branch_to_m_file(input_file_path, output_file_path, new_branch, function_name):

    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    branch_start_index = None

    for i, line in enumerate(lines):
        if 'mpc.branch =' in line:
            branch_start_index = i + 1  

    for i, line in enumerate(lines):
        if line.startswith('function mpc ='):
            lines[i] = f"function mpc = {function_name}\n"  

    if branch_start_index is not None:

        lines.insert(branch_start_index, f"\t{new_branch};\n")


    with open(output_file_path, 'w') as file:
        file.writelines(lines)

input_file_path = 'pglib-opf-master/pglib_opf_case14_ieee.m'


buses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

bus_pairs = list(itertools.combinations(buses, 2))

sampled_pairs = random.sample(bus_pairs, k=min(5, len(bus_pairs))) 

for fbus, tbus in sampled_pairs:
    r = 0.0192
    x = 0.0575
    b = 0.0528
    rateA = 138
    rateB = 138
    rateC = 138
    ratio = 0.0
    angle = 0.0
    status = 1
    angmin = -30.0
    angmax = 30.0

    new_branch = f"{fbus}\t {tbus}\t {r}\t {x}\t {b}\t {rateA}\t {rateB}\t {rateC}\t {ratio}\t {angle}\t {status}\t {angmin}\t {angmax}"

    output_file_path = f'CASE14_branch/pglib_opf_case14_ieee_modified_branch_{fbus}_{tbus}.m'
    function_name = f"pglib_opf_case14_ieee_modified_branch_{fbus}_{tbus}"

    add_branch_to_m_file(input_file_path, output_file_path, new_branch, function_name)

print("Branches added successfully to the new files.")