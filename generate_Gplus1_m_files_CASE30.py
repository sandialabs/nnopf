def add_generator_to_m_file(input_file_path, output_file_path, new_gen, new_gencost):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    gen_start_index = None
    gencost_start_index = None

    for i, line in enumerate(lines):
        if 'mpc.gen =' in line:
            gen_start_index = i + 1 
        elif 'mpc.gencost =' in line:
            gencost_start_index = i + 2  

    for i, line in enumerate(lines):
        if line.startswith('function mpc ='):
            lines[i] = f"function mpc = {output_file_path[4:-2]}\n"  

    if gen_start_index is not None:

        lines.insert(gen_start_index, f"\t{new_gen};\n")

    if gencost_start_index is not None:

        lines.insert(gencost_start_index, f"\t{new_gencost};\n")

    with open(output_file_path, 'w') as file:
        file.writelines(lines)



input_file_path = 'pglib-opf-master/pglib_opf_case30_ieee.m'

for bus in [1, 2, 5, 11, 13]:
    new_generator = f"{bus}\t 100.0\t 0.0\t 50.0\t -10.0\t 1.0\t 100.0\t 1\t 150.0\t 10.0"  #FIX THESE TO BE ACCURATE
    new_gencost = "2\t 0.0\t 0.0\t 3\t 0.000000\t 18.421528\t 0.000000"  
    
    output_file_path = f'bus/pglib_opf_case30_ieee_modified_bus_{bus}.m'  

    add_generator_to_m_file(input_file_path, output_file_path, new_generator, new_gencost)