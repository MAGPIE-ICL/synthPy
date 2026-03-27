import json

notebook_path = r"C:\Users\samma\programming\synthPy\evaluation\SLOTH\quad_test.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modified = False

# 1. Add 'import sys' to first code cell if missing
first_code_cell = next((c for c in nb.get('cells', []) if c.get('cell_type') == 'code'), None)
if first_code_cell:
    source = first_code_cell.get('source', [])
    if not any("import sys" in line for line in source):
        source.insert(0, "import sys\n")
        modified = True
        print("Added 'import sys' to first cell.")

# 2. Fix lwl parameter if still missing (just in case)
target_string = "solutions, duration = solver.solve(beam_definition.s0, domain, probing_extent, save_points_per_region = 128, return_raw_results = True, rtol = 1e-3, atol = 1e-6)"
replacement_string = "solutions, duration = solver.solve(beam_definition.s0, domain, probing_extent, save_points_per_region = 128, return_raw_results = True, rtol = 1e-3, atol = 1e-6, lwl = lwl)"

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if target_string in line and 'lwl =' not in line:
                source[i] = line.replace(target_string, replacement_string)
                modified = True
                print(f"Modified lwl in cell source line {i}")

if modified:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched successfully.")
else:
    print("No changes needed in notebook.")
