import os
import re
import subprocess

# Regex to match filenames like: memory-domain100_rays512-20250702-104708.prof
filename_pattern = re.compile(r"memory-domain(\d+)_rays(\d+)-\d{8}-\d{6}\.prof")
# Regex to extract the total memory and unit from pprof output
memory_pattern = re.compile(r"Showing nodes accounting for ([\d.]+)([kM]?B), 100% of")

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
pprof_cmd = os.path.expanduser("~/go/bin/pprof")
target_bin = "/bin/ls"

def convert_to_mb(value: float, unit: str) -> float:
    """Convert memory value to megabytes."""
    if unit == "B":
        return value / 1_000_000
    elif unit == "kB":
        return value / 1_000
    elif unit == "MB":
        return value
    else:
        raise ValueError(f"Unknown memory unit: {unit}")

files = os.listdir(current_dir)
file_count = len(files)

import numpy as np

# [0, :] - represents domain, [1, :] - no. rays, [2, :] - total memory
results_matrix = np.zeros((3, file_count))

for i, filename in enumerate(files):
    match = filename_pattern.match(filename)
    if not match:
        continue

    domain = int(match.group(1))
    rays = int(match.group(2))
    filepath = os.path.join(current_dir, filename)

    try:
        result = subprocess.run(
            [pprof_cmd, "--top", target_bin, filepath],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout
        memory_match = memory_pattern.search(output)

        if memory_match:
            mem_value = float(memory_match.group(1))
            unit = memory_match.group(2)
            mem_mb = convert_to_mb(mem_value, unit)

            print(f"File: {filename} → Domain: {domain}, Rays: {rays}, Memory: {mem_mb:.2f} MB")

            results_matrix[0, i] = domain
            results_matrix[1, i] = rays
            results_matrix[2, i] = mem_mb
        else:
            print(f"File: {filename} → Domain: {domain}, Rays: {rays}, Memory: NOT FOUND")

            results_matrix[0, i] = domain
            results_matrix[1, i] = rays
            results_matrix[2, i] = 0
    except subprocess.CalledProcessError as e:
        print(f"Error running pprof on {filename}: {e.stderr.strip()}")

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

for i in range(file_count):
    ax.scatter(results_matrix[0, i], results_matrix[1, i], results_matrix[2, i], marker = 'o')

ax.set_xlabel('domain divisions')
ax.set_ylabel('no. of rays')
ax.set_zlabel('total memory usage')

plt.show()