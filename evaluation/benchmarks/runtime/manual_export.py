import pandas as pd
import re
import argparse

def mem_to_bytes(mem_str):
    match = re.match(r"([0-9.]+)\s*(MB|GB)", mem_str.strip())
    if not match:
        return None
    value, unit = match.groups()
    value = float(value)
    if unit == "MB":
        return int(value * 1024 * 1024)
    elif unit == "GB":
        return int(value * 1024 * 1024 * 1024)
    return None

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--drop", type=int)
args = parser.parse_args()

# Read fixed-width formatted input file
df = pd.read_fwf("input.txt")

# Convert memory columns from "X Y" (e.g. "1.5 GB") to bytes (int)
for col in ["domainSize", "raySize", "totalMemory"]:
    df[col] = df[col].apply(mem_to_bytes)

# Drop a row if requested
if args.drop is not None:
    df = df.drop(index=args.drop)

# Convert runtime columns to numeric (should already be numeric)
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
df["legacyRuntime"] = pd.to_numeric(df["legacyRuntime"], errors="coerce")

# Save to CSV
df.to_csv("adjusted_data.csv", index=False)