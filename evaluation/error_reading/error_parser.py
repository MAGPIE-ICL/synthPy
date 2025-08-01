import json
import re
import os

import argparse
import string

# call script from directory of relevant .py output and pass arge of filename
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Name of file to parse.")

args = parser.parse_args()

if args.file is not None:
    filename = args.file
else:
    filename = "error"

# Load JSON data from a file
with open(filename, "r") as f:
    data = json.load(f)

# Extract and decode the fields
name = data.get("name", "")
message = data.get("message", "").encode('utf-8').decode('unicode_escape')
stack = data.get("stack", "").encode('utf-8').decode('unicode_escape')

# Optional: remove ANSI color codes (like \x1b[0;31m)
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
message_clean = ansi_escape.sub('', message)
stack_clean = ansi_escape.sub('', stack)

# Combine and write to file
output = f"{name}: {message_clean}\n\n{'='*80}\n\n{stack_clean}"

#output into directory error_log, if no dir, create one

# Write to readable file
with open("parsed_stack_trace.txt", "w") as f:
    f.write(output)

print("Formatted stack trace written to 'parsed_stack_trace.txt'.")