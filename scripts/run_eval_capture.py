"""Capture batch evaluation output to a file."""
import sys
import os
import io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Suppress all logging to stderr  
import logging
logging.disable(logging.CRITICAL)

# Capture stdout
old_stdout = sys.stdout
sys.stdout = io.StringIO()

from eval.batch_evaluate import run_evaluation
from config import settings

input_dir = str(settings.BASE_DIR / "data" / "washing_machine" / "Washing Machines")
run_evaluation(input_dir)

output = sys.stdout.getvalue()
sys.stdout = old_stdout

# Write to file
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_output.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("```\n")
    f.write(output)
    f.write("```\n")

# Print summary to real stdout
lines = output.split("\n")
for line in lines:
    stripped = line.strip()
    if any(k in stripped for k in ["Accuracy", "Correct", "Incorrect", "Skipped", "time", "FAILURE"]):
        print(stripped)
