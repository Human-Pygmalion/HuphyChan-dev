"""Print + save the scenario numbers:  python3 tests/run_all.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scenarios  # noqa: E402

d = scenarios.all_numbers()
out = os.path.join(os.path.dirname(HERE), "results", "test_numbers.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(d, open(out, "w"), indent=1)
print(json.dumps(d, indent=1))
