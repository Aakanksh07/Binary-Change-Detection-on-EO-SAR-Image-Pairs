"""
eval.py — Evaluate model on a test split.
Usage: python eval.py --data_path /path/to/test --weights /path/to/best_model.pth
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, help='Path to test split folder')
parser.add_argument('--weights',   required=True, help='Path to model checkpoint .pth')
args = parser.parse_args()

print(f"Data path : {args.data_path}")
print(f"Weights   : {args.weights}")
print("To run full evaluation, open galaxeye_complete.ipynb and run Cell 15.")
print("Pass your data_path and weights path in the CONFIG cell before running.")
