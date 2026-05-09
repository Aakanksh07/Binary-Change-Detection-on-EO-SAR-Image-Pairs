"""
train.py — Train the change detection model.
Usage: python train.py --config config.yaml
"""
import argparse, yaml
print("To train, run the notebook galaxeye_complete.ipynb cell by cell.")
print("All hyperparameters are in config.yaml.")
print("Training command: open galaxeye_complete.ipynb and run Cell 13.")

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)
print(f"\nLoaded config: {args.config}")
for k, v in config.items():
    print(f"  {k}: {v}")
