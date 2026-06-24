"""
train.py
--------
Backward-compatibility shim.
Delegates to train_regression.py (the renamed module).
"""
from train_regression import train  # noqa: F401

if __name__ == "__main__":
    train()
