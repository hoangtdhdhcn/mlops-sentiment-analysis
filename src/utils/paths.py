import os

# Directory containing this file
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory
PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

# Output directories
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Create folders if they don't exist
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
