import sys
from pathlib import Path

# Make the repo root importable so that `from src.xxx import ...` works
# when pytest is invoked from any working directory.
sys.path.insert(0, str(Path(__file__).parent))
