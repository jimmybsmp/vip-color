import sys
from pathlib import Path

# The synthetic-DNG helper lives beside the tests, not in the package.
sys.path.insert(0, str(Path(__file__).parent))
