import sys
from pathlib import Path

# Add the src directory to the sys.path using pathlib
project_root = Path(__file__).parent.parent  # Navigate two levels up
src_path = project_root / "src"
if src_path not in sys.path:
    sys.path.append(str(src_path))
