from pathlib import Path

_module_root_dir = Path(__file__).parent.parent
import sys
sys.path.append(str(_module_root_dir))