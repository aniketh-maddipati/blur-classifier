from __future__ import annotations

import runpy
import sys
from pathlib import Path


STUB_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUB_DIR.parents[1]

sys.path.insert(0, str(STUB_DIR))
sys.path.insert(1, str(REPO_ROOT))
sys.argv = ["cull_review_app.py", *sys.argv[1:]]

runpy.run_path(str(REPO_ROOT / "cull_review_app.py"), run_name="__main__")
