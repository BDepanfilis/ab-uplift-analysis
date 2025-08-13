#!/usr/bin/env python
"""
Config utilities: resolve project root and data path lazily (no import-time I/O).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def find_project_root(markers=("src",), extra_markers=("pyproject.toml", ".git")) -> Path:
    """
    Walk upward from cwd to find a directory that looks like the project root.
    """
    p = Path.cwd().resolve()
    for _ in range(10):
        has_marker = any((p / m).exists() for m in markers) or any((p / m).exists() for m in extra_markers)
        if has_marker:
            return p
        if p.parent == p:
            break
        p = p.parent
    # Fallback to directory containing this file
    return Path(__file__).resolve().parent.parent  # project root is one level up from src/

@dataclass
class DataConfig:
    env_var: str = "DATA_PATH"
    default_rel: str = "data/criteo-uplift-v2.1.parquet"
    root: Optional[Path] = None

    def data_path(self, *, must_exist: bool = False) -> Path:
        """
        Resolve dataset path from env var or default location under the project root.
        Does not access filesystem unless must_exist=True.
        """
        root = self.root or find_project_root()
        env_path = os.getenv(self.env_var, "").strip()
        candidate = Path(env_path) if env_path else (root / self.default_rel)
        candidate = candidate.expanduser().resolve()
        if must_exist and not candidate.exists():
            raise FileNotFoundError(
                f"Dataset not found at {candidate}. "
                f"Set {self.env_var}=/path/to/file or place file under {root/'data'}"
            )
        return candidate

# Public helpers
_cfg = DataConfig()

def get_data_path(must_exist: bool = False) -> Path:
    """Convenience wrapper."""
    return _cfg.data_path(must_exist=must_exist)

if __name__ == "__main__":
    p = get_data_path(must_exist=False)
    print(f"[config] Resolved data path -> {p}")
    if p.exists():
        st = p.stat()
        print(f"[config] Exists: size={st.st_size/1024/1024:.2f} MB, mtime={time.ctime(st.st_mtime)}")
    else:
        print("[config] File does not exist (set DATA_PATH env or place under data/).")
