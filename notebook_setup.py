"""Runtime setup helpers shared by the project notebooks.

The notebook imports this module instead of carrying separate Colab and local
setup blocks.  Keeping it at the repository root makes it importable before
the ``src`` directory has been added to ``sys.path``.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path


DEFAULT_DRIVE_OUTPUT_DIR = (
    "/content/drive/MyDrive/Uni-Masters/Group Project/outputs/exp_histories"
)


def setup_notebook(in_colab: bool, root: Path) -> None:
    """Configure paths and dependencies after the notebook locates the project."""
    if in_colab:
        _setup_colab_dependencies()
    _prepend_sys_path(root / "src")
    os.chdir(root)
    print(f"Running in {'Colab' if in_colab else 'VS Code'} from {root}")


def enable_colab_downloads(in_colab: bool) -> None:
    """Apply Colab's browser-like user agent required by the city download."""
    if not in_colab:
        return

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
    urllib.request.install_opener(opener)


def mount_drive_output(in_colab: bool, output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR) -> Path:
    """Mount Google Drive in Colab and return the selected output directory."""
    if not in_colab:
        raise RuntimeError("Google Drive output is available only when running in Colab.")

    from google.colab import drive

    drive.mount("/content/drive")
    return Path(output_dir)


def _setup_colab_dependencies() -> None:
    """Install Colab-only dependencies and expose SUMO's tool modules."""
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "eclipse-sumo"], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )

    import sumo

    os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    _prepend_sys_path(Path(sumo.SUMO_HOME) / "tools")


def _prepend_sys_path(path: Path) -> None:
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)
