"""Install the runtime packages that conda cannot provide for this interpreter.

Why this exists
---------------
CompleteBin's validated environment is deliberately hybrid: conda supplies the
interpreter, NumPy/SciPy/scikit-learn, pysam, hnswlib, tqdm and the five
bioinformatics programs, while PyTorch and the graph libraries were installed
with pip. Some of those pip versions have no conda build for Python 3.9
(``pytorch 2.8.0``, ``python-igraph 1.0.0``, ``leidenalg 0.12.0``,
``psutil 7.2.2``, ``faiss-cpu 1.8.0``), so a pure-conda solve is impossible.
The conda package therefore declares only what conda can satisfy and points
users here for the rest.

Both sides are pinned with ``==`` so the installed combination is exactly the
one this release was validated against.

Usage
-----
    completebin-pip-deps
    completebin-pip-deps --index-url https://download.pytorch.org/whl/cu128

Any extra arguments are forwarded to ``pip install``, which is how you select a
CPU-only or a specific CUDA build of PyTorch.
"""
import subprocess
import sys

# Keep in sync with completebin-conda-env.yml and requirements.txt.
PIP_REQUIREMENTS = [
    "torch==2.8.0",
    "python-igraph==1.0.0",
    "leidenalg==0.12.0",
    "psutil==7.2.2",
    "faiss-cpu==1.8.0",
]


def main(argv=None) -> int:
    """Install the pip-provided runtime packages; return pip's exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, "-m", "pip", "install", "--no-input"]
    command += PIP_REQUIREMENTS
    command += argv

    print("CompleteBin: the conda package covers part of the runtime stack;")
    print("these packages are installed with pip at the pinned versions:")
    for spec in PIP_REQUIREMENTS:
        print(f"  {spec}")
    print()
    print("$ " + " ".join(command), flush=True)

    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
