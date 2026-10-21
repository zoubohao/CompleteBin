"""Packaging metadata for CompleteBin.

Version is read from ``CompleteBin/version.py`` so the runtime version string and
the PyPI version can never drift apart.
"""

import re
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent

README = (HERE / "README.md").read_text(encoding="utf-8")

# Read bin_v ("v1.2.3.6") straight from the source file instead of importing the
# package: under PEP 517 the build backend does not put the source root on
# sys.path, so `import CompleteBin` fails while preparing metadata.
_VERSION_FILE = (HERE / "CompleteBin" / "version.py").read_text(encoding="utf-8")
_match = re.search(r"""bin_v\s*=\s*["']([^"']+)["']""", _VERSION_FILE)
if _match is None:
    raise RuntimeError("cannot parse bin_v from CompleteBin/version.py")

# PEP 440 wants "1.2.3.6" on the index; the runtime string keeps the leading v.
VERSION = _match.group(1).lstrip("v")

# Deliberately empty.
#
# Every package CompleteBin imports (numpy, scipy, scikit-learn, torch, pysam,
# hnswlib, python-igraph, leidenalg, psutil, tqdm) is a compiled or
# BLAS/CUDA-coupled package whose conda build is tied to the rest of the
# environment. Declaring them here would make pip resolve, upgrade or duplicate
# them on top of an existing conda installation -- pip would, for example, try
# to move scipy 1.8.0 to a newer release, which drags numpy with it and breaks
# the environment. The supported installation path is the conda environment in
# completebin-conda-env.yml; `completebin` refuses to start with a clear message
# listing whatever is missing.
#
# `pip install completebin[full]` remains available for users who want pip to
# bring the whole scientific stack into a fresh, non-conda environment.
INSTALL_REQUIRES: list = []

EXTRAS_REQUIRE = {
    # Complete runtime set, pinned only at the lower bound the code needs.
    # Exact pins, matching completebin-conda-env.yml and the reference
    # environment. "==" rather than ">=" because a newer numpy / scipy /
    # scikit-learn can break the compiled extensions built against the older
    # ABI, or change numerics.
    "full": [
        "numpy==1.23.5",
        "scipy==1.8.0",
        "scikit-learn==0.23.2",
        "torch==2.8.0",
        "pysam==0.22.1",
        "hnswlib==0.8.0",
        "igraph==1.0.0",
        "leidenalg==0.12.0",
        "psutil==7.2.2",
        "tqdm==4.67.1",
        "faiss-cpu==1.8.0",
    ],
    # Exact KNN backend on its own, for a bare `pip install completebin`.
    "faiss": ["faiss-cpu==1.8.0"],
}

setup(
    name="completebin",
    version=VERSION,
    author="Bohao Zou",
    author_email="csbhzou@comp.hkbu.edu.hk",
    description="Metagenomic binning of contigs with dynamic contrastive learning.",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=("LICENSE",),
    python_requires=">=3.9",
    # Restrict to the package itself: the repository also contains top-level
    # analysis helpers that must not be shipped.
    packages=find_packages(include=["CompleteBin", "CompleteBin.*"]),
    include_package_data=False,
    package_data={"CompleteBin": []},
    exclude_package_data={"CompleteBin": ["__pycache__/*", "*.pyc", "*.pyo"]},
    zip_safe=False,
    entry_points={"console_scripts": ["completebin=CompleteBin.cli:main"]},
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
