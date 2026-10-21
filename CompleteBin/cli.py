"""Command-line interface for the CompleteBin binning workflow."""

import argparse
import os
from pathlib import Path
import warnings

from CompleteBin.version import bin_v


def parse_bool(value: str) -> bool:
    """Parse explicit boolean values while producing an argparse-friendly error."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value (true/false), received {value!r}"
    )


def positive_integer(value: str) -> int:
    """Return a positive integer or raise a clear command-line error."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def non_negative_integer(value: str) -> int:
    """Return a non-negative integer (0 allowed) or raise a clear error.

    种子取 0 是合法值，所以这里不能复用 positive_integer。
    """
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser without importing the full pipeline."""
    parser = argparse.ArgumentParser(
        prog="completebin",
        description=(
            "Recover metagenome-assembled genomes (MAGs) from a contig FASTA "
            "and one or more coordinate-sorted BAM files."
        ),
        epilog=(
            "Examples:\n"
            "  # Run the complete workflow on one sample\n"
            "  completebin --contigs assembly.fasta --bams sample.sorted.bam "
            "--output bins --temp-dir work --device cuda:0\n\n"
            "  # Run steps on separate nodes using the same temporary directory\n"
            "  completebin -c assembly.fasta -b sample.sorted.bam -o bins -temp work "
            "--step-num 1 --device cpu\n"
            "  completebin -c assembly.fasta -b sample.sorted.bam -o bins -temp work "
            "--step-num 2 --device cuda:0\n"
            "  completebin -c assembly.fasta -b sample.sorted.bam -o bins -temp work "
            "--step-num 3 --device cpu\n\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"CompleteBin {bin_v}",
        help="Print the CompleteBin version and exit.",
    )

    required = parser.add_argument_group("required input and output")
    required.add_argument(
        "-c", "--contig_path", "--contigs", dest="contig_path", required=True,
        metavar="FASTA", help="Input contig FASTA file.",
    )
    required.add_argument(
        "-b", "--sorted_bams_paths", "--bams", dest="sorted_bams_paths",
        nargs="+", required=True, metavar="BAM",
        help="One or more coordinate-sorted BAM files. Use one BAM for a single "
             "sample or multiple BAMs for multi-sample coverage.",
    )
    required.add_argument(
        "-o", "--output_path", "--output", dest="output_path", required=True,
        metavar="DIR", help="Directory for final MAG FASTAs and MetaInfo.tsv.",
    )
    required.add_argument(
        "-temp", "--temp_file_path", "--temp-dir", dest="temp_file_path",
        required=True, metavar="DIR",
        help="Persistent working directory. Reuse it when running split steps.",
    )

    training = parser.add_argument_group("training options")
    training.add_argument(
        "--device", default="cpu", metavar="DEVICE",
        help="Training device, for example 'cpu' or 'cuda:0'. A GPU with about "
             "32 GB memory is recommended at the default batch size. (default: cpu)",
    )
    training.add_argument(
        "--dropout_prob", "--dropout", dest="dropout_prob", type=float, default=0.15,
        metavar="FLOAT", help="Dropout probability during training. (default: 0.15)",
    )
    training.add_argument(
        "--min_contig_length", "--min-contig-length", dest="min_contig_length",
        type=int, default=850, metavar="BP",
        help="Initial minimum contig length in bp; it must be at least 768. "
             "(default: 850)",
    )
    training.add_argument(
        "--batch_size", "--batch-size", dest="batch_size", type=positive_integer,
        default=544, metavar="N", help="Training batch size. (default: 544)",
    )
    training.add_argument(
        "--base_epoch", "--base-epoch", dest="base_epoch", type=positive_integer,
        default=35, metavar="N", help="Baseline number of training epochs. (default: 35)",
    )

    execution = parser.add_argument_group("workflow and clustering options")
    execution.add_argument(
        "-db", "--db_files_path", "--db", dest="db_files_path", metavar="DIR",
        help="CompleteBin database directory. If omitted, CompleteBin_DB is used.",
    )
    execution.add_argument(
        "-l_mode", "--leiden_iter_mode", "--leiden-mode", dest="leiden_iter_mode",
        choices=("accurate", "fast"), default="fast", metavar="MODE",
        help="Leiden optimisation: 'accurate' converges fully; 'fast' uses adaptive "
             "early stopping. (default: fast)",
    )
    execution.add_argument(
        "--leiden_seed", "--leiden-seed", dest="leiden_seed", type=non_negative_integer,
        default=42, metavar="SEED",
        help="Random seed for the leidenalg Optimiser. This is a separate parameter "
             "from the global training seed (which stays 2048); changing it does not "
             "retrain the model. (default: 42)",
    )
    execution.add_argument(
        "--cpu_workers", "--cpu-workers", dest="cpu_workers", type=positive_integer,
        metavar="N", help="Workers for preprocessing, marker-gene calling and clustering. "
        "Default: automatic.",
    )
    execution.add_argument(
        "--step_num", "--step-num", dest="step_num", type=int, choices=(1, 2, 3),
        metavar="STEP",
        help="Run only one resumable stage: 1=data preparation (CPU), 2=training, "
             "3=clustering (CPU). Omit to run all stages.",
    )
    execution.add_argument(
        "--auto_disable_pretrain", "--auto-disable-pretrain", dest="auto_disable_pretrain",
        type=parse_bool, default=True, metavar="BOOL",
        help="Automatically skip pretrained weights when input statistics indicate "
             "they are unsuitable. Accepts true/false. (default: true)",
    )
    execution.add_argument(
        "--ensemble_clustering", "--ensemble-clustering", dest="ensemble_clustering",
        type=parse_bool, default=True, metavar="BOOL",
        help="Which clustering scheme to use. Accepts true/false.\n"
             "  true (DEFAULT): the original scheme -- round-1 grid -> round-2 SCG "
             "re-clustering -> Galah consensus -> contig deduplication -> round-3 "
             "k-means.\n"
             "  false: the improved H1 scheme -- after the round-1 Leiden "
             "grid, pick ONE partition by SCG + embedding-compactness scoring, then "
             "coarse-filter (length>=20kb or n_contigs>=3) + fine-filter "
             "(emb_spread<=0.15 or length>=200kb or single-contig), then re-cluster the "
             "discarded contigs with k-means. Output is further trimmed by an "
             "embedding-compactness rule so that 3000+ MAGs drop to ~1500 while NC and "
             "MQ are unchanged. Skips round-2 SCG re-clustering, Galah and contig "
             "deduplication.\n"
             "See scripts/grid_reform_run_20261001/IMPROVEMENT.md. (default: true)",
    )
    execution.add_argument(
        "--pretrain-model-weight", dest="pretrain_model_weight_path", default=None,
        metavar="PTH",
        help="Path to the pretrained checkpoint to use. Defaults to "
             "<db>/CheckPoint/pretrain_weight_ep127.pth.",
    )
    execution.add_argument(
        "--low-gap-extra-batches", dest="low_gap_extra_batches", type=int, default=3,
        metavar="N",
        help="How many extra mini-batches of short contigs to pull in when lowering "
             "the training length threshold (default: 3). Larger = lower threshold, "
             "more (shorter) contigs enter training.",
    )
    execution.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and print the resolved configuration without running binning.",
    )
    return parser


def existing_file(parser: argparse.ArgumentParser, raw_path: str, label: str) -> str:
    """Expand a path and require it to be an existing regular file."""
    path = Path(raw_path).expanduser()
    if not path.is_file():
        parser.error(f"{label} does not exist or is not a file: {path}")
    return str(path.resolve())


def validate_and_prepare_paths(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Validate user-controlled inputs and create normal output directories."""
    args.contig_path = existing_file(parser, args.contig_path, "contig FASTA")
    args.sorted_bams_paths = [
        existing_file(parser, bam, "BAM file") for bam in args.sorted_bams_paths
    ]
    if args.min_contig_length < 768:
        parser.error("--min-contig-length must be at least 768 bp")
    if not 0.0 <= args.dropout_prob < 1.0:
        parser.error("--dropout must be in the range [0, 1)")

    database = args.db_files_path or os.environ.get("CompleteBin_DB")
    if not database:
        parser.error("provide --db DIR or set the CompleteBin_DB environment variable")
    database_path = Path(database).expanduser()
    if not database_path.is_dir():
        parser.error(f"CompleteBin database directory does not exist: {database_path}")
    args.db_files_path = str(database_path.resolve())

    for attribute, label in (("output_path", "output directory"),
                             ("temp_file_path", "temporary directory")):
        path = Path(getattr(args, attribute)).expanduser()
        if path.exists() and not path.is_dir():
            parser.error(f"{label} exists but is not a directory: {path}")
        if not args.dry_run:
            path.mkdir(parents=True, exist_ok=True)
        setattr(args, attribute, str(path.resolve()))


def print_dry_run(args: argparse.Namespace) -> None:
    """Print the exact validated configuration without starting the pipeline."""
    print("CompleteBin dry run: inputs validated; binning was not started.")
    print(f"  contigs: {args.contig_path}")
    print(f"  BAMs ({len(args.sorted_bams_paths)}):")
    for bam in args.sorted_bams_paths:
        print(f"    - {bam}")
    print(f"  output: {args.output_path}")
    print(f"  temporary directory: {args.temp_file_path}")
    print(f"  database: {args.db_files_path}")
    print(f"  device: {args.device}")
    print(f"  step: {args.step_num if args.step_num is not None else 'all'}")
    print(f"  leiden_iter_mode: {args.leiden_iter_mode}")
    print(f"  leiden_seed: {args.leiden_seed}")
    print(f"  ensemble_clustering: {args.ensemble_clustering}"
          f"{'' if args.ensemble_clustering else '  (improved H1 scheme, see IMPROVEMENT.md)'}")


# Runtime packages that CompleteBin imports but does not declare on PyPI.
# They are scientific or binary packages whose conda builds are coupled to the
# linear-algebra, CUDA and compiler stack of the environment; letting pip pull
# PyPI wheels for them would duplicate or upgrade what conda already manages and
# break the environment. They are therefore installed with conda, and this list
# turns the resulting ModuleNotFoundError into actionable instructions.
_RUNTIME_IMPORTS = [
    # (import name, conda package name, which side installs it)
    ("numpy", "numpy", "conda"),
    ("scipy", "scipy", "conda"),
    ("sklearn", "scikit-learn", "conda"),
    ("pysam", "pysam", "conda"),
    ("hnswlib", "hnswlib", "conda"),
    ("tqdm", "tqdm", "conda"),
    ("torch", "torch", "pip"),
    ("igraph", "python-igraph", "pip"),
    ("leidenalg", "leidenalg", "pip"),
    ("psutil", "psutil", "pip"),
]


def check_runtime_dependencies() -> None:
    """Exit with instructions when a runtime package is missing.

    CompleteBin's validated environment is hybrid: conda supplies the
    interpreter, NumPy/SciPy/scikit-learn, pysam, hnswlib, tqdm and the
    bioinformatics programs, while PyTorch and the graph libraries come from pip
    because conda has no Python 3.9 build for those versions. The message below
    only suggests each package on the side that can actually provide it.
    """
    import importlib.util

    missing = [
        (module, conda_name, side)
        for module, conda_name, side in _RUNTIME_IMPORTS
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return

    modules = ", ".join(module for module, _, _ in missing)
    conda_missing = [name for _, name, side in missing if side == "conda"]
    pip_missing = [name for _, name, side in missing if side == "pip"]

    lines = [
        f"CompleteBin is missing required runtime packages: {modules}",
        "",
        "The validated environment is hybrid: conda provides the interpreter,",
        "NumPy/SciPy/scikit-learn, pysam, hnswlib, tqdm and the bioinformatics",
        "programs; PyTorch and the graph libraries come from pip, because conda",
        "has no Python 3.9 build for those exact versions.",
        "",
    ]
    if conda_missing:
        lines += [
            "Install the missing conda packages:",
            "",
            "    conda install -c conda-forge -c bioconda "
            + " ".join(conda_missing),
            "",
        ]
    if pip_missing:
        lines += [
            "Install the missing pip packages (pinned to the validated versions):",
            "",
            "    completebin-pip-deps",
            "",
        ]
    lines += [
        "Or build the whole environment from the reference file:",
        "",
        "    conda env create -f completebin-conda-env.yml",
        "",
        "See the 'Dependencies' section of README.md.",
    ]
    raise SystemExit("\n".join(lines))


def main() -> None:
    """Parse CLI arguments, validate inputs, and start CompleteBin."""
    parser = build_parser()
    args = parser.parse_args()
    validate_and_prepare_paths(parser, args)
    if args.dry_run:
        print_dry_run(args)
        return

    check_runtime_dependencies()

    # Delay importing the full pipeline so --help, --version and --dry-run can
    # run without loading heavy runtime dependencies such as PyTorch.
    from CompleteBin.Binning_steps import binning_with_all_steps

    warnings.filterwarnings("ignore")
    binning_with_all_steps(
        contig_file_path=args.contig_path,
        sorted_bam_file_list=args.sorted_bams_paths,
        temp_file_folder_path=args.temp_file_path,
        bin_output_folder_path=args.output_path,
        db_folder_path=args.db_files_path,
        min_contig_length=args.min_contig_length,
        leiden_iter_mode=args.leiden_iter_mode,
        leiden_seed=args.leiden_seed,
        drop_p=args.dropout_prob,
        batch_size=args.batch_size,
        base_epoch=args.base_epoch,
        training_device=args.device,
        cpu_workers=args.cpu_workers,
        step_num=args.step_num,
        auto_disable_pretrain=args.auto_disable_pretrain,
        pretrain_model_weight_path=args.pretrain_model_weight_path,
        low_gap_extra_batches=args.low_gap_extra_batches,
        # CLI 默认 True = 走原始 ensemble 方案（round-1 → round-2 → Galah → 去重 → round-3）。
        # `binning_with_all_steps` 的函数默认值也是 True，两边一致。
        ensemble_clustering=args.ensemble_clustering,
    )


if __name__ == "__main__":
    main()
