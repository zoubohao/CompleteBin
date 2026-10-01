# CompleteBin

**Paper --> CompleteBin: A transformer-based framework unlocks microbial dark matter through improved short contig binning**

CompleteBin is a metagenomic binning workflow that recovers
metagenome-assembled genomes (MAGs) from a contig FASTA and one or more
coordinate-sorted BAM files. It combines coverage and k-mer features with a
pretrained sequence model, dynamic contrastive learning, Leiden clustering.

The main reference is *CompleteBin: A transformer-based framework unlocks microbial dark matter through improved short contig binning*.

## Requirements

- Linux
- Python 3.9.25
- One or more coordinate-sorted BAM files corresponding to the input contigs
- CompleteBin database files and pretrained weights
- Sufficient RAM for the coverage profiles

Training can run on CPU, but a CUDA GPU is recommended for practical runtime.
The default training batch size (`544`) may require roughly 16 GB of GPU memory;
reduce `--batch-size` if GPU memory is limited.

## Installation

Install CompleteBin in the following order.

1. Create and activate the Conda environment. This installs the supported
   Python version and the required external tools (including Prodigal, HMMER,
   FragGeneScan, Galah, and Skani).

```bash
conda env create --name completebin --file completebin-conda-env.yml
conda activate completebin
```

2. Install a PyTorch build appropriate for your operating system, GPU, and
   CUDA driver. For example, the local CUDA 12.8 installation uses PyTorch
   `2.8.0+cu128`:

```bash
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

3. Install the released CompleteBin package from PyPI:

```bash
pip install completebin==1.2.3.4
```

Confirm the installation with:

```bash
completebin --version
completebin --help
```

When working directly from a source checkout, use the package form rather
than running `CompleteBin/cli.py` as a standalone script:

```bash
python -m CompleteBin.cli --help
```

## Database setup

CompleteBin requires the marker sets, HMM resources, and pretrained model
weights in a CompleteBin database directory. Download `CompleteBin-DB.zip`
from the [database release](https://drive.google.com/file/d/1wJMoA-JikRbxC-UjQ_L3P77AUFH41R5I/view?usp=sharing), unpack it, then use either
of the following approaches.

Set the database path once in your shell configuration:

```bash
export CompleteBin_DB=/path/to/CompleteBin-DB
```

Or pass it for an individual command:

```bash
completebin ... --db /path/to/CompleteBin-DB
```

The CLI validates that the selected database directory exists before starting.

## Quick start

### Complete workflow

Run the full workflow with a contig FASTA, one or more coordinate-sorted BAMs,
an output directory, and a persistent temporary directory:

```bash
completebin \
  --contigs assembly.contigs.fasta \
  --bams sample.sorted.bam \
  --output results/completebin \
  --temp-dir work/completebin \
  --device cuda:0 \
  --db /path/to/CompleteBin-DB
```

For multiple coverage samples, provide every BAM after `--bams`:

```bash
completebin \
  --contigs coassembly.contigs.fasta \
  --bams sample1.sorted.bam sample2.sorted.bam sample3.sorted.bam \
  --output results/completebin \
  --temp-dir work/completebin \
  --device cuda:0
```

Before launching a long job, validate paths and inspect the resolved settings:

```bash
completebin \
  --contigs assembly.contigs.fasta \
  --bams sample.sorted.bam \
  --output results/completebin \
  --temp-dir work/completebin \
  --dry-run
```

`--dry-run` does not create directories or start binning. CompleteBin checks
that the FASTA, BAM files, and database directory exist. A BAM index is not a
CLI requirement.

### Split CPU/GPU workflow

The work directory is resumable. This allows data preparation and clustering
to run on CPU nodes while model training runs on a GPU node. Use the *same*
`--temp-dir`, input FASTA, BAM list, output directory, and database for all
three commands.

```bash
# 1. Data preparation and coverage features (CPU node)
completebin -c assembly.contigs.fasta -b sample.sorted.bam \
  -o results/completebin -temp work/completebin \
  --step-num 1 --device cpu --cpu-workers 64

# 2. Model training and embedding generation (GPU node)
completebin -c assembly.contigs.fasta -b sample.sorted.bam \
  -o results/completebin -temp work/completebin \
  --step-num 2 --device cuda:0

# 3. Clustering, polish, dereplication, and final export (CPU node)
completebin -c assembly.contigs.fasta -b sample.sorted.bam \
  -o results/completebin -temp work/completebin \
  --step-num 3 --device cpu --cpu-workers 64
```

Do not delete or change the temporary directory between stages. Stages 1 and 2
reuse cached files when they already exist.

## Command-line options

Run `completebin --help` for the authoritative option list. The commonly used
options are shown below; underscore and hyphen spellings are equivalent where
both are listed.

| Option | Default | Description |
|---|---:|---|
| `-c`, `--contigs` | required | Input contig FASTA. Legacy name: `--contig_path`. |
| `-b`, `--bams` | required | One or more coordinate-sorted BAM files. Legacy name: `--sorted_bams_paths`. |
| `-o`, `--output` | required | Final MAG output directory. Legacy name: `--output_path`. |
| `-temp`, `--temp-dir` | required | Persistent work directory. Legacy name: `--temp_file_path`. |
| `--db` | `CompleteBin_DB` | CompleteBin database directory. |
| `--device` | `cpu` | Training device, e.g. `cpu` or `cuda:0`. |
| `--cpu-workers` | automatic | Workers for preprocessing, marker-gene calling, and clustering. |
| `--batch-size` | `544` | Training batch size. Reduce this if GPU memory is insufficient. |
| `--base-epoch` | `35` | Baseline training epochs. |
| `--dropout` | `0.15` | Training dropout probability. |
| `--min-contig-length` | `850` | Initial contig-length threshold in bp; values below 768 are rejected. |
| `--leiden-mode` | `accurate` | `accurate` converges fully; `fast` uses adaptive early stopping. |
| `--step-num` | all stages | Run only stage `1`, `2`, or `3` of the split workflow. |
| `--auto-disable-pretrain` | `true` | Automatically skip pretrained weights when input statistics indicate they are unsuitable. |
| `--dry-run` | off | Validate inputs and show the resolved configuration without binning. |

Improved Phase-2 polish is enabled by the pipeline. When its internal evidence
checks are insufficient for a particular bin, the cleaner uses its safety
fallback for that bin.

## Python API

The CLI calls `binning_with_all_steps`. It can also be used directly:

```python
from CompleteBin.Binning_steps import binning_with_all_steps

binning_with_all_steps(
    contig_file_path="assembly.contigs.fasta",
    sorted_bam_file_list=["sample.sorted.bam"],
    temp_file_folder_path="work/completebin",
    bin_output_folder_path="results/completebin",
    db_folder_path="/path/to/CompleteBin-DB",
    training_device="cuda:0",
)
```

See the docstring of `binning_with_all_steps` for the full Python API,
including training and split-workflow parameters.

=======
## Files in the output directory
- #### The binned MAGs.
- #### MetaInfo.tsv
This file contains the following columns: 

1. MAG name 
2. MAG quality （Only using Bac and Arc SCGs to do the evaluation）
3. completeness of MAG 
4. contamination of MAG 
5. MAG quality 

## Minimum System Requirements for Running CompleteBin
- System: Linux
- CPU: No restriction.
- RAM: > 180 GB
- GPU: The GPU memory must be equal to or greater than 12 GB.

## Repo Contents
- [CompleteBin](./CompleteBin): The main code (Python) of CompleteBin.

## Outputs

The final output directory contains:

- `CompleteBin_*.fasta`: recovered MAG FASTA files.
- `MetaInfo.tsv`: one row per output MAG, including its name, estimated
  completeness, estimated contamination, and quality category.

The temporary directory stores reusable coverage, feature, embedding,
marker-gene, and clustering artefacts. Keep it while resuming a split run or
investigating intermediate results.

## Repository layout

- `CompleteBin/`: Python package and binning implementation.
- `CompleteBin/cli.py`: command-line interface.
