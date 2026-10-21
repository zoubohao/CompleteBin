# CompleteBin

CompleteBin recovers metagenome-assembled genomes (MAGs) from a contig FASTA
file and one or more coordinate-sorted BAM files. It combines coverage and
k-mer features with a pretrained sequence model, dynamic contrastive learning,
and Leiden clustering, then polishes, dereplicates, and exports the final bins.

The method is described in *Dynamic Contrastive Learning with Pretrained Deep
Language Model Enhances Metagenome Binning for Contigs*.

- **Version:** 1.2.3.6
- **Python:** 3.9 or newer
- **Platform:** Linux
- **License:** MIT

---

## Installation

CompleteBin is published on **anaconda.org as `csbhzou/completebin`**. There is no
need to install from source or from PyPI.

### 1. Create an environment and install the package

```bash
conda create -n completebin \
  -c csbhzou -c conda-forge -c bioconda \
  "completebin=1.2.3.6=py_3"

conda activate completebin      # required: the launcher resolves "python" from PATH
completebin --version           # CompleteBin v1.2.3.6
```

To add it to an environment you already use, replace the first command with:

```bash
conda install -c csbhzou -c conda-forge -c bioconda "completebin=1.2.3.6=py_3"
```

`"completebin=1.2.3.6=py_3"` pins both the version and the build, which is what
makes the install reproducible. Drop the `=py_3` suffix to let conda take the
newest build of 1.2.3.6 instead.

The three channels are required, not optional:

| Channel | Provides |
|---|---|
| `csbhzou` | `completebin` itself |
| `conda-forge` | python 3.9.25, numpy, scipy, scikit-learn, pysam, hnswlib, tqdm |
| `bioconda` | prodigal, hmmer, fraggenescan, galah, skani |

### 2. Install the pip-side packages once

Five packages have no Python 3.9 conda build, so they are installed by a
companion command instead of by conda:

```bash
completebin-pip-deps
```

| Package | Version | Why not conda |
|---|---|---|
| `torch` | `2.8.0` | conda-forge builds pytorch >= 2.8 only for Python >= 3.10 |
| `python-igraph` | `1.0.0` | no cp39 conda build |
| `leidenalg` | `0.12.0` | no cp39 conda build |
| `psutil` | `7.2.2` | no cp39 conda build |
| `faiss-cpu` | `1.8.0` | no cp39 conda build |

For a specific PyTorch build, forward pip arguments:

```bash
completebin-pip-deps --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
completebin-pip-deps --index-url https://download.pytorch.org/whl/cpu     # CPU only
```

`completebin --help`, `--version` and `--dry-run` work **before** this step; the
dependency check only runs when binning actually starts, and it prints the exact
command to run when something is missing.

### 3. Point CompleteBin at the database

The database is **not** part of the conda package (it is 322 MB and changes far
less often than the code). Download it once from Google Drive:
<https://drive.google.com/file/d/1kNFsv-DVWcdjOu3WF2mFZBDAWwucsTJE/view?usp=sharing>

```bash
# gdown handles Google Drive's "file too large to scan" confirmation page
pip install gdown
gdown 1kNFsv-DVWcdjOu3WF2mFZBDAWwucsTJE -O completebin-db-v1.2.3.6.tar.gz

sha256sum completebin-db-v1.2.3.6.tar.gz   # 193a0c123b60fccbb48fad133e7994b7f01f85af735b191e2ccb7802794256a4
tar -xzf completebin-db-v1.2.3.6.tar.gz
export CompleteBin_DB=$PWD/CompleteBin-DB-v1.2.3.6
```

Without `gdown`, the direct URL works as well:

```bash
wget -O completebin-db-v1.2.3.6.tar.gz \
  "https://drive.google.com/uc?export=download&confirm=t&id=1kNFsv-DVWcdjOu3WF2mFZBDAWwucsTJE"
```

Always check the checksum: a Drive download that fetched the confirmation page
instead of the file produces a file of a few kilobytes, and the `sha256sum` step
above catches it.

Add that `export` line to `~/.bashrc` to make it permanent, or pass `--db` per
run. See the *Database* section for the directory layout.

### Verify the installation

```bash
completebin --version
completebin --help | grep "true (DEFAULT)"        # --ensemble-clustering defaults to true
completebin -c contigs.fasta -b sample.sorted.bam \
  -o out -temp tmp --db "$CompleteBin_DB" --dry-run
```

## Database

CompleteBin needs a data directory containing marker sets, HMM models, and the
pretrained weights. **The database is distributed neither through conda nor
through PyPI**: the pretrained checkpoint alone is 329 MB, and bundling it would
add that to every environment while the database changes far less often than the
code. It is published as a separate tarball on Google Drive.

### Download and unpack

Hosted on Google Drive:
<https://drive.google.com/file/d/1kNFsv-DVWcdjOu3WF2mFZBDAWwucsTJE/view?usp=sharing>

```bash
# completebin-db-v1.2.3.6.tar.gz, 307 MiB (322 MB)
pip install gdown
gdown 1kNFsv-DVWcdjOu3WF2mFZBDAWwucsTJE -O completebin-db-v1.2.3.6.tar.gz

# or, without gdown:
# wget -O completebin-db-v1.2.3.6.tar.gz "https://drive.google.com/uc?export=download&confirm=t&id=1kNFsv-DVWcdjOu3WF2mFZBDAWwucsTJE"

# Verify the archive (SHA-256 shown below)
sha256sum completebin-db-v1.2.3.6.tar.gz

tar -xzf completebin-db-v1.2.3.6.tar.gz
```

`SHA-256: 193a0c123b60fccbb48fad133e7994b7f01f85af735b191e2ccb7802794256a4`

### Expected layout

```text
CompleteBin-DB-v1.2.3.6/
├── CheckPoint/
│   └── pretrain_weight_ep127.pth              # 329 MB, epoch-127 pretrained weights
├── data/
│   ├── markersets.ms                          # marker gene sets per domain
│   └── pfam_file.dat                          # Pfam alignments (Stockholm)
└── HMM/
    ├── 40_marker.hmm                          # 40 single-copy marker HMMs
    ├── bac_arc_hmm_model_extended.hmm         # bacterial/archaeal HMM models
    └── get_40marker.pl                        # marker summary script
```

### Point CompleteBin at the database

Either export the path once:

```bash
export CompleteBin_DB=/path/to/CompleteBin-DB-v1.2.3.6
```

or pass it per command with `--db`. The CLI validates that the directory exists
before starting.

To confirm the pretrained weights are actually loaded, check the start of the
log for this line. If it reports `reinitialised 49` instead of
`loaded 49 tensors`, the weights did not load:

```text
--> pretrain_model: loaded 49 tensors, reinitialised 0 (shape/name mismatch): []
```

---

## Quick start

### Full workflow

```bash
completebin \
  --contigs assembly.contigs.fasta \
  --bams sample.sorted.bam \
  --output results/completebin \
  --temp-dir work/completebin \
  --device cuda:0 \
  --db "$CompleteBin_DB"
```

For multi-sample coverage, pass every BAM after `--bams`:

```bash
completebin \
  --contigs coassembly.contigs.fasta \
  --bams sample1.sorted.bam sample2.sorted.bam sample3.sorted.bam \
  --output results/completebin \
  --temp-dir work/completebin \
  --device cuda:0
```

Validate inputs and print the resolved configuration without binning:

```bash
completebin \
  --contigs assembly.contigs.fasta \
  --bams sample.sorted.bam \
  --output results/completebin \
  --temp-dir work/completebin \
  --dry-run
```

### Split CPU/GPU workflow

The temporary directory is resumable, so preparation and clustering can run on
CPU nodes while training runs on a GPU node. Use the **same** temporary
directory, inputs, output directory, and database for all three commands.

```bash
# Stage 1 - data preparation and coverage features (CPU)
completebin -c assembly.contigs.fasta -b sample.sorted.bam \
  -o results/completebin -temp work/completebin \
  --step-num 1 --device cpu --cpu-workers 64

# Stage 2 - model training and embedding generation (GPU)
completebin -c assembly.contigs.fasta -b sample.sorted.bam \
  -o results/completebin -temp work/completebin \
  --step-num 2 --device cuda:0

# Stage 3 - clustering, polishing, dereplication, export (CPU)
completebin -c assembly.contigs.fasta -b sample.sorted.bam \
  -o results/completebin -temp work/completebin \
  --step-num 3 --device cpu --cpu-workers 64
```

Cached artefacts from a completed stage are reused when the same temporary
directory is given again.

---

## Command-line options

`completebin --help` is authoritative. Underscore and hyphen spellings are
interchangeable.

| Option | Default | Description |
|---|---:|---|
| `-c`, `--contigs` | required | Input contig FASTA. Alias: `--contig_path`. |
| `-b`, `--bams` | required | One or more coordinate-sorted BAM files. Alias: `--sorted_bams_paths`. |
| `-o`, `--output` | required | Directory for final MAGs and `MetaInfo.tsv`. Alias: `--output_path`. |
| `-temp`, `--temp-dir` | required | Persistent working directory. Alias: `--temp_file_path`. |
| `-db`, `--db` | `CompleteBin_DB` | Database directory. |
| `--device` | `cpu` | Training device, e.g. `cpu` or `cuda:0`. |
| `--cpu-workers` | automatic | Workers for preprocessing, marker-gene calling, and clustering. Default is one third of the available cores plus one. |
| `--batch-size` | `544` | Training batch size. Reduce when GPU memory is short. |
| `--base-epoch` | `35` | Baseline number of training epochs. |
| `--dropout` | `0.15` | Dropout probability during training. |
| `--min-contig-length` | `850` | Initial minimum contig length in bp. Values below 768 are rejected; the effective threshold may be adjusted from input statistics. |
| `--leiden-mode` | `fast` | `fast` uses adaptive early stopping; `accurate` converges fully. |
| `--leiden-seed` | `42` | Random seed for the Leiden optimiser. Independent of the training seed, which is fixed at 2048. |
| `--ensemble-clustering` | `true` | `true` (default) runs the original chain: round-1 grid, round-2 SCG re-clustering, Galah consensus, contig deduplication, round-3 k-means. `false` switches to the improved single-partition scheme (one partition chosen from the round-1 grid, coarse + fine filtering, k-means on the discarded contigs). |
| `--step-num` | all stages | Run only stage `1`, `2`, or `3`. |
| `--auto-disable-pretrain` | `true` | Skip the pretrained encoder automatically when input statistics suggest it would hurt. |
| `--pretrain-model-weight` | `<db>/CheckPoint/pretrain_weight_ep127.pth` | Override the pretrained checkpoint. |
| `--low-gap-extra-batches` | `3` | Extra mini-batches of short contigs pulled in when the training length threshold is lowered. |
| `--dry-run` | off | Validate inputs and print the resolved configuration. |
| `--version` | - | Print the version and exit. |

---

## Outputs

```text
results/completebin/
├── CompleteBin_1.fasta
├── CompleteBin_2.fasta
├── ...
└── MetaInfo.tsv
```

- `CompleteBin_<N>.fasta` - one FASTA per recovered MAG.
- `MetaInfo.tsv` - one row per MAG with five tab-separated columns: MAG name,
  the label `SCG_EVAL(Comp,Cont,Quality)`, completeness, contamination, and the
  quality category. Completeness and contamination are estimated from
  single-copy marker genes.

Quality categories follow the usual MIMAG convention: high quality means
completeness >= 90% with contamination <= 5%, medium quality means completeness
>= 50% with contamination <= 10%.

The temporary directory keeps coverage profiles, training features, embeddings,
marker-gene calls, and clustering intermediates. Keep it while resuming a split
run or investigating an intermediate result, and delete it once the run is
finished.

---

## Python API

The CLI is a thin wrapper around `binning_with_all_steps`:

```python
from CompleteBin.Binning_steps import binning_with_all_steps

binning_with_all_steps(
    contig_file_path="assembly.contigs.fasta",
    sorted_bam_file_list=["sample.sorted.bam"],
    temp_file_folder_path="work/completebin",
    bin_output_folder_path="results/completebin",
    db_folder_path="/path/to/CompleteBin-DB-v1.2.3.6",
    training_device="cuda:0",
)
```

The docstring covers the full parameter set, including the split-workflow
`step_num`, training hyperparameters, and clustering switches.

---

## Environment variables

| Variable | Meaning |
|---|---|
| `CompleteBin_DB` | Default database directory, used when `--db` is omitted. |

---

## What's new in 1.2.3.6

- Coverage profiling, contrastive training, clustering, and dereplication are
  unchanged in interface but re-tuned internally; the pretrained checkpoint is
  the epoch-127 model stored as `CheckPoint/pretrain_weight_ep127.pth`.
- `--ensemble-clustering` defaults to **`true`**, i.e. the original chain
  (round-1 Leiden grid, round-2 SCG re-clustering, Galah consensus, contig
  deduplication, round-3 k-means). Pass `--ensemble-clustering false` to opt
  into the improved single-partition scheme instead: one partition is chosen
  from the round-1 grid by marker-gene and embedding-compactness scoring,
  coarse- and fine-filtered, and the discarded contigs are re-clustered with
  k-means, which trims the MAG count without losing high- and medium-quality
  genomes.
- `--leiden-seed` is a separate, exposed parameter; the training seed stays
  fixed at 2048 for reproducibility.
- **Distribution.** `conda install -c csbhzou -c conda-forge -c bioconda completebin`
  is now the primary install path; the same code is also on PyPI
  (`pip install completebin`, no hard dependencies — see below).
- Packaging: the wheel declares no hard PyPI dependencies on purpose. The
  scientific stack must come from conda (`completebin-conda-env.yml`), because
  pip would upgrade or duplicate packages such as `scipy`, `scikit-learn`,
  `numpy` and `torch` and break the environment. `pip install completebin[full]`
  covers fresh non-conda environments, and `[faiss]` adds the optional exact-KNN
  backend. A startup check now names any missing package and the conda command
  that installs it, instead of failing with a bare `ModuleNotFoundError`.

---

## Citation

If CompleteBin contributes to your work, please cite:

> Dynamic Contrastive Learning with Pretrained Deep Language Model Enhances
> Metagenome Binning for Contigs.

---

## License

MIT. See `LICENSE`.
