# DeeperBin
**Paper --> Dynamic Contrastive Learning with Pretrained Deep Language Model Enhances Metagenome Binning for Contigs**

DeeperBin is a binner to cluter the contigs with dynamic contrastive learning and pretrained deep language model.

## Installation (Have Verified):
#### 1. FIRST STEP (Create Conda Environment for DeeperBin)
Create DeeperBin's conda environment by using this command:
```
conda env create -n DeeperBin -f deeperbin-conda-env.yml
```

**And**

download PyTorch v2.1.0 -cu*** (or higher version) from **[http://pytorch.org/](http://pytorch.org/)** if you want to use GPUs (We highly recommend to use GPUs). For example:
```
conda activate DeeperBin
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 2. SECOND STEP (Install Codes of DeeperBin)
After preparing the environment, the code of DeeperBin can be installed via pip simply. 
```
conda activate DeeperBin
pip install DeeperBin==1.0.7
```
This installation will run for around 10 minutes.

## Download Pretrained Weight and Other Files for Running
Download the pretrained weight and other files (**DeeperBin-DB.zip**) for running DeeperBin from this **[LINK](https://drive.google.com/file/d/1MLpt68I7MVZPKvwkjCOgDi0yPLfWRz7E/view?usp=sharing)**.


#### 1. Set Environmental Variable
- Unzip the downloaded file (**DeeperBin-DB.zip**) and set an **environmental variable** called "DeeperBin_DB" by adding the following line to the last line of .bashrc file (The path of the file: ~/.bashrc):
```
export DeeperBin_DB=/path/of/this/DeeperBin-DB/
```
For example: 'export DeeperBin_DB=/home/csbhzou/software/DeeperBin-DB/'.

- Save the .bashrc file, and then execute:
```
source .bashrc
```

#### 2. Using the '-db' or '--db_files_path' flag in CLI

- **You can set the '-db' flag in CLI to the path of the 'DeeperBin-DB' folder if you do not want to set the environmental variable.**


## Running DeeperBin


**1.  You can run DeeperBin with 'clean' mode through the following command:**

```
deeperbin -h
DeeperBin version: *** v1.0.7 ***
usage: deeperbin [-h] -c CONTIG_PATH -b SORTED_BAMS_PATHS [SORTED_BAMS_PATHS ...] -o OUTPUT_PATH -temp TEMP_FILE_PATH [-db DB_FILES_PATH] [--device DEVICE]
                 [--n_views N_VIEWS] [--min_contig_length MIN_CONTIG_LENGTH] [--batch_size BATCH_SIZE] [--epoch_base EPOCH_BASE] [--num_workers NUM_WORKERS]

DeeperBin Is a Binner with Dynamic Contrastive Learning with Pretrained Deep Language Model.

optional arguments:
  -h, --help            show this help message and exit
  -c CONTIG_PATH, --contig_path CONTIG_PATH
                        Contig fasta file path.
  -b SORTED_BAMS_PATHS [SORTED_BAMS_PATHS ...], --sorted_bams_paths SORTED_BAMS_PATHS [SORTED_BAMS_PATHS ...]
                        The sorted bam files path. You can set one bam file for single-sample binning and multiple bam files for multi-sample binning.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        The folder to store final MAGs.
  -temp TEMP_FILE_PATH, --temp_file_path TEMP_FILE_PATH
                        The folder to store temporay files.
  -db DB_FILES_PATH, --db_files_path DB_FILES_PATH
                        The folder contains temporay files. You can ignore it if you set the 'DeeperBin_DB' environmental variable.
  --device DEVICE       The device for training. Default is cuda:0. We highly recommand to use GPU but not CPU. You can adjust 'batch_size' parameter to fit
                        your GPU's memory. We need 24GB GPU memory to run the default settings. You can use CPU if you set this parameter with 'cpu'.
  --n_views N_VIEWS     Number of views to generate for each contig during training. Defaults to 6.
  --min_contig_length MIN_CONTIG_LENGTH
                        The minimum length of contigs for binning. Defaults to 768.
  --batch_size BATCH_SIZE
                        The batch size. Defaults to 1024.
  --epoch_base EPOCH_BASE
                        Number of basic training epoches. Defaults to 36.
  --num_workers NUM_WORKERS
                        Number of cpus for clustering contigs. Defaults to None. We would set 1 / 2 of total cpus if it is None.
```


**2.  You can run DeeperBin through the **binning_with_all_steps** function in Python.**

```
from DeeperBin.Binning import binning_with_all_steps

if __name__ == "__main__":
    contig_path = "contigs.fasta"
    bam_list = ["bam_file1", "bam_file2"]
    temp_path = "/temp_folder/"
    bin_output_folder = "/bin_output_folder/"

    binning_with_all_steps(
        contig_file_path=contig_path,
        sorted_bam_file_list=bam_list,
        temp_file_folder_path=temp_path,
        bin_output_folder_path=bin_output_folder,
        db_folder_path="./DeepMetaBin-DB",
        training_device="cuda:0",
    )
```

## Files in the output directory
- #### The binned MAGs.

- #### MetaInfo.tsv
This file contains the following columns: 

1. MAG name (first column), 
2. completeness of MAG (second column), 
3. contamination of MAG (third column), 
4. MAG quality (fourth column),

## Minimum System Requirements for Running DeeperBin
- System: Linux
- CPU: No restriction.
- RAM: >= 80 GB
- GPU: The GPU memory must be equal to or greater than 6GB.

## Our System Config
- System: NVIDIA DGX Server Version 5.5.1 (GNU/Linux 5.4.0-131-generic x86_64)
- CPU: AMD EPYC 7742 64-Core Processor (2 Sockets)
- RAM: 1TB
- GPU: 8 GPUs (A100-40GB)

## Repo Contents
- [DeeperBin-DB](./DeeperBin-DB): The model weights and other necessary files for running DeeperBin.
- [DeeperBin](./DeeperBin): The main codes (Python) of DeeperBin.






