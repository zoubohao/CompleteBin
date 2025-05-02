# CompleteBin
**Paper --> Dynamic Contrastive Learning with Pretrained Deep Language Model Enhances Metagenome Binning for Contigs**

CompleteBin is a binner to cluster the contigs with dynamic contrastive learning and a pretrained deep language model.

## Installation (Have Verified):
#### 0. Please install Python 3.9.15.

#### 1. FIRST STEP (Create Conda Environment for CompleteBin)
Create CompleteBin's conda environment by using this command:
```
conda env create -n CompleteBin -f completebin-conda-env.yml
```

**And**

Download PyTorch v2.1.0 -cu*** (or higher version) from **[http://pytorch.org/](http://pytorch.org/)** if you want to use GPUs (We highly recommend using GPUs). For example:
```
conda activate CompleteBin
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 2. SECOND STEP (Install Codes of CompleteBin)
After preparing the environment, the code of CompleteBin can be installed via pip. 
```
conda activate CompleteBin
pip install CompleteBin==1.0.4
```
This installation will run for around 10 minutes.

## Download Pretrained Weight and Other Files for Running
Download the pretrained weight and other files (**CompleteBin-DB.zip**) for running CompleteBin from this **[LINK](https://drive.google.com/file/d/1MLpt68I7MVZPKvwkjCOgDi0yPLfWRz7E/view?usp=sharing)**.


#### 1. Set Environmental Variable
- Unzip the downloaded file (**CompleteBin-DB.zip**) and set an **environmental variable** called "CompleteBin_DB" by adding the following line to the last line of the .bashrc file (The path of the file: ~/.bashrc):
```
export CompleteBin_DB=/path/of/this/CompleteBin-DB/
```
For example: 'export CompleteBin_DB=/home/csbhzou/software/CompleteBin-DB/'.

- Save the .bashrc file, and then execute:
```
source .bashrc
```

#### 2. Using the '-db' or '--db_files_path' flag in CLI

- **You can set the '-db' flag in CLI to the path of the 'CompleteBin-DB' folder if you do not want to set the environmental variable.**


## Running CompleteBin


**1.  You can run CompleteBin with 'clean' mode through the following command:**

```
CompleteBin version: *** v1.0.4 ***
usage: completebin [-h] -c CONTIG_PATH -b SORTED_BAMS_PATHS [SORTED_BAMS_PATHS ...] -o OUTPUT_PATH -temp TEMP_FILE_PATH [-db DB_FILES_PATH] [--device DEVICE]
                   [--n_views N_VIEWS] [--min_contig_length MIN_CONTIG_LENGTH] [--batch_size BATCH_SIZE] [--base_epoch BASE_EPOCH] [--num_workers NUM_WORKERS]
                   [--auto_min_length AUTO_MIN_LENGTH] [--step_num STEP_NUM]

CompleteBin is a Binner with Sequence Patch Embedding, Pretrained Deep Language Model, and Dynamic Contrastive Learning.

optional arguments:
  -h, --help            show this help message and exit
  -c CONTIG_PATH, --contig_path CONTIG_PATH
                        Contig fasta file path.
  -b SORTED_BAMS_PATHS [SORTED_BAMS_PATHS ...], --sorted_bams_paths SORTED_BAMS_PATHS [SORTED_BAMS_PATHS ...]
                        The sorted BAM files path. You can set a BAM file for single-sample binning and multiple BAM files for multi-sample binning.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        The folder to store the final MAGs.
  -temp TEMP_FILE_PATH, --temp_file_path TEMP_FILE_PATH
                        The folder to store temporary files during binning processing.
  -db DB_FILES_PATH, --db_files_path DB_FILES_PATH
                        The folder contains database files. You can ignore it if you set the 'CompleteBin_DB' environmental variable.
  --device DEVICE       The device uses for training. The default is CPU. We highly recommend using GPU, but not CPU. We need 24GB of GPU memory to run the default settings. You can adjust the 'batch_size' parameter to fit your
                        GPU's memory. You can use CPU if you set this parameter to 'cpu'.
  --n_views N_VIEWS     Number of views to generate for each contig during training. Defaults to 6.
  --min_contig_length MIN_CONTIG_LENGTH
                        The minimum length of contigs for binning. Defaults to 900.
  --batch_size BATCH_SIZE
                        The batch size. Defaults to 1024.
  --base_epoch BASE_EPOCH
                        Number of basic training epochs. Defaults to 35.
  --num_workers NUM_WORKERS
                        Number of CPUs for clustering contigs. Defaults to None. We would set 1 / 3 of the total CPUs if it is None.
  --auto_min_length AUTO_MIN_LENGTH
                        Auto-determining the min length for this sample.
  --step_num STEP_NUM   The binning procedure can be divided into 3 steps. The first step (step 1) is to process the training data. Focusing on using
                        CPU. The second step (step 2) is the training procedure. Focusing on using the GPU. The third step (step 3) is clustering. Focusing on using
                        CPU. This function would combine these 3 steps if this parameter is None. Defaults to None.
```


**2.  You can run CompleteBin through the **binning_with_all_steps** function in Python.**

```
from Src.Binning_steps import binning_with_all_steps

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
        db_folder_path="./CompleteBin-DB",
        training_device="cuda:0",
    )
```

## Files in the output directory
- #### The binned MAGs.

- #### MetaInfo.tsv
This file contains the following columns: 

1. MAG name (first column), 
2. completeness of MAG (third column), 
3. contamination of MAG (fourth column), 
4. MAG quality (fifth column),

## Minimum System Requirements for Running CompleteBin
- System: Linux
- CPU: No restriction.
- RAM: >= 80 GB
- GPU: The GPU memory must be equal to or greater than 24GB.

## Our System Config
- System: NVIDIA DGX Server Version 5.5.1 (GNU/Linux 5.4.0-131-generic x86_64)
- CPU: AMD EPYC 7742 64-Core Processor (2 Sockets)
- RAM: 1TB
- GPU: 8 GPUs (A100-40GB)

## Repo Contents
- [CompleteBin-DB](./CompleteBin-DB): The model weights and other necessary files for running CompleteBin.
- [Src](./CompleteBin): The main code (Python) of CompleteBin.






