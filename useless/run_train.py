
#coding: utf-8
import argparse
import sys
import pynvml
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9,10,11,12,13" 
import torch
import time


if __name__ == "__main__":
    
    myparser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]), description="DeeperBin Is a Binner with Dynamic Contrastive Learning with Pretrained Deep Language Model."
    )

    # Add parameters, required settings
    myparser.add_argument(
        "-m",
        "--gpu_memory",
        type=int,
        required=True,
        help="Contig fasta file path.")
    
    myparser.add_argument(
        "-i",
        "--gpu_ids",
        required=True,
        type=str,
        nargs="+",
        help="The sorted bam files path. " + \
            " You can set one bam file for single-sample binning and multiple bam files for multi-sample binning.")
    
    args = myparser.parse_args()
    gpu_mem = args.gpu_memory
    
    base = 1024**3 ## bit
    total_param = gpu_mem * base
    base_parameters =  100 * 100 * 8 * 3 * 1.1
    multi = total_param // int(base_parameters)
    
    pynvml.nvmlInit()
    while 1:
        for i in args.gpu_ids:
            i = int(i)
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(i))
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print('第'+str(i)+'块GPU剩余显存'+str(meminfo.free/(1024**3))+'GB')
            if meminfo.free/(1024**2)>=0:
                print(multi)
                a = torch.zeros((multi, 100, 100), dtype=torch.float64, requires_grad=True, device=f"cuda:{i}")
                b = torch.randn((multi, 100, 100), dtype=torch.float64, requires_grad=True, device=f"cuda:{i}")
                z = a @ b
                print('GPU has been grabbed!')
                time.sleep(5)
            else:
                print(f"GPU: {i} out of memory.")
