

import numpy as np

from Src.IO import readPickle

if __name__ == "__main__":
    training_data_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine-auto-and-contrast/marine-sample-5-temp-folder/training_data/training_data.npy"
    contig_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine-auto-and-contrast/marine-sample-5-temp-folder/contigname2seq_str.pkl"
    
    contigname2seq = readPickle(contig_path)
    save_array = np.load(training_data_path, allow_pickle=True)
    for cur_contigname, cur_tuples in save_array:
        print(cur_tuples)
        print(contigname2seq[">" + cur_contigname])
        break
    


