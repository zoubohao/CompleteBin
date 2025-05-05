
import torch
import numpy as np

if __name__ == "__main__":
    # base_pair_arrary = np.arange(100)
    # cutoff = np.percentile(base_pair_arrary, q = 97.5)
    # del_index = base_pair_arrary > cutoff
    # base_pair_arrary[del_index] = cutoff
    # print(base_pair_arrary, cutoff, del_index)
    a = torch.randn(10, 5)
    b = torch.rand(1)[None, :]
    print(a.shape, b.shape)
    print(a)
    print(b)
    print(a / b)


