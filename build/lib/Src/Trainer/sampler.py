


import random
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from Src.logger import get_logger

logger = get_logger()


class DeeperBinSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, data_size: int, batch_size: int) -> None:
        self.data_size = data_size
        self.batch_size = batch_size
        self.train_step = self.data_size // self.batch_size
        assert self.train_step > 0, ValueError(f'The number of contigs smaller than one batch size.')
        self.whole_trained_data_size = self.train_step * self.batch_size
        
        self.expand_ratio = 1
        if self.train_step + 1 < 60:
            self.expand_ratio = 60 // (self.train_step + 1) + 1
        
        self.final_sample = self.get_indices()
        logger.info(f"--> The training step for one epoch is {len(self.final_sample) / self.batch_size}. New")
    
    def complete_last_batch(self, last_indices):
        redundent_indices = set(last_indices.tolist())
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(self.data_size, generator=generator).numpy()
        for index in indices:
            if int(index) not in redundent_indices:
                redundent_indices.add(int(index))
            if len(redundent_indices) >= self.batch_size:
                break
        return np.array(list(redundent_indices))
    
    def get_indices(self):
        indices_list = []
        for i in range(self.expand_ratio):
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            indices = torch.randperm(self.data_size, generator=generator).numpy()
            # complete the last batch
            last_batch_indices = self.complete_last_batch(indices[self.whole_trained_data_size:])
            indices = np.concatenate([indices[0: self.whole_trained_data_size], last_batch_indices], axis=0)
            indices_list.append(indices)
        final_sample = np.concatenate(indices_list, axis=0)
        return final_sample

    def __iter__(self) -> Iterator[List[int]]:
        yield from map(int, self.get_indices())

    def __len__(self) -> int:
        return len(self.final_sample)




            # redundent_indices = set(indices[self.whole_trained_data_size:].tolist())
            # while len(redundent_indices) < self.batch_size:
            #     num = random.randint(0, self.data_size-1)
            #     if num not in redundent_indices:
            #         redundent_indices.add(num)
            # last_batch_indices = np.array(list(redundent_indices))