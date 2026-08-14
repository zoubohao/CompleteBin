
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from CompleteBin.logger import get_logger

logger = get_logger()


class DeeperBinSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, data_size: int, batch_size: int, min_training_step: int, seed: int = 2048) -> None:
        self.data_size = data_size
        self.batch_size = batch_size
        self.train_step = self.data_size // self.batch_size
        assert data_size % batch_size == 0, ValueError(f'The size of data can not divide batch_size')
        self.base_seed = seed
        self.expand_ratio = 1
        if self.train_step + 10 < min_training_step:
            self.expand_ratio = min_training_step // self.train_step + 1
        self.epoch = 0
        self.final_sample = self.get_indices()
        logger.info(f"--> The training step for one epoch is {len(self.final_sample) / self.batch_size}.")

    def get_indices(self):
        indices_list = []
        for i in range(self.expand_ratio):
            generator = torch.Generator()
            generator.manual_seed(self.base_seed + self.epoch * self.expand_ratio + i)
            indices = torch.randperm(self.data_size, generator=generator).numpy()
            indices_list.append(indices)
        final_sample = np.concatenate(indices_list, axis=0)
        return final_sample

    def __iter__(self) -> Iterator[List[int]]:
        self.epoch += 1
        yield from map(int, self.get_indices())

    def __len__(self) -> int:
        return len(self.final_sample)
