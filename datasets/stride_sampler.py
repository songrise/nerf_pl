# -*- coding : utf-8 -*-
# @FileName  : stride_sampler.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Jun 18, 2022
# @Github    : https://github.com/songrise
# @Description: Stride-sampling for a given dataset of rays/images.

import torch

class StrideSampler(torch.utils.data.sampler.Sampler):
    """Stride-sampling for a given dataset of rays/images.
    """
    def __init__(self, dataset, batch_size, stride, h,w):
        self.dataset = dataset
        self.batch_size = batch_size
        self.stride = stride

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    @property
    def indices(self):
        return list(range(0, len(self.dataset), self.stride))

    @property
    def num_samples(self):
        return len(self.indices) // self.batch_size