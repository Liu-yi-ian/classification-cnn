"""Draw the same number of training examples from every class each epoch."""
import math

import torch
from torch.utils.data import Sampler


class BalancedSampler(Sampler):
    """Shuffle class-balanced blocks while keeping the requested optimizer batch size."""
    def __init__(self, dataset, config):
        self.seed = config['seed']
        self.epoch = 0
        self.class_count = len(dataset.classes)
        self.per_class = config['samples_per_class']
        # Twelve classes do not divide a batch of 32. A block of 96 images
        # contains eight examples per class and spans three optimizer updates.
        self.block_size = math.lcm(config['batch_size'], self.class_count)
        self.block_per_class = self.block_size // self.class_count
        self.pools = [[i for i, label in enumerate(dataset.labels) if label == c]
                      for c in range(self.class_count)]

    def set_epoch(self, epoch):
        """Change the shuffle deterministically for the next epoch."""
        self.epoch = epoch

    def __len__(self):
        return self.class_count * self.per_class

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = []
        for pool in self.pools:
            # Cycle through shuffled class pools. Repeated draws let small
            # classes contribute the same number of examples as larger ones.
            values = []
            usable = len(pool) // self.block_per_class * self.block_per_class
            while len(values) < self.per_class:
                order = torch.randperm(len(pool), generator=generator)[:usable]
                values.extend(pool[i] for i in order.tolist())
            draws.append(values[:self.per_class])

        indices = []
        for offset in range(0, self.per_class, self.block_per_class):
            block = [i for values in draws
                     for i in values[offset:offset + self.block_per_class]]
            # Mix class order within the block before the DataLoader forms batches.
            order = torch.randperm(len(block), generator=generator).tolist()
            indices.extend(block[i] for i in order)
        return iter(indices)
