import torch
from torch.utils.data import Sampler
import numpy as np



class PartitionBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, partition_size, shuffle=True):
        """
        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): The size of the batches to generate.
            partition_size (int): Size of each partition (number of samples per partition).
            shuffle (bool): Whether to shuffle samples within each partition.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.partition_size = partition_size
        self.shuffle = shuffle
        
        self.num_samples = len(dataset)
        self.num_partitions = (self.num_samples + self.partition_size - 1) // self.partition_size
        self.partitions = self._create_partitions()

    def _create_partitions(self):
        indices = np.arange(self.num_samples)
        
        partitions = [
            indices[i * self.partition_size:min((i + 1) * self.partition_size, self.num_samples)]
            for i in range(self.num_partitions)
        ]
        return partitions

    def __iter__(self):
        for partition in self.partitions:
            if self.shuffle:
                np.random.shuffle(partition)
            for i in range(0, len(partition), self.batch_size):
                batch = partition[i:i + self.batch_size]
                yield batch

    def __len__(self):
        # Total number of batches across all partitions
        return sum(
            (len(part) + self.batch_size - 1) // self.batch_size
            for part in self.partitions
        )