import torch
from torch.utils.data import Sampler
import numpy as np



class PartitionBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_partitions, shuffle=True):
        """
        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): The size of the batches to generate.
            num_partitions (int): Number of partitions to divide the dataset into.
            shuffle (bool): Whether to shuffle samples within each partition.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.shuffle = shuffle
        
        self.num_samples = len(dataset)
        self.partition_size = self.num_samples // self.num_partitions
        self.partitions = self._create_partitions()

    def _create_partitions(self):
        indices = np.arange(self.num_samples)
        
        # Partition and handle leftovers
        partitions = [
            indices[i * self.partition_size:(i + 1) * self.partition_size]
            for i in range(self.num_partitions)
        ]
        # Include any leftover samples in the last partition
        leftover = indices[self.num_partitions * self.partition_size:]
        if leftover.size > 0:
            partitions[-1] = np.concatenate((partitions[-1], leftover))
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