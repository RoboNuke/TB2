from typing import List, Optional, Tuple, Union
from skrl.memories.torch import RandomMemory
import torch
import numpy as np


class AdaptiveRandomMemory(RandomMemory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def sample_all(
        self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample all data from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from memory.
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of torch.Tensor list
        """
        # sequential order
        if sequence_length > 1:
            if mini_batches > 1:
                batches = np.array_split(self.all_sequence_indexes, mini_batches)
                return [[self.tensors_view[name][batch] for name in names] for batch in batches]
            return [[self.tensors_view[name][self.all_sequence_indexes] for name in names]]

        # default order
        if mini_batches > 1:
            if self.filled:
                batch_size = (self.memory_size * self.num_envs) // mini_batches
            else:
                print("Data Points:", (self.num_envs * self.memory_index + self.env_index))
                batch_size = (self.num_envs * self.memory_index + self.env_index) // mini_batches
                print("Batch Size:", batch_size)
            batches = [(batch_size * i, batch_size * (i + 1)) for i in range(mini_batches)]
            return [[self.tensors_view[name][batch[0] : batch[1]] for name in names] for batch in batches]
        return [[self.tensors_view[name] for name in names]]
    
    def get_tensor_by_name(self, name: str, keepdim: bool = True) -> torch.Tensor:
        """Get a tensor by its name

        :param name: Name of the tensor to retrieve
        :type name: str
        :param keepdim: Keep the tensor's shape (memory size, number of environments, size) (default: ``True``)
                        If False, the returned tensor will have a shape of (memory size * number of environments, size)
        :type keepdim: bool, optional

        :raises KeyError: The tensor does not exist

        :return: Tensor
        :rtype: torch.Tensor
        """
        return self.tensors[name][:self.memory_index, :, :] if keepdim else self.tensors_view[name]

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor) -> None:
        """Set a tensor by its name

        :param name: Name of the tensor to set
        :type name: str
        :param tensor: Tensor to set
        :type tensor: torch.Tensor

        :raises KeyError: The tensor does not exist
        """
        with torch.no_grad():
            self.tensors[name][:self.memory_index, :, :].copy_(tensor)