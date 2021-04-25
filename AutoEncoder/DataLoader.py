from typing import List

import numpy as np
import torch
import scipy
from scipy.sparse import csr_matrix, csc_matrix


class DatasetRNA(torch.utils.data.IterableDataset):
    """
        Dataset object for loading tensors from a matrix, rows are cells and columns are genes

        Parameters
        ----------
        counts_mtx_list
            A list of csr_matrix, should have same rows
        shuffle
            Whether the data should be shuffled
        """

    def __init__(
            self,
            counts_mtx_list: List[csr_matrix] = None,
            shuffle=True,
    ):
        counts_mtx = scipy.sparse.vstack(counts_mtx_list)
        dataset_indices = []
        for i in range(len(counts_mtx_list)):
            dataset_indices += [i] * counts_mtx_list[i].shape[0]
        dataset_indices = np.array(dataset_indices)
        if shuffle:
            index = np.arange(counts_mtx.shape[0])
            np.random.shuffle(index)
            counts_mtx = counts_mtx[index, :]
            dataset_indices = dataset_indices[index]
        self.counts_mtx = counts_mtx
        self.dataset_indices = dataset_indices
        self.n_cells = counts_mtx.shape[0]
        self.n_genes = counts_mtx.shape[1]
        self.original_index = index

    def __iter__(self):
        for i in range(self.counts_mtx.shape[0]):
            yield (self.counts_mtx[i, :].toarray().reshape((self.n_genes,)), np.int(self.dataset_indices[i]))

