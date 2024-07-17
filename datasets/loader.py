import jax
import numpy as np
from torch.utils.data import default_collate, DataLoader

from .dataset import TorchedDataset


def numpy_collate(batch):
    return jax.tree_util.tree_map(np.asarray, default_collate(batch))


class NumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def create_data_loader(
    num_data,
    datadir,
    batch_size,
    time_history,
    time_future,
    preprocess_fn,
    make_channels,
    shuffle,
    **kwargs
):
    dataset = TorchedDataset(
        num_data=num_data,
        datadir=datadir,
        time_history=time_history,
        time_future=time_future,
        preprocess_fn=preprocess_fn,
        make_channels=make_channels,
    )
    return NumpyLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
