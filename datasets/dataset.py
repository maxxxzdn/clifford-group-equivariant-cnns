import torch
from torch.utils.data import Dataset
import os


class TorchedDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        time_history: int,
        time_future: int,
        preprocess_fn: callable,
        make_channels: bool = False,
        num_data: int = -1,
    ):
        self.datadir = datadir
        self.time_history = time_history
        self.time_future = time_future
        self.preprocess_fn = preprocess_fn
        self.make_channels = make_channels

        if num_data == -1:
            self.num_data = len(os.listdir(self.datadir))
        else:
            self.num_data = num_data
            if num_data > len(os.listdir(self.datadir)):
                raise ValueError(
                    f"num_data={num_data} but only {len(os.listdir(self.datadir))} data points found in directory."
                )

    def __len__(self):
        return self.num_data

    def preprocess(self, x):
        return self.preprocess_fn(x)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.datadir, f"{idx}.pt"))
        data = self.preprocess(data)
        start = torch.randint(
            0, data.shape[0] - self.time_history - self.time_future, (1,)
        ).item()
        sample = data[start : start + self.time_history + self.time_future]
        x = sample[: self.time_history]
        y = sample[self.time_history :]

        if self.make_channels:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        return x, y
