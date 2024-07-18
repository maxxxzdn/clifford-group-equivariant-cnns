import os
import h5py
import torch
import numpy as np
from torchvision.transforms import Resize
from tqdm import tqdm


def preprocess():
    raw_dir = "raw"
    output_base_dir = "."

    # Create output directories if they don't exist
    for partition in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_base_dir, partition), exist_ok=True)

    # Initialize counters for each partition
    counters = {"train": 0, "valid": 0, "test": 0}

    # Create a resizing function
    resize = Resize((64, 64))

    print("Preprocessing data...")
    for filename in tqdm(os.listdir(raw_dir)):
        if filename.endswith(".h5"):
            file_path = os.path.join(raw_dir, filename)

            with h5py.File(file_path, "r") as f:
                # Determine the partition
                partition = (
                    "train"
                    if "train" in filename
                    else "valid" if "valid" in filename else "test"
                )

                # Extract fields
                u = f[partition]["u"][:]
                vx = f[partition]["vx"][:]
                vy = f[partition]["vy"][:]

                # Stack fields
                data = np.stack([u, vx, vy], axis=-1)

                # Convert to torch tensor (float32)
                data_tensor = torch.from_numpy(data).float()

                # Resize
                B, T, X, Y, C = data_tensor.shape
                data_tensor = data_tensor.permute(0, 1, 4, 2, 3).reshape(B * T, C, X, Y)
                data_tensor = resize(data_tensor)
                data_tensor = data_tensor.reshape(B, T, C, 64, 64).permute(
                    0, 1, 3, 4, 2
                )

                # Save each element of batch as a separate file
                for i in range(data_tensor.shape[0]):
                    output_filename = f"{counters[partition]}.pt"
                    output_path = os.path.join(
                        output_base_dir, partition, output_filename
                    )
                    torch.save(data_tensor[i], output_path)
                    counters[partition] += 1

    print("Preprocessing complete!")
    for partition, count in counters.items():
        print(f"Total {partition} samples: {count}")


if __name__ == "__main__":
    preprocess()
