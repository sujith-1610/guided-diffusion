import math
import random
from scipy.io import loadmat
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

data_dir = "/content/drive/MyDrive/train_mat"

def load_data(
    *,
    data_dir=data_dir,
    batch_size,
    image_size=None,  # Not used for .mat files
    class_cond=False,
    deterministic=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (data, kwargs) pairs.

    Each pair consists of:
    - Data: Tensor from .mat files.
    - kwargs: Optional metadata or labels.

    :param data_dir: Dataset directory containing .mat files.
    :param batch_size: Batch size of each returned pair.
    :param class_cond: If True, include a "y" key in returned dicts for class labels.
    :param deterministic: If True, yield results in a deterministic order.
    :param random_flip: If True, randomly flip the data for augmentation.
    """
    if not data_dir:
        raise ValueError("Unspecified data directory")
    
    print(f"Loading data from: {data_dir}")
    all_files = _list_mat_files_recursively(data_dir)
    classes = None

    if class_cond:
        # Extract class names from file names
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = MatDataset(
        mat_paths=all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_flip=random_flip,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )

    while True:
        yield from loader


def _list_mat_files_recursively(data_dir):
    """
    Recursively list all .mat files in the specified directory.
    """
    print(f"Listing .mat files in directory: {data_dir}")
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        if entry.endswith(".mat"):  # Only include .mat files
            results.append(full_path)
        elif bf.isdir(full_path):  # Recurse into subdirectories
            results.extend(_list_mat_files_recursively(full_path))
    return results


class MatDataset(Dataset):
    def __init__(
        self,
        mat_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_flip=True,
    ):
        """
        Dataset for loading and processing .mat files.

        :param mat_paths: List of paths to .mat files.
        :param classes: Optional class labels for each .mat file.
        :param shard: Current shard index (for distributed training).
        :param num_shards: Total number of shards (for distributed training).
        :param random_flip: Whether to randomly flip data for augmentation.
        """
        super().__init__()
        self.local_mats = mat_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_mats)

    def __getitem__(self, idx):
        path = self.local_mats[idx]
        # Load the .mat file
        data = loadmat(path)  # Load .mat file content
        if "data" not in data:  # Replace "data" with the actual key in your .mat files
            raise KeyError(f"'data' key not found in {path}")

        mat_data = data["data"]  # Extract the data array
        mat_data = mat_data.astype(np.float32)  # Convert to float if necessary

        # Normalize the data between -1 and 1
        mat_data = (mat_data - mat_data.min()) / (mat_data.max() - mat_data.min()) * 2 - 1

        # Optional: Random flip for augmentation
        if self.random_flip and random.random() < 0.5:
            mat_data = np.flip(mat_data, axis=1)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # Return the data and optional labels
        return mat_data, out_dict
