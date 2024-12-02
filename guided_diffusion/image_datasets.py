import math
import random
import numpy as np
from PIL import Image
import blobfile as bf
import mpi4py
from mpi4py import MPI
import scipy.io
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
data_dir = "/kaggle/input/lpet-new-1/train_mat_2"

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each image is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "mat"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        data = loadmat(path)  # Load .mat file content

        if "img" not in data:  # Replace 'img' with the correct key if necessary
            raise KeyError(f"'img' key not found in {path}")

        mat_data = data["img"]  # Extract the image data from the .mat file
        mat_data = mat_data.astype(np.float32)  # Convert to float if necessary

        # Ensure contiguity and copy the numpy array to avoid negative strides
        mat_data = np.ascontiguousarray(mat_data)  # Make it contiguous if it's not
        mat_data = np.copy(mat_data)  # Ensure it's copied
        
        mat_data = np.squeeze(mat_data)  # Removes singleton dimension if present

        # Normalize the data between -1 and 1 if necessary (optional)
        mat_data = (mat_data - mat_data.min()) / (mat_data.max() - mat_data.min()) * 2 - 1

        # Ensure that mat_data is of correct shape: (height, width, channels)
        if mat_data.ndim == 2:  # If it's 2D, assume a single channel (grayscale)
            mat_data = mat_data[:, :, np.newaxis]  # Add a dummy channel dimension
            mat_data = (mat_data - np.min(mat_data)) / (np.max(mat_data) - np.min(mat_data))  # Normalize to [0, 1]
            mat_data = (mat_data * 255).astype(np.uint8)  # Scale to [0, 255]

        # Convert grayscale (1 channel) to RGB (3 channels)
        mat_data = np.stack((mat_data, mat_data, mat_data), axis=-1)  # Shape becomes (256, 128, 3)

        # Optional: Random flip for augmentation
        if self.random_flip and random.random() < 0.5:
            mat_data = np.flip(mat_data, axis=1)
            

        # Ensure it's a contiguous array before conversion to tensor
        mat_data = np.ascontiguousarray(mat_data)
        mat_data = np.copy(mat_data)  # Make sure the array is a new copy
        mat_data = np.squeeze(mat_data)
        mat_data = mat_data.astype(np.float32) / 127.5 - 1
        
        # Convert to PyTorch format: [C, H, W] instead of [H, W, C]
        mat_data = np.transpose(mat_data, (2, 0, 1))  # Shape becomes (3, 256, 128)

        

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # Print the shape for debugging
        #print(f"Shape of mat_data: {mat_data.shape}")
        #print(f"Dtype of mat_data: {mat_data.dtype}")

        # Return the data and optional labels
        return mat_data, out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
