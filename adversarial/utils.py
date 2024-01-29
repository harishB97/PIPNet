import heapq
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def unshuffle_dataloader(dataloader):
    if type(dataloader.dataset) == ImageFolder:
        dataset = dataloader.dataset
    else:
        dataset = dataloader.dataset.dataset.dataset
    new_dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers
    )
    return new_dataloader

def get_heap():
    list_ = []
    heapq.heapify(list_)
    return list_

def get_normalize_transform():
    return transforms.Normalize(mean=mean,std=std)

def undo_preprocess(x):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def get_less_activation_locations_mask(tensor, threshold, window_size):
    """
    tensor.shape => [H, W]
    """
    # Find the max value and its location
    max_val = torch.max(tensor)
    max_loc = torch.nonzero(tensor == max_val, as_tuple=False)[0]

    # Calculate half window size for easier calculations
    half_window = window_size // 2

    # Create a mask for the window and the threshold
    mask = torch.ones_like(tensor, dtype=torch.bool)

    # Handling the edge cases for the window
    for i in range(max(0, max_loc[0] - half_window), min(tensor.shape[0], max_loc[0] + half_window + 1)):
        for j in range(max(0, max_loc[1] - half_window), min(tensor.shape[1], max_loc[1] + half_window + 1)):
            mask[i, j] = False

    # Apply the threshold mask
    mask &= tensor <= threshold

    return mask

def get_less_activation_locations(tensor, threshold, window_size):
    """
    tensor.shape => [H, W]
    """
    mask = get_less_activation_locations_mask(tensor, threshold, window_size)

    # Extract the valid locations
    valid_locations = torch.nonzero(mask, as_tuple=False).numpy().tolist()

    # Convert the locations to tuples
    valid_locations = [tuple(loc) for loc in valid_locations]

    return valid_locations

# # Test the function with a 3x3 window
# window_size_example = 3
# valid_locations_pytorch_window = get_valid_locations_pytorch(tensor_example_pytorch, threshold_example_pytorch, window_size_example)
# valid_locations_pytorch_window
