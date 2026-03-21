# import libraries
import os
import os
import torch
import monai
import torch
import monai
import contextlib
import matplotlib
import contextlib
import matplotlib
import numpy as np
import torchvision
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from torchinfo import summary
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torchvision import datasets
from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
from monai.apps import MedNISTDataset
from monai.apps import MedNISTDataset
from monai.networks.layers import Act
from torch.utils.data import random_split
from torch.utils.data import random_split
from monai.data import Dataset, DataLoader
from IPython.display import Image, display
from monai.data import Dataset, DataLoader
from IPython.display import Image, display
from monai.networks.nets import AutoencoderKL
from matplotlib.animation import FuncAnimation
from monai.data import Dataset as MonaiDataset
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from monai.transforms import Compose, LoadImage, ToTensor
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.transforms import LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Resized
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Resized

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Monai version: {monai.__version__}")


print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Monai version: {monai.__version__}")
##

# Path to store MedNIST dataset
data_dir = "data/MedNIST"

# Checks if data has already been downloaded
download = not os.path.exists(data_dir)

# Load dataset without reloading if data already exists
train_set = MedNISTDataset(root_dir="data", section="training", download=download, seed=0)
test_set = MedNISTDataset(root_dir="data", section="validation", download=download, seed=0)

##
# how many samples per batch to load
batch_size = 64
image_size = 64
train_valid_ratio = 0.8
minv = 0  # min intensity value of each image after rescaling
maxv = 1  # max intensity value of each image after rescaling

labels = ["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"]
selected_label = labels[4]

# keep only the Hand images
train_datalist = [
    {"image": item["image"], "label": selected_label} for item in train_set.data if item["class_name"] == selected_label
]
test_datalist = [
    {"image": item["image"], "label": selected_label} for item in test_set.data if item["class_name"] == selected_label
]

all_transforms = [
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=minv, b_max=maxv, clip=True),
    Resized(keys=["image"], spatial_size=[image_size, image_size]),
]

train_dataset = Dataset(data=train_datalist, transform=Compose(all_transforms))
test_data = Dataset(data=test_datalist, transform=Compose(all_transforms))

# split the train_data into a train (80%) and valid (20%) subdataset
train_size = int(train_valid_ratio * len(train_dataset))  # 80% for training
valid_size = len(train_dataset) - train_size  # 20% for validation
train_data, valid_data = random_split(train_dataset, [train_size, valid_size])

# prepare data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

print(f"Training dataset size: {len(train_loader.dataset)}")
print(f"Validation dataset size: {len(valid_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")
