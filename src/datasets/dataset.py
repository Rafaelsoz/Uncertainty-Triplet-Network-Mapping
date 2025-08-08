from numpy import ndarray
from typing import Callable
from PIL import Image

from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(
        self,
        paths: ndarray,
        targets: ndarray,
        class_to_idx: dict,
        transform: Callable = None
    ):
        self.paths = paths
        self.targets = targets
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

      img_path, label = self.paths[idx],  self.targets[idx]
      image = Image.open(img_path)

      if self.transform:
          image = self.transform(image)

      label = self.class_to_idx[label]

      return image, label