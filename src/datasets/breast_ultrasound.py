import kagglehub
import numpy as np
from pathlib import Path

from ..utils import pre_processing_images

def get_breast_ultrasond_annotations(
    pre_processing: bool = True,
    target_size = (224, 224),
    image_dir: str = 'current_dataset',
    binary_data: bool = True
):
    root = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")

    path = Path(root)
    paths = list(path.glob('**/*.png'))

    paths = [str(p) for p in paths if "_mask" not in str(p)]
    targets = [p.split('/')[-2] for p in paths]

    if pre_processing:
        paths = pre_processing_images(image_dir, paths, target_size)

    class_to_idx = {str(cls_):idx for idx, cls_ in enumerate(np.unique(targets))}

    if binary_data:
        class_to_idx['benign'] = 1
        class_to_idx['malignant'] = 1
        class_to_idx['normal'] = 0
        
    return np.array(paths), np.array(targets), class_to_idx