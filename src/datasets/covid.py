import kagglehub
import numpy as np
from pathlib import Path

from ..utils.pre_processing import pre_processing_images

def get_covid_annotations(
    pre_processing: bool = True,
    target_size = (224, 224),
    image_dir: str = 'current_dataset'
):
    root = kagglehub.dataset_download("plameneduardo/sarscov2-ctscan-dataset")
    
    path = Path(root)
    paths = list(path.glob('**/*.png'))
    
    paths = [str(p) for p in paths ]
    targets = [p.split('/')[-2] for p in paths]
    
    if pre_processing:
        paths = pre_processing_images(image_dir, paths, target_size)

    class_to_idx = {'non-COVID': 0, 'COVID': 1}
    
    return np.array(paths), np.array(targets), class_to_idx
