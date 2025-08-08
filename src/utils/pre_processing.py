import os
import shutil

from PIL import Image
from tqdm import tqdm

def pre_processing_images(
    dir: str,
    paths: list,
    target_size: tuple = (224, 224)
):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    new_paths = []
    for idx, path in tqdm(enumerate(paths), total=len(paths)):

        new_path = f"{dir}/image_{idx}.jpg"
        new_paths.append(new_path)

        image = Image.open(path)
        image = image.convert('RGB')
        image = image.resize(target_size)
        image.save(new_path, format="JPEG")

    return new_paths