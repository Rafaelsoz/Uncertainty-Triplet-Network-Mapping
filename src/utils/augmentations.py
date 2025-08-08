from random import uniform
from torchvision.transforms.functional import InterpolationMode, affine

class CustomAugmentation:
    def __init__(self, translate_range=15, shear_range=15, rotate_range=25):
        self.translate_range = translate_range
        self.shear_range = shear_range
        self.rotate_range = rotate_range

    def __call__(self, img):
        # Step 1: Translate
        tx = uniform(-self.translate_range, self.translate_range)
        ty = uniform(-self.translate_range, self.translate_range)
        img = affine(
            img,
            angle=0,
            translate=(tx, ty),
            scale=1.0,
            shear=0,
            interpolation=InterpolationMode.BILINEAR
        )

        # Step 2: Shearing
        shear_x = uniform(-self.shear_range, self.shear_range)
        shear_y = uniform(-self.shear_range, self.shear_range)
        img = affine(
            img,
            angle=0,
            translate=(0, 0),
            scale=1.0,
            shear=(shear_x, shear_y),
            interpolation=InterpolationMode.BILINEAR
        )

        # Step 3: Rotation
        angle = uniform(-self.rotate_range, self.rotate_range)
        img = affine(
            img,
            angle=angle,
            translate=(0, 0),
            scale=1.0,
            shear=0,
            interpolation=InterpolationMode.BILINEAR
        )

        return img
