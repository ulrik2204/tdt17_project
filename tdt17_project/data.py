from typing import Any, Callable, Literal

import albumentations as A
import albumentations.pytorch as AP
import numpy as np
from torchvision.datasets import Cityscapes


def get_dataset(
    path: str,
    split: Literal["train"] | Literal["val"] | Literal["test"] = "train",
    transform_image_and_target: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    transform_image: Callable[[Any], Any] | None = None,
    target_transform: Callable[[Any], Any] | None = None,
):
    return Cityscapes(
        path,
        split=split,
        mode="fine",
        target_type="semantic",
        transforms=transform_image_and_target,
        transform=transform_image,
        target_transform=target_transform,
    )


def get_image_target_transform():
    trans = A.Compose(
        [
            A.Resize(256, 256),
            A.RandomResizedCrop(256, 256),
            A.RandomRotate90(),
            A.ColorJitter(),
            A.MotionBlur(),
            A.OneOf(
                [
                    A.GaussNoise(),
                    A.OpticalDistortion(),
                    A.ElasticTransform(),
                    A.GridDistortion(),
                    A.RandomBrightnessContrast(),
                ],
                p=0.5,
            ),
            AP.ToTensorV2(),
        ]
    )

    def transform_image_and_target(image, target):
        transformed = trans(image=np.array(image), mask=np.array(target))
        return transformed["image"], transformed["mask"]

    return transform_image_and_target


def get_val_test_transform():
    trans = A.Compose(
        [
            A.Resize(256, 256),
            AP.ToTensorV2(),
        ]
    )

    def transform_image_and_target(image, target):
        transformed = trans(image=np.array(image), mask=np.array(target))
        return transformed["image"], transformed["mask"]

    return transform_image_and_target
