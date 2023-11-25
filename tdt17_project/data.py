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


def apply_albumentations(image, target, transform):
    transformed = transform(image=np.array(image), mask=np.array(target))
    return transformed["image"], transformed["mask"]


def get_albumentations_transform_function(transform):
    def transform_image_and_target(image, target):
        return apply_albumentations(image, target, transform)

    return transform_image_and_target


def get_image_target_transform():
    trans = A.Compose(
        [
            A.Resize(512, 512),
            A.RandomResizedCrop(256, 256),
            A.Normalize(),
            A.ColorJitter(),
            A.HorizontalFlip(),
            A.ToGray(p=0.1),
            AP.ToTensorV2(),
        ]
    )

    return get_albumentations_transform_function(trans)


def get_val_test_transform():
    trans = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(),
            AP.ToTensorV2(),
        ]
    )
    return get_albumentations_transform_function(trans)
