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
            # Add more transforms here
            # A.HorizontalFlip(p=0.5),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
