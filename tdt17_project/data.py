from typing import Literal

from torchvision.datasets import Cityscapes


def get_dataset(
    path: str, split: Literal["train"] | Literal["val"] | Literal["test"] = "train"
):
    return Cityscapes(
        path,
        split=split,
        mode="fine",
        target_type="semantic",
        transform=None,
        target_transform=None,
    )
