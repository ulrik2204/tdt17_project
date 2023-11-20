from typing import Any, Callable, Literal

from torchvision.datasets import Cityscapes


def get_dataset(
    path: str,
    split: Literal["train"] | Literal["val"] | Literal["test"] = "train",
    transforms: Callable[[Any], Any] | None = None,
    target_transform: Callable[[Any], Any] | None = None,
):
    return Cityscapes(
        path,
        split=split,
        mode="fine",
        target_type="semantic",
        transforms=transforms,
        target_transform=target_transform,
    )
