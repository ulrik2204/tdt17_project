from typing import TypedDict

import torch
import torch.nn as nn
from segmentation_models_pytorch import Unet, UnetPlusPlus
from torch.optim import Optimizer


def get_unet_model(in_channels: int, out_channels: int):
    return Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_channels,
    )


def get_unetpluplus_model(in_channels: int, out_channels: int):
    return UnetPlusPlus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        decoder_attention_type="scse",
        in_channels=in_channels,
        classes=out_channels,
    )


class ModelCheckpoint(TypedDict):
    epoch: int
    state_dict: dict[str, torch.Tensor]
    optimizer: dict[str, torch.Tensor]


def save_state_dict(model: nn.Module, optimizer: Optimizer, epoch: int, path: str):
    checkpoint: ModelCheckpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_state_dict(model: nn.Module, optimizer: Optimizer | None, path: str):
    checkpoint: ModelCheckpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"] - 1
