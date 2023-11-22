from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
import torch.nn as nn
import torch.optim
from matplotlib import pyplot as plt
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from tdt17_project.data import (
    get_dataset,
    get_image_target_transform,
    get_val_test_transform,
)
from tdt17_project.model import get_unet_model
from tdt17_project.utils import decode_segmap, encode_segmap

DATASET_BASE_PATH = "/cluster/projects/vc/data/ad/open/Cityscapes"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.005
WEIGHTS_FOLDER = "./weights"


def process_batch(
    image: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module,
    loss_criterion: nn.Module,
    metric: Callable[[Any, Any], Any],
    device: str,
):
    image = image.to(device).float()
    target = encode_segmap(target.to(device).long())  # to remove unwanted classes
    pred = model(image)
    loss = loss_criterion(pred, target)
    metric_score = metric(pred, target).detach().cpu()
    return loss, metric_score


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    optimizer: Optimizer,
    loss_criterion: nn.Module,
    metric: Callable[[Any, Any], Any],
    epochs: int,
    save_folder: str,
    device: str = "cuda",
):
    model.train()
    best_loss = 10000000
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        total_loss = 0
        total_metric = 0
        for index, (image, target) in (
            pbar := tqdm(enumerate(train_dl), total=len(train_dl))
        ):
            loss, metric_score = process_batch(
                image, target, model, loss_criterion, metric, device
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            total_metric += metric_score

            pbar.set_postfix_str(
                f"TEST: average loss {total_loss/(index+1):.3f}, {metric.__name__}: {total_metric/(index+1):.3f}"
            )
        validation_loss = evaluate_model(
            model, val_dl, loss_criterion, metric, "VAL", device
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
            torch.save(
                model.state_dict(),
                f"{save_folder}{time}_{validation_loss:.0f}_model.pt",
            )
        # scheduler.step() # (after epoch)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_criterion: nn.Module,
    metric: Callable[[Any, Any], Any],
    title: str,
    device="cuda",
):
    model.eval()
    total_loss = 0
    total_metric = 0
    with torch.no_grad():
        for index, (image, target) in (
            pbar := tqdm(enumerate(dataloader), total=len(dataloader))
        ):
            loss, metric_score = process_batch(
                image, target, model, loss_criterion, metric, device
            )
            total_loss += loss.detach().cpu()
            total_metric += metric_score.detach().cpu()
            pbar.set_postfix_str(
                f"{title}: average loss {total_loss/(index+1):.3f}, avg {metric.__name__}: {total_metric/(index+1):.3f}"
            )
    return total_loss / len(dataloader)


def show_image_segmentation_sample(real_image, pred_mask, real_mask):
    fig, ax = plt.subplots(ncols=3, figsize=(16, 50), facecolor="white")
    ax[0].imshow(np.moveaxis(real_image.numpy(), 0, 2))  # (3,256, 512)
    # ax[1].imshow(encoded_mask,cmap='gray') #(256, 512)
    ax[1].imshow(real_mask)  # (256, 512, 3)
    ax[2].imshow(pred_mask)  # (256, 512, 3)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[0].set_title("Input Image")
    ax[1].set_title("Ground mask")
    ax[2].set_title("Predicted mask")
    plt.savefig("result.png", bbox_inches="tight")


def show_model_segmentation_sample(model: nn.Module, examples: list[tuple[Any, Any]]):
    print("\n--- Model examples ---")
    model.eval()
    with torch.no_grad():
        for image, target in examples:
            pred = model(image).detach().cpu()
            decoded_pred = decode_segmap(torch.argmax(pred, dim=0).numpy())
            decoded_mask = decode_segmap(target.numpy())
            show_image_segmentation_sample(image, decoded_pred, decoded_mask)


@click.command()
@click.option(
    "--dataset_path", default=DATASET_BASE_PATH, help="Path to the Cityscapes dataset"
)
@click.option("--epochs", default=EPOCHS, help="Amount of epochs to train")
@click.option(
    "--batch_size",
    default=BATCH_SIZE,
    help="Batch size for each training step",
)
@click.option("--learning_rate", default=LEARNING_RATE, help="Learning rate")
@click.option("--use_test_set", default=True, help="If a test directory exists")
@click.option("--device", default="cuda", help="Device to train on")
@click.option(
    "--weights_folder", default=WEIGHTS_FOLDER, help="Folder to save weights in"
)
def main(
    dataset_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    use_test_set: bool,
    device: str,
    weights_folder: str,
):
    print(
        "Args: ",
        dataset_path,
        epochs,
        batch_size,
        learning_rate,
        device,
        weights_folder,
    )
    save_folder = Path(weights_folder)
    save_folder.mkdir(exist_ok=True)
    training_transforms = get_image_target_transform()
    val_test_transforms = get_val_test_transform()
    train_data = get_dataset(
        dataset_path, "train", transform_image_and_target=training_transforms
    )
    val_data = get_dataset(
        dataset_path,
        "val",
        transform_image_and_target=val_test_transforms,
    )
    test_data = (
        get_dataset(
            dataset_path,
            "test",
            transform_image_and_target=val_test_transforms,
        )
        if use_test_set
        else None
    )
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dl = (
        DataLoader(test_data, batch_size=batch_size, shuffle=False)
        if test_data
        else None
    )

    # model = UNetImpl(n_channels=3, n_classes=len(train_data.classes))
    num_classes = len(train_data.classes)
    model = get_unet_model(in_channels=3, out_channels=num_classes)
    model.to(device)
    # TODO: What should the softmax dim be?
    # loss_criterion = MulticlassDiceLoss(
    #     num_classes=len(train_data.classes), softmax_dim=1
    # )
    loss_criterion = DiceLoss(mode="multiclass")
    iou_meteric = IoU()

    def mean_iou_score(preds, targets):
        targets_one_hot = (
            torch.nn.functional.one_hot(targets.long(), num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        return iou_meteric(preds, targets_one_hot).mean()

    # TODO: Switch optimizer?
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # TODO: Use scheduler?
    print("Starting training")
    train_model(
        model,
        train_dl,
        val_dl,
        optimizer,
        loss_criterion,
        mean_iou_score,
        epochs,
        save_folder.as_posix(),
        device,
    )
    print("\n*** Finished training model ***\n")
    evaluate_model(
        model,
        test_dl if test_dl else val_dl,
        loss_criterion,
        mean_iou_score,
        "TEST" if test_dl else "TEST (val dataset)",
        device,
    )
    sample_image, target_mask = val_data[0]
    show_model_segmentation_sample(model, [(sample_image, target_mask)])


if __name__ == "__main__":
    main()
