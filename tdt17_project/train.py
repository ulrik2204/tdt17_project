from datetime import datetime
from pathlib import Path

import click
import torch.nn as nn
import torch.optim
from segmentation_models_pytorch.metrics import get_stats, iou_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from tdt17_project.data import get_dataset, get_image_target_transform
from tdt17_project.loss import get_dice_loss
from tdt17_project.model import get_unet_model

DATASET_BASE_PATH = "/cluster/projects/vc/data/ad/open/Cityscapes"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.005
WEIGHTS_FOLDER = "./weights"


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    optimizer: Optimizer,
    loss_criterion: nn.Module,
    epochs: int,
    device="cuda",
):
    model.train()
    best_loss = 10000000
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        total_loss = 0
        total_iou = 0
        for index, (image, target) in (
            pbar := tqdm(enumerate(train_dl), total=len(train_dl))
        ):
            image, target = image.to(device).float(), target.to(device).long()
            pred = model(image)
            loss = loss_criterion(pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            total_iou += iou_per_class(pred, target).detach().cpu()

            pbar.set_postfix_str(
                f"TEST: average loss {total_loss/(index+1):.3f}, mIoU: {total_iou/(index+1):.3f}"
            )
        validation_loss = evaluate_model(model, val_dl, loss_criterion, "VAL", device)
        if validation_loss < best_loss:
            best_loss = validation_loss
            time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
            torch.save(model.state_dict(), f"{time}_{validation_loss:.0f}_model.pt")
        # scheduler.step() # (after epoch)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_criterion: nn.Module,
    title: str,
    device="cuda",
):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for index, (image, target) in (
            pbar := tqdm(enumerate(dataloader), total=len(dataloader))
        ):
            image, target = image.to(device), target.to(device)
            pred = model(image)
            loss = loss_criterion(pred, target)
            total_loss += loss.detach().cpu()
            total_iou += iou_per_class(pred, target).detach().cpu()

            pbar.set_postfix_str(
                f"{title}: average loss {total_loss/(index+1):.3f}, mIoU: {total_iou/(index+1):.3f}"
            )
    return total_loss / len(dataloader)


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
    Path(weights_folder).mkdir(exist_ok=True)
    transform_image_and_target = get_image_target_transform()
    train_data = get_dataset(
        dataset_path, "train", transform_image_and_target=transform_image_and_target
    )
    val_data = get_dataset(
        dataset_path,
        "val",
        transform_image=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )
    test_data = (
        get_dataset(
            dataset_path,
            "test",
            transform_image=transforms.ToTensor(),
            target_transform=transforms.ToTensor(),
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
    model = get_unet_model(in_channels=3, out_channels=len(train_data.classes))
    model.to(device)
    # TODO: What should the softmax dim be?
    # loss_criterion = MulticlassDiceLoss(
    #     num_classes=len(train_data.classes), softmax_dim=1
    # )
    loss_criterion = get_dice_loss()
    # TODO: Switch optimizer?
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # TODO: Use scheduler?
    print("Starting training")
    train_model(model, train_dl, val_dl, optimizer, loss_criterion, epochs, device)
    print("\n*** Finished training model ***\n")
    evaluate_model(
        model,
        test_dl if test_dl else val_dl,
        loss_criterion,
        "TEST" if test_dl else "TEST (val dataset)",
        device,
    )


if __name__ == "__main__":
    main()


def iou_per_class(pred: torch.FloatTensor, target: torch.LongTensor):
    tp, fp, fn, tn = get_stats(pred, target, mode="multilabel", threshold=0.5)
    # Then compute IoU
    return iou_score(tp, fp, fn, tn, reduction="micro")
