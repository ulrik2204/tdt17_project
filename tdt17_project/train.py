import click
import torch.nn as nn
import torch.optim
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from tdt17_project.data import get_dataset
from tdt17_project.loss import MulticlassDiceLoss
from tdt17_project.model import UNet

DATASET_BASE_PATH = "/cluster/projects/vc/data/ad/open/Cityscapes/gtFine"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.005


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
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        total_loss = 0
        total_iou = 0
        for index, (image, target) in (
            pbar := tqdm(enumerate(train_dl), total=len(train_dl))
        ):
            print("image", image)
            print("target", target)
            plt.imshow(image)
            image, target = image.to(device), target.to(device)
            pred = model(image)
            loss = loss_criterion(pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            total_iou += box_iou(pred, target)

            pbar.set_postfix_str(
                f"TEST: average loss {total_loss/(index+1):.3f}, average acc {total_iou/(index+1):.3f}"
            )
        evaluate_model(model, val_dl, loss_criterion, "VAL", device)
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
            total_iou += box_iou(pred, target)

            pbar.set_postfix_str(
                f"{title}: average loss {total_loss/(index+1):.3f}, mIoU: {total_iou/(index+1):.3f}"
            )


@click.command()
@click.argument(
    "--dataset_path",
    prompt="Path to the Cityscapes dataset",
    type=str,
    default=DATASET_BASE_PATH,
)
@click.argument(
    "--epochs",
    prompt="Amount of epochs to train",
    type=int,
    default=EPOCHS,
)
@click.argument(
    "--batch_size",
    prompt="Batch size for each training step",
    type=str,
    default=BATCH_SIZE,
)
@click.argument(
    "--learning_rate", prompt="Learning rate", type=float, default=LEARNING_RATE
)
@click.argument("--device", prompt="Device to train on", type=str, default="cuda")
def main(
    dataset_path: str, epochs: int, batch_size: int, learning_rate: float, device: str
):
    # TODO: Add transforms
    train, val, test = (
        get_dataset(dataset_path, "train"),
        get_dataset(dataset_path, "val"),
        get_dataset(dataset_path, "test"),
    )
    train_dl, val_dl, test_dl = (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
    model = UNet(3, 3)
    # TODO: What should the softmax dim be?
    loss_criterion = MulticlassDiceLoss(num_classes=len(train.classes), softmax_dim=1)
    # TODO: Switch optimizer?
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # TODO: Use scheduler?
    print("Starting training")
    train_model(model, train_dl, val_dl, optimizer, loss_criterion, epochs, device)
    print("\n*** Finished training model ***\n")
    evaluate_model(model, test_dl, loss_criterion, "TEST", device)


if __name__ == "__main__":
    main()
    main()
