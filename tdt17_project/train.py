from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple, Protocol, TypedDict, cast

import click
import numpy as np
import torch.nn as nn
import torch.optim
import wandb
from matplotlib import pyplot as plt
from segmentation_models_pytorch.losses import DiceLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from tdt17_project.data import (
    get_dataset,
    get_image_target_transform,
    get_val_test_transform,
)
from tdt17_project.model import get_unet_model
from tdt17_project.utils import CityscapesContants, decode_segmap, encode_segmap

DATASET_BASE_PATH = "/cluster/projects/vc/data/ad/open/Cityscapes"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHTS_FOLDER = "./weights"


class DisplayMetricFn(Protocol):
    def __call__(self, score: torch.Tensor) -> tuple[str, Any]:
        ...


class ProcessBatchResult(NamedTuple):
    loss: torch.Tensor
    metric_scores: list[torch.Tensor]
    pred: torch.Tensor
    target_prep: torch.Tensor


def process_batch(
    image: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module,
    loss_criterion: nn.Module,
    metrics: list[Callable[[Any, Any], torch.Tensor]],
    device: str = "cuda",
) -> ProcessBatchResult:
    image_prep = image.to(device).float()
    target_prep = encode_segmap(target.to(device).long())  # to remove unwanted classes
    pred = model(image_prep)
    loss = loss_criterion(pred, target_prep)
    metric_scores = [metric(pred, target_prep) for metric in metrics]
    # print("metric_scores", metric_scores)
    return ProcessBatchResult(loss, metric_scores, pred, target_prep)


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


def load_state_dict(model: nn.Module, optimizer: Optimizer, path: str):
    checkpoint: ModelCheckpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"] - 1


class AfterEpochFn(Protocol):
    def __call__(
        self,
        model: nn.Module,
        loss_criterion: nn.Module,
        epoch: int,
    ) -> str | None:
        ...


class Scheduler(Protocol):
    def step(self, metrics, epoch=None):
        ...


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    loss_criterion: nn.Module,
    epochs: int,
    metrics: list[Callable[[Any, Any], Any]],
    display_metrics_fns: list[DisplayMetricFn],
    save_folder: Path,
    device: str = "cuda",
) -> str:
    model.train()
    best_model_name = ""
    best_loss = 10000.0
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        total_loss = 0
        total_metric = 0
        for index, (image, target) in (
            pbar := tqdm(enumerate(train_dl), total=len(train_dl))
        ):
            loss, metric_scores, *_ = process_batch(
                image,
                target,
                model,
                loss_criterion,
                [metrics[0]],
                device,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            total_metric += metric_scores[0]
            metric_name, metric_score = display_metrics_fns[0](metric_scores[0])

            pbar.set_postfix_str(
                f"TRAIN: avg loss {total_loss/(index+1):.3f}, "
                + f"{metric_name}: {metric_score:.3f}"
            )
        avg_eval_loss, avg_eval_metric_scores = evaluate_model(
            model=model,
            dataloader=val_dl,
            loss_criterion=loss_criterion,
            metrics=metrics,
            first_metric_name="mIoU",
            title="VAL",
            device=device,
        )
        scheduler.step(avg_eval_loss)
        log_scores(
            avg_eval_loss,
            float(total_loss / len(train_dl)),
            avg_eval_metric_scores,
            display_metrics_fns,
            log_to_wandb=True,
        )
        if epoch % 3 == 0 and avg_eval_loss < best_loss:
            print("Saving!")
            best_loss = avg_eval_loss
            time = datetime.now().strftime("%d%H%M")
            best_model_name = f"{time}_model.pt"
            model_path = (save_folder / best_model_name).as_posix()
            save_state_dict(model, optimizer, epoch, model_path)

        # scheduler.step() # (after epoch)
    return best_model_name


class EvalResult(NamedTuple):
    avg_loss: float
    avg_metric_scores: list[torch.Tensor]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_criterion: nn.Module,
    metrics: list[Callable[[Any, Any], Any]],
    first_metric_name: str = "metric",
    title: str = "VAL",
    device="cuda",
) -> EvalResult:
    model.eval()
    total_loss = 0
    all_metric_totals: list[torch.Tensor | int] = [0 for _ in range(len(metrics))]
    with torch.no_grad():
        for index, (image, target) in (
            pbar := tqdm(enumerate(dataloader), total=len(dataloader))
        ):
            loss, metric_scores, *_ = process_batch(
                image,
                target,
                model,
                loss_criterion,
                metrics,
                device,
            )
            total_loss += loss.detach().cpu()
            for i, metric_score in enumerate(metric_scores):
                all_metric_totals[i] += metric_score.detach().cpu()
            avg_running_metric_score = all_metric_totals[0] / (index + 1)
            pbar.set_postfix_str(
                f"{title}: average loss {total_loss/(index+1):.3f}, "
                + f"avg {first_metric_name}: {avg_running_metric_score:.3f}"
            )
    avg_metrics = [
        cast(torch.Tensor, metric_total / len(dataloader))
        for metric_total in all_metric_totals
    ]
    return EvalResult(float(total_loss / len(dataloader)), avg_metrics)


def log_scores(
    val_loss: float,
    train_loss: float | None,
    metric_scores: list[torch.Tensor],
    display_metric_fns: list[DisplayMetricFn],
    log_to_wandb: bool = True,
):
    print("-- Loss and Metrics --")
    print("Val loss:", val_loss)
    log_dict = dict(val_loss=val_loss)
    if train_loss:
        print("Train loss:", train_loss)
        log_dict["train_loss"] = train_loss
    for metric_score, display_fn in zip(metric_scores, display_metric_fns):
        text, score = display_fn(metric_score)
        print(text, ": ", score)
        log_dict[text] = score
    if log_to_wandb:
        wandb.log(log_dict)


def plot_image_mask_and_pred(
    real_image, target_mask, pred_mask, score_str: str, image_name: str
):
    _, ax = plt.subplots(ncols=3, figsize=(16, 50), facecolor="white")
    ax[0].imshow(np.moveaxis(real_image.numpy(), 0, 2))
    ax[1].imshow(target_mask)
    ax[2].imshow(pred_mask)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[0].set_title("Input Image")
    ax[1].set_title("Ground mask")
    ax[2].set_title(f"Predicted mask, {score_str}")
    plt.savefig(image_name, bbox_inches="tight")


def show_model_segmentation_sample(
    model: nn.Module,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    loss_criterion: nn.Module,
    metrics: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    display_metric_fns: list[DisplayMetricFn],
    device: str = "cuda",
    model_name: str = "model",
):
    print("\n*** Model examples ***")
    model.eval()
    with torch.no_grad():
        for i, (image, target) in enumerate(examples):
            loss, metric_scores, pred, target_prep = process_batch(
                image.unsqueeze(0),
                target.unsqueeze(0),
                model,
                loss_criterion,
                metrics,
                device,
            )
            decoded_pred = decode_segmap(
                torch.argmax(pred.squeeze(0).detach().cpu(), dim=0)
            )
            decoded_target = decode_segmap(target_prep.squeeze(0).detach().cpu())
            log_scores(
                val_loss=float(loss),
                train_loss=None,
                metric_scores=metric_scores,
                display_metric_fns=display_metric_fns,
                log_to_wandb=False,
            )
            plot_image_mask_and_pred(
                image,
                decoded_target,
                decoded_pred,
                f"[loss: {loss:.3f}, {display_metric_fns[0](metric_scores[0])}",
                f"{model_name}_sample_{i}.png",
            )


def display_miou_score(score: torch.Tensor):
    return "mIoU", float(score)


def display_classwise_iou_score(score: torch.Tensor):
    return "classwise mIoU score", dict(
        zip(CityscapesContants.CLASS_NAMES, score.tolist())
    )


def display_weighted_iou_score(score: torch.Tensor):
    return "weighted mIoU score", float(score)


def init_wandb(epochs: int, batch_size: int, learning_rate: float):
    wandb_cache_path = Path("./.wandb")
    time = datetime.now().strftime("%m%d%H%M")
    wandb_cache_path.mkdir(parents=True, exist_ok=True)
    wandb.login()
    wandb.init(
        project="tdt17_project",
        name=f"Experiment {time}",
        dir=wandb_cache_path.as_posix(),
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    )


@click.command()
@click.option(
    "--dataset-path", default=DATASET_BASE_PATH, help="Path to the Cityscapes dataset"
)
@click.option("--epochs", default=EPOCHS, help="Amount of epochs to train")
@click.option(
    "--batch-size",
    default=BATCH_SIZE,
    help="Batch size for each training step",
)
@click.option("--learning-rate", default=LEARNING_RATE, help="Learning rate")
@click.option("--use-test-set", default=True, help="If a test directory exists")
@click.option("--device", default="cuda", help="Device to train on")
@click.option(
    "--weights-folder", default=WEIGHTS_FOLDER, help="Folder to save weights in"
)
@click.option(
    "--resume-from-weights",
    default=None,
    help="Path to weights to resume training on. If none, starts from scratch.",
)
def main(
    dataset_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    use_test_set: bool,
    device: str,
    weights_folder: str,
    resume_from_weights: str,
):
    print(
        f"""Args:
        dataset_path={dataset_path},
        epochs={epochs},
        batch_size={batch_size},
        learning_rate={learning_rate},
        use_test_set={use_test_set},
        device={device},
        weights_folder={weights_folder},
        resume_from_weights={resume_from_weights}
        """
    )
    init_wandb(epochs, batch_size, learning_rate)
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
    num_classes = len(CityscapesContants.VALID_CLASSES)
    model = get_unet_model(in_channels=3, out_channels=num_classes)
    model.to(device)
    if resume_from_weights:
        model.load_state_dict(torch.load(resume_from_weights))

    loss_criterion = DiceLoss(mode="multiclass")
    mean_iou_score_fn = MulticlassJaccardIndex(
        num_classes=num_classes, average="macro"
    ).to(device)
    classwise_iou_score_fn = MulticlassJaccardIndex(
        num_classes=num_classes, average="none"
    ).to(device)
    weighted_iou_score_fn = MulticlassJaccardIndex(
        num_classes=num_classes, average="weighted"
    ).to(device)

    metrics = [mean_iou_score_fn, classwise_iou_score_fn, weighted_iou_score_fn]

    display_metrics: list[DisplayMetricFn] = [
        display_miou_score,
        display_classwise_iou_score,
        display_weighted_iou_score,
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, verbose=True
    )
    show_model_segmentation_sample(
        model,
        [val_data[5]],
        loss_criterion,
        metrics,
        display_metric_fns=display_metrics,
        device=device,
        model_name=resume_from_weights,
    )
    print("Starting training")

    best_model_name = train_model(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_criterion=loss_criterion,
        epochs=epochs,
        metrics=metrics,
        display_metrics_fns=display_metrics,
        save_folder=save_folder,
        device=device,
    )
    print("\n*** Finished training model ***\n")
    model.load_state_dict(torch.load((save_folder / best_model_name).as_posix()))
    print("*** Loaded best model from training ***")
    print("\n *** Testing best model ***")
    evaluate_model(
        model=model,
        dataloader=test_dl if test_dl else val_dl,
        loss_criterion=loss_criterion,
        metrics=metrics,
        first_metric_name="mIoU",
        title="TEST" if test_dl else "TEST (val dataset)",
        device=device,
    )
    show_model_segmentation_sample(
        model,
        [val_data[5]],
        loss_criterion,
        metrics,
        display_metric_fns=display_metrics,
        device=device,
        model_name=best_model_name,
    )


if __name__ == "__main__":
    main()
