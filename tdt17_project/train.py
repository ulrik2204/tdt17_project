from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple, Protocol

import click
import numpy as np
import torch.nn as nn
import torch.optim
from matplotlib import pyplot as plt
from segmentation_models_pytorch.losses import DiceLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from tdt17_project.data import (
    get_dataset,
    get_image_target_transform,
    get_val_test_transform,
)
from tdt17_project.metrics import classwise_iou_score, mean_iou_score
from tdt17_project.model import get_unet_model
from tdt17_project.utils import CityscapesContants, decode_segmap, encode_segmap

DATASET_BASE_PATH = "/cluster/projects/vc/data/ad/open/Cityscapes"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.005
WEIGHTS_FOLDER = "./weights"


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


class AfterEpochFn(Protocol):
    def __call__(
        self,
        model: nn.Module,
        loss_criterion: nn.Module,
        epoch: int,
        best_loss: torch.Tensor,
    ) -> str | None:
        ...


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    optimizer: Optimizer,
    loss_criterion: nn.Module,
    epochs: int,
    running_metric: Callable[[Any, Any], Any],
    after_epoch: AfterEpochFn | None = None,
    device: str = "cuda",
) -> str:
    model.train()
    best_loss = torch.Tensor([10000000.0]).to(device)
    best_model_name = ""
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
                [running_metric],
                device,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            total_metric += metric_scores[0]
            best_loss = loss if loss < best_loss else best_loss

            pbar.set_postfix_str(
                f"TEST: avg loss {total_loss/(index+1):.3f}, "
                + f"avg {running_metric.__name__}: {total_metric/(index+1):.3f}"
            )
        best_model_result = (
            after_epoch(model, loss_criterion, epoch, best_loss)
            if after_epoch
            else None
        )
        best_model_name = best_model_result if best_model_result else ""

        # scheduler.step() # (after epoch)
    return best_model_name


class EvalResult(NamedTuple):
    avg_loss: float
    avg_metric_scores: list[torch.Tensor]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_criterion: nn.Module,
    running_metric: Callable[[Any, Any], Any],
    eval_metrics: list[Callable[[Any, Any], Any]] | None = None,
    title: str = "VAL",
    device="cuda",
) -> EvalResult:
    model.eval()
    total_loss = 0
    metrics = [running_metric] + (eval_metrics or [])
    all_metric_totals = [torch.tensor(0) for _ in range(len(metrics))]
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
                + f"avg {running_metric.__name__}: {avg_running_metric_score:.3f}"
            )
    avg_metrics = [metric_total / len(dataloader) for metric_total in all_metric_totals]
    return EvalResult(float(total_loss / len(dataloader)), avg_metrics)


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
    examples: list[tuple[Any, Any]],
    loss_criterion: nn.Module,
    metrics: list[Callable[[Any, Any], Any]],
    device: str,
    model_name: str = "model",
):
    print("\n--- Model examples ---")
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
            print("Average mIoU per class per epoch:", metric_scores[0])
            avg_miou_scores_per_class = dict(
                zip(CityscapesContants.CLASS_NAMES, metric_scores[1].tolist())
            )
            print("mIoU per epoch per class:\n", avg_miou_scores_per_class)
            plot_image_mask_and_pred(
                image,
                decoded_target,
                decoded_pred,
                f"[loss: {loss:.3f}, {metrics[0].__name__}: {float(metric_scores[0]):.3f}]",
                f"{model_name}_sample_{i}.png",
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

    def miou_score(preds: torch.Tensor, targets: torch.Tensor):
        return mean_iou_score(preds, targets, num_classes=num_classes)

    def classwise_iou(preds: torch.Tensor, targets: torch.Tensor):
        return classwise_iou_score(preds, targets, num_classes=num_classes)

    def after_epoch(
        model: nn.Module, loss_criterion: nn.Module, epoch: int, best_loss: torch.Tensor
    ) -> str | None:
        avg_eval_loss, avg_eval_metric_scores = evaluate_model(
            model=model,
            dataloader=val_dl,
            loss_criterion=loss_criterion,
            running_metric=miou_score,
            eval_metrics=[classwise_iou],
            title="VAL",
            device=device,
        )
        # Show metric scores
        print("\n--- Eval metrics ---")
        print("Average mIoU per class per epoch:", avg_eval_metric_scores[0])
        avg_miou_scores_per_class = dict(
            zip(CityscapesContants.CLASS_NAMES, avg_eval_metric_scores[1].tolist())
        )
        print("mIoU per epoch per class:\n", avg_miou_scores_per_class)
        if epoch % 5 == 0 and avg_eval_loss < best_loss:
            time = datetime.now().strftime("%d%H%M")
            best_model_name = f"{time}_model.pt"
            model_path = (save_folder / best_model_name).as_posix()
            torch.save(model.state_dict(), model_path)
            return best_model_name
        return None

    # TODO: Switch optimizer?
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # TODO: Use scheduler?
    sample_image, target_mask = val_data[5]
    show_model_segmentation_sample(
        model,
        [(sample_image, target_mask)],
        loss_criterion,
        [miou_score, classwise_iou],
        device,
        model_name=resume_from_weights,
    )
    print("Starting training")

    best_model_name = train_model(
        model=model,
        train_dl=train_dl,
        optimizer=optimizer,
        loss_criterion=loss_criterion,
        epochs=epochs,
        running_metric=miou_score,
        after_epoch=after_epoch,
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
        running_metric=miou_score,
        eval_metrics=[classwise_iou],
        title="TEST" if test_dl else "TEST (val dataset)",
        device=device,
    )
    sample_image, target_mask = val_data[5]
    show_model_segmentation_sample(
        model,
        [(sample_image, target_mask)],
        loss_criterion,
        [miou_score, classwise_iou],
        device,
        best_model_name,
    )


if __name__ == "__main__":
    main()
