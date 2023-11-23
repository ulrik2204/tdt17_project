import torch
from segmentation_models_pytorch.utils.functional import iou
from torch.nn.functional import one_hot
from torchmetrics.classification import MulticlassJaccardIndex

# def classwise_iou_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
#     print("preds.size", preds.size())
#     print("target.size", targets.size())
#     targets_one_hot = one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).long()
#     print("targets_one_hot.size", targets_one_hot.size())
#     # output one_hot: [batch_size, height, width, num_classes]
#     # after permute: [batch_size, num_classes, height, width], which is what we want
#     # iou = IoU()
#     out = iou(preds, targets_one_hot, threshold=0.5)
#     print("out", out)
#     return out


def classwise_weighted_iou_score(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
):
    # Flatten the output and target tensors to make them compatible for the IoU calculation
    targets_one_hot = one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).long()
    iou_scores = torch.zeros(num_classes, dtype=torch.float32, device=preds.device)

    for class_idx in range(num_classes):
        # Calculate IoU for the current class
        iou_scores[class_idx] = iou(
            preds[:, class_idx, :, :],
            targets_one_hot[:, class_idx, :, :],
            threshold=0.5,
        )
    return iou_scores


def mean_iou_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    jaccard = MulticlassJaccardIndex(task="multiclass", num_classes=num_classes).to("cuda")
    return jaccard(preds, targets)
    # return classwise_weighted_iou_score(preds, targets, num_classes).mean()
