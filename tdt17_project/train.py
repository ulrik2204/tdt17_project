from torch.utils.data import DataLoader

from tdt17_project.data import get_dataset
from tdt17_project.model import SemanticSegmentationModel

DATASET_BASE_PATH = "/cluster/projects/vc/data/ad/open/Cityscapes/gtFine"
BATCH_SIZE = 32


def main():
    train, val, test = (
        get_dataset(DATASET_BASE_PATH, "train"),
        get_dataset(DATASET_BASE_PATH, "val"),
        get_dataset(DATASET_BASE_PATH, "test"),
    )
    train_dl, val_dl, test_dl = (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=False),
    )
    model = SemanticSegmentationModel()
    print("Starting training")
    ...


if __name__ == "__main__":
    main()
