from typing import Self

import numpy as np

# Taken from https://github.com/talhaanwarch/youtube-tutorials/blob/main/cityscape-tutorial.ipynb


class Constant:
    def __new__(cls) -> Self:
        raise ValueError("Cannot instantiate this class")


class CityscapesContants(Constant):
    IGNORE_INDEX = 255
    VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    VALID_CLASSES = [
        IGNORE_INDEX,
        7,
        8,
        11,
        12,
        13,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        31,
        32,
        33,
    ]
    CLASS_NAMES = [
        "unlabelled",
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]
    CLASS_MAP = dict(
        zip(
            VALID_CLASSES,
            range(len(VALID_CLASSES)),
        )
    )
    COLORS = [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    LABEL_COLORS = dict(zip(range(len(VALID_CLASSES)), COLORS))


def encode_segmap(mask):
    # remove unwanted classes and recitify the labels of wanted classes
    for void_class in CityscapesContants.VOID_CLASSES:
        mask[mask == void_class] = CityscapesContants.IGNORE_INDEX
    for valid_class in CityscapesContants.VALID_CLASSES:
        mask[mask == valid_class] = CityscapesContants.CLASS_MAP[valid_class]
    return mask


def decode_segmap(segmap):
    # convert gray scale to color
    temp = segmap.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    n_classes = len(CityscapesContants.VALID_CLASSES)
    for label in range(0, n_classes):
        r[temp == label] = CityscapesContants.LABEL_COLORS[label][0]
        g[temp == label] = CityscapesContants.LABEL_COLORS[label][1]
        b[temp == label] = CityscapesContants.LABEL_COLORS[label][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
