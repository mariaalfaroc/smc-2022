import os
from typing import List, Dict, Tuple

INPUT_EXTENSION = {"omr": "_distorted.jpg", "amt": ".wav"}
LABEL_EXTENSION = ".semantic"


# Get all the folds filenames for each data partition
# folds = {"train": [".../train_gt_fold0.dat", ".../train_gt_fold1.dat", ...], "val": [...], "test": [...]}
def get_folds_filenames() -> Dict[str, List[str]]:
    folds_dir = f"dataset/folds"

    folds = {"train": [], "val": [], "test": []}
    for fname in os.listdir(folds_dir):
        if fname.startswith("train"):
            folds["train"].append(os.path.join(folds_dir, fname))
        elif fname.startswith("val"):
            folds["val"].append(os.path.join(folds_dir, fname))
        elif fname.startswith("test"):
            folds["test"].append(os.path.join(folds_dir, fname))

    assert (
        len(folds["train"]) == len(folds["val"]) == len(folds["test"])
    ), "Folds are not balanced!"

    return {k: sorted(v) for k, v in folds.items()}


# Get all images and labels filenames
# of a corresponding fold filename
def get_datafold_filenames(
    task: str, fold_filename: list
) -> Tuple[List[str], List[str]]:
    images_filenames = []
    labels_filenames = []
    with open(fold_filename) as f:
        lines = f.read().splitlines()
    for line in lines:
        common_path = f"dataset/Corpus/{line}/{line}"
        images_filenames.append(common_path + INPUT_EXTENSION[task])
        labels_filenames.append(common_path + LABEL_EXTENSION)
    return images_filenames, labels_filenames
