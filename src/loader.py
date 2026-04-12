from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ContentDataset(Dataset):
    def __init__(self, labels: Path, data: Path):
        """
        e.g. labels="dataset/IAM64_train.txt", data="dataset/IAM64-new/test"
        """
        self.data = parse_labels(labels)
        self.data_folder = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        ref, txt = self.data[item]

        return {
            "ref": wrapping_prep_img(self.data_folder / ref),
            "transcript": txt,
        }

class StyleDataset(Dataset):
    def __init__(self, labels: Path, data: Path):
        """
        e.g. labels="dataset/IAM64_train.txt", data="dataset/IAM64-new/test"
        """
        self.data = parse_labels(labels)
        self.data_folder = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        ref, txt = self.data[item]

        return {
            "ref": prep_img(self.data_folder / ref),
            "transcript": txt,
        }


def parse_labels(path: Path) -> list[Tuple[str, str]]:
    res = []
    with open(path) as f:
        for line in f.readlines():
            spl = line.split()
            if len(spl) < 2:
                continue
            left = spl[0]
            word = " ".join(spl[1:])
            spl = left.split(",")
            writer = Path(spl[0])
            image = spl[1] + ".png"
            res.append((writer / image, word))
    return res

def prep_img_base(file, res_h=64):
    image = Image.open(file).convert("RGB")

    if image._size[1] != res_h:
        w = res_h * image._size[0] // image._size[1]
        image = image.resize((w, res_h))

    image = np.array(image)
    image /= 255.0
    return 1 - image

def prep_img(file, res_h=64):
    image = prep_img_base(file, res_h)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    return torch.tensor(image, dtype=torch.float)


def wrapping_prep_img(file, wrap_size=256, res_h=64):
    img = prep_img_base(file, res_h)

    res = np.zeros((wrap_size, wrap_size, img.shape[2]))

    for (i, t) in enumerate(np.split(img, wrap_size)):
        w, h, _ = t.shape
        res[0:w, i:i+h, :] = t

    img = np.transpose(img, (2, 0, 1))

    return torch.tensor(img, dtype=torch.float)
