import itertools
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class IAMDataset(Dataset):
    def __init__(self, labels: Path, data: Path):
        """
        e.g. labels="dataset/IAM64_train.txt", data="dataset/IAM64-new/test"
        """
        self.writer_dict = parse_labels(labels)
        self.data = [
            d for list in self.writer_dict.values() for d in list
        ]
        self.data_folder = data
        self.generator = np.random.default_rng()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        writer, expected, txt = self.data[id]
        style = self.generator.choice(self.writer_dict[writer])[1]

        return {
            "style": prep_img(self.data_folder / style),
            "expected": wrapping_prep_img(self.data_folder / expected),
            "transcript": txt,
        }


def parse_labels(path: Path) -> dict[str, list[Tuple[str, str, str]]]:
    res = {}
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
            if writer not in res:
                res[writer] = []
            res[writer].append((writer, writer / image, word))
    return res


def prep_img_base(file, res_h=64):
    image = Image.open(file).convert("RGB")

    width, height = image.size
    if height != res_h:
        w = res_h * width // height
        image = image.resize((w, res_h))

    image = np.array(image).astype(np.float32)
    image /= 255.0
    return 1 - image


def prep_img(file, res_h=64):
    image = prep_img_base(file, res_h)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    return torch.tensor(image, dtype=torch.float)


def wrapping_prep_img(file, wrap_size=256, res_h=64):
    img = prep_img_base(file, res_h)

    res = np.zeros((wrap_size, wrap_size, img.shape[2]))

    splits = range(wrap_size, wrap_size // res_h * wrap_size, wrap_size)
    for i, t in enumerate(np.split(img, splits, axis=1)):
        h, w, _ = t.shape
        if w == 0:
            break
        res[i : i + h, 0:w, :] = t

    img = np.transpose(res, (2, 0, 1))

    return torch.tensor(img, dtype=torch.float)


def decode_img(img: torch.Tensor, height=64) -> torch.Tensor:
    image = np.transpose(img, (1, 2, 0)).astype(np.float32)
    h, w, c = image.shape
    res = np.zeros((height, h // height * w, c))
    for i, t in enumerate(np.split(img, h // height)):
        res[:, i : i + w, :] = t

    return torch.tensor(res, dtype=torch.float)


def collate_fn_padd(batch, device):
    style = [item["style"] for item in batch]
    expected = [item["expected"] for item in batch]
    transcript = [item["transcript"] for item in batch]

    widths = [img.shape[2] for img in style]
    max_width = max(widths)

    batch_size = len(style)
    channels = style[0].shape[0]
    height = style[0].shape[1]

    padded_imgs = torch.zeros(batch_size, channels, height, max_width)

    for i, img in enumerate(style):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img
    padded_imgs = padded_imgs

    targets = torch.stack(expected)

    return {
        "style": padded_imgs.to(device),
        "expected": targets.to(device),
        "transcript": transcript,
    }
