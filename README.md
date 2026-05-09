# knn

Image generation - diffusion models.

This project is implementation of model that can generate handwritten text
based on example of handwriting and the text to generate.

## Dataset

The dataset we used is available [here](
    https://drive.google.com/drive/folders/108TB-z2ytAZSIEzND94dyufybjpqVyn6
).

## Usage

Show help:
```sh
./main.py --help
```

Pretrain style model:
```sh
./main.py train-style [-d <path-to-dataset>] [-e <epoch-cnt>] \
    [-b <batch-size>] [-o <output-dir>]
```

Train the model:
```sh
./main.py train [-d <path-to-dataset>] [-e <epoch-cnt>] [-b <batch-size>] \
    [-m <model-to-continue-training>] [-o <output-dir>] \
    [--style-model <pretrained-style-model>]
```

Run the model on style image and text:
```sh
./main.py run -m <path-to-model> -t <text> -s <style-image> [-o <output-img>]
```

# Links

- [DiT](https://github.com/facebookresearch/DiT)

Inspiration:
- [One-DM](https://arxiv.org/pdf/2409.04004)
- [DiffusionPen](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11492.pdf)
