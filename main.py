#!/usr/bin/python

import argparse
from pathlib import Path

from torchvision.utils import save_image

from src.loader import decode_img, prep_img
from src.sample import Sampler, sample
from src.train import Trainer
from src.train_style import StyleTrainer


def main():
    parser = argparse.ArgumentParser(
        prog="knn", description="Handwritten text generating"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Choose a mode to run"
    )

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "-d",
        "--dataset",
        default=Path("dataset"),
        type=Path,
        help="Path to the dataset directory",
    )
    train_parser.add_argument(
        "-e", "--epochs", default=10, type=int, help="Number of training epochs"
    )
    train_parser.add_argument(
        "-b", "--batch", default=32, type=int, help="Batch size used for training"
    )
    train_parser.add_argument(
        "-o",
        "--output",
        default=Path("trained"),
        type=Path,
        help="directory with resulting trained models",
    )

    run_parser = subparsers.add_parser("run", help="Runs the model")
    run_parser.add_argument(
        "-m", "--model", type=Path, help="Path to the trained .pt model", required=True
    )
    run_parser.add_argument(
        "-s",
        "--style",
        type=Path,
        help="Path to the style reference image",
        required=True,
    )
    run_parser.add_argument(
        "-t", "--text", type=str, help="The text to generate", required=True
    )
    run_parser.add_argument(
        "-o",
        "--output",
        default=Path("output.png"),
        type=Path,
        help="Path to the output image",
    )

    test_parser = subparsers.add_parser("test", help="Tests the model")
    test_parser.add_argument(
        "-d",
        "--dataset",
        default=Path("dataset"),
        type=Path,
        help="Path to the dataset directory",
    )
    test_parser.add_argument(
        "-m", "--model", type=Path, help="Path to the trained .pt model", required=True
    )
    test_parser.add_argument(
        "--style-model",
        type=Path,
        help="Path to the trained .pt model",
        required=False,
    )
    test_parser.add_argument(
        "-b", "--batch", default=32, type=int, help="Batch size used for training"
    )
    test_parser.add_argument(
        "-o",
        "--output",
        default=Path("output.png"),
        type=Path,
        help="Path to the output image",
    )

    train_style_parser = subparsers.add_parser(
        "train-style", help="Train the style model"
    )
    train_style_parser.add_argument(
        "-d",
        "--dataset",
        default=Path("dataset"),
        type=Path,
        help="Path to the dataset directory",
    )
    train_style_parser.add_argument(
        "-e", "--epochs", default=10, type=int, help="Number of training epochs"
    )
    train_style_parser.add_argument(
        "-b", "--batch", default=32, type=int, help="Batch size used for training"
    )
    train_style_parser.add_argument(
        "-o",
        "--output",
        default=Path("trained-style"),
        type=Path,
        help="directory with resulting trained models",
    )

    args = parser.parse_args()
    if args.command == "train":
        trainer = Trainer(args)
        trainer.train()
        trainer.save("last.pt")
    elif args.command == "run":
        sampler = Sampler(args)
        style = prep_img(args.style)
        res = decode_img(sampler.sample(style, args.text))
        save_image(res, args.output, nrow=4, normalize=True, value_range=(-1, 1))
    elif args.command == "test":
        sampler = Sampler(args)
        diff, cer, fid = sampler.eval()
        print(f"Style diff: {diff}, OCR CER: {cer}, FID: {fid}")
    elif args.command == "train-style":
        trainer = StyleTrainer(args)
        trainer.train()
        trainer.save("last.pt")


if __name__ == "__main__":
    main()
