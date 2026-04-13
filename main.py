#!/usr/bin/python

import argparse
from pathlib import Path

from src.sample import sample
from src.train import Trainer


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

    args = parser.parse_args()
    if args.command == "train":
        trainer = Trainer(args)
        trainer.train()
        trainer.save("last.pt")
    elif args.command == "run":
        sample(args)


if __name__ == "__main__":
    main()
