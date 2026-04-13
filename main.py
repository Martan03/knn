#!/usr/bin/python

import argparse
from pathlib import Path

from src.train import Trainer


def main():
    print("Hello knn")
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
        "-b", "--batch", default=64, type=int, help="Batch size used for training"
    )

    _run_parser = subparsers.add_parser("run", help="Runs the model")

    args = parser.parse_args()
    if args.command == "train":
        print("training")
        trainer = Trainer(args)
        trainer.train()
        trainer.save("last.pt")
    elif args.command == "run":
        print("Not implemented...")


if __name__ == "__main__":
    main()
