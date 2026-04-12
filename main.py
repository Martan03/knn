import argparse


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
        default="dataset",
        type=str,
        help="Path to the dataset directory",
    )

    _run_parser = subparsers.add_parser("run", help="Runs the model")

    args = parser.parse_args()
    if args.command == "train":
        pass
    elif args.command == "run":
        print("Not implemented...")


if __name__ == "__main__":
    main()
