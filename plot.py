#!/usr/bin/python

import sys

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: ./plot.py <path_to_log>")
        sys.exit(1)

    filename = sys.argv[1]
    losses = []

    with open(filename, "r") as file:
        for line in file:
            losses.append(float(line.strip()))

    epochs = list(range(1, len(losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")

    plt.savefig("plot.png")


if __name__ == "__main__":
    main()
