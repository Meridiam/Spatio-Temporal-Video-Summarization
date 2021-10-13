import json
import os

import matplotlib.pyplot as plt
import argparse

import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("Visualize Object Centers")
    parser.add_argument(
        "--input_path", default="", help="resolved object clusters pathname"
    )
    return parser

def main(args):
    if args.input_path == "":
        print(f'Resolved object clusters path name cannot be blank!')
        exit(1)

    fig = plt.figure()
    ax = fig.add_subplot()

    classes = []
    with open(args.input_path, 'r') as f:
        for line in f:
            classes.append(json.loads(line)["representative_point"]["class"])

    classes = np.unique(classes).tolist()

    # visualize clusters
    with open(args.input_path, 'r') as f:
        cluster_idx = 0
        cmap = plt.cm.get_cmap('hsv', len(classes))
        for line in f:
            cluster_data = json.loads(line)

            obj = cluster_data["representative_point"]

            # for point in cluster_data["majority_points"]:
            ax.scatter(obj["worldCenter"][0], obj["worldCenter"][2], c=cmap(classes.index(obj["class"])))
            ax.annotate(f'{obj["class"]}', (obj["worldCenter"][0], obj["worldCenter"][2]))

            cluster_idx += 1

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
