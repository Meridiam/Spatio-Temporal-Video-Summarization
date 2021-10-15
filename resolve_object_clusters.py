import argparse
import json
import os
import time
from loguru import logger

from sklearn.cluster import DBSCAN
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("Get Object Centers")
    parser.add_argument(
        "--input_path", default="", help="world transform tagged points pathname"
    )
    parser.add_argument(
        "--save_path", default="ObjectCluster_outputs", help="pathname for results folder"
    )
    parser.add_argument(
        "--eps", type=int, default=1, help="DBSCAN parameter: maximum distance between two samples in neighborhood"
    )
    parser.add_argument(
        "--min_samples", type=int, default=5, help="DBSCAN parameter: number of samples in neighborhood to form core point"
    )
    return parser


def parse_points(points_filename):
    with open(points_filename) as f:
        logger.info(f'Parsing 3D world points from {points_filename}')

        points = []
        for line in f:
            points.append(json.loads(line))

        return points


def main(args):
    if args.input_path == "":
        print(f'World transform tagged points path name cannot be blank!')
        exit(1)

    save_folder = os.path.join(
        args.save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )
    os.makedirs(save_folder, exist_ok=True)

    logger.info(f'Save location configured as {save_folder}')

    points = np.array(parse_points(args.input_path))

    x = [entry["worldCenter"] for entry in points]

    # fit clustering over candidate object points
    clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    clustering.fit(x)

    # remove outlier class
    cluster_ids = np.unique(clustering.labels_).tolist()
    if -1 in cluster_ids:
        cluster_ids.remove(-1)

    # organize clusters
    num_clusters = 0
    with open(os.path.join(save_folder, 'object_clusters'), 'w+') as f:
        for cluster_id in cluster_ids:
            mask = clustering.labels_ == cluster_id

            cluster_points = points[mask]

            # calculate majority class
            majority_class = cluster_points[0]["class"]
            majority_count = 0
            for point in cluster_points:
                if point["class"] == majority_class:
                    majority_count += 1
                else:
                    majority_count -= 1

                if majority_count == 0:
                    majority_class = point["class"]
                    majority_count = 1

            majority_points = []
            majority_count = 0
            representative_point = {
                "score": 0.0
            }
            for point in cluster_points:
                if point["class"] == majority_class:
                    majority_points.append(point)
                    majority_count += 1

                    if point["score"] > representative_point["score"]:
                        representative_point = point

            # if no majority class exists, cluster is too noisy so we skip
            if not majority_count > len(cluster_points) / 2:
                continue

            num_clusters += 1
            cluster = {
                "representative_point": representative_point,
                "majority_points": majority_points,
                "centroid": np.mean([v["worldCenter"] for v in majority_points], axis=0).tolist()
            }
            f.write(f'{json.dumps(cluster)}\n')

    logger.info(f'Found {num_clusters} class-coherent clusters')


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
