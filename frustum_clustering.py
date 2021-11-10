import argparse
import json
import os
import time

import numpy as np
from loguru import logger
from scipy.stats import mode
from sklearn.cluster import DBSCAN

from utils import parse_framewise_camera_pose, parse_intrinsics, world_to_image_space

#  python .\frustum_clustering.py --detections_path .\YOLOX_outputs\yolox_l\vis_res\newLivingRoomDataset_WithPointCloud --pcloud_path .\DepthMap_outputs\NewLivingRoomDepthMaps_NoPointBlowups\denoised_point_cloud --framewise_path .\newLivingRoomDataset_WithPointCloud\framewiseData.txt --intrinsics_path .\newLivingRoomDataset_WithPointCloud\intrinsics.txt

def make_parser():
    parser = argparse.ArgumentParser("Frustum-Aware Clustering")
    parser.add_argument(
        "--detections_path", default="", help="yolox detections folder pathname"
    )
    parser.add_argument(
        "--pcloud_path", default="", help="point cloud file pathname"
    )
    parser.add_argument(
        "--framewise_path", default="", help="framewise camera pose file pathname"
    )
    parser.add_argument(
        "--intrinsics_path", default="", help="camera intrinsics file pathname"
    )
    parser.add_argument(
        "--save_path", default="FrustumClustering_outputs", help="pathname for results folder"
    )
    return parser


def cloud_to_image_space(points, camera_pose, fx, fy, cx, cy):
    projected_points = dict()
    for point in points:
        nd_point = np.array([float(point[0]), float(point[1]), float(point[2])])
        img_x, img_y, depth_from_camera = world_to_image_space(camera_pose, nd_point, fx, fy, cx, cy)

        projected_points[f'{img_x},{img_y}'] = {
            "point": nd_point,
            "local_depth": depth_from_camera
        }

    return projected_points


def load_point_cloud(path):
    with open(path) as file:
        return json.load(file)


def get_id(point):
    return f'{point[0]},{point[1]},{point[2]}'


def get_points_in_frustum(detection, projected_cloud):
    bbox_points = dict()
    for image_pos_string, point in projected_cloud.items():
        toks = image_pos_string.split(",")
        image_pos = [float(toks[0]), float(toks[1])]

        if detection["bbox_p0"][0] <= image_pos[0] <= detection["bbox_p1"][0] and \
                detection["bbox_p0"][1] <= image_pos[1] <= detection["bbox_p1"][1]:
            bbox_points[get_id(point["point"])] = point

    return bbox_points


def get_median_depth(detection_points):
    return np.median(np.array([point["local_depth"] for _, point in detection_points.items()]))


def augment_detections_with_frustum_points(detections, projected_cloud):
    augmented_detections = []
    for detection in detections:
        augmented_detections.append({
            "detection": detection,
            "frustum_points": get_points_in_frustum(detection, projected_cloud)
        })

    return augmented_detections


def main(args):
    if args.detections_path == "":
        print(f'Detections path name cannot be blank!')
        exit(1)

    if args.pcloud_path == "":
        print(f'Point cloud path name cannot be blank!')
        exit(1)

    if args.framewise_path == "":
        print(f'Framewise camera pose path name cannot be blank!')
        exit(1)

    if args.intrinsics_path == "":
        print(f'Camera intrinsics path name cannot be blank!')
        exit(1)

    save_folder = os.path.join(
        args.save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )
    os.makedirs(save_folder, exist_ok=True)

    logger.info(f'Save location configured as {save_folder}')

    fx_d, fy_d, cx_d, cy_d = parse_intrinsics(args.intrinsics_path)

    camera_poses = parse_framewise_camera_pose(args.framewise_path)

    raw_points = load_point_cloud(args.pcloud_path)

    cluster_classes = dict()
    labelled_points = dict()
    detections_pathname = os.path.join(args.detections_path, 'detections')
    num_entries = sum(1 for _ in open(detections_pathname))
    next_cluster_id = 0
    with open(detections_pathname) as datafile:
        logger.info(f'Ingesting frame-level detections from {detections_pathname}')

        entry_idx = 0
        for line in datafile:
            removed_points = set()
            frame_detections = json.loads(line)
            camera_pose = camera_poses[int(frame_detections["frame_idx"])]
            assert int(frame_detections["frame_idx"]) == int(camera_pose["frame_idx"])

            t0 = time.time()

            projected_points = cloud_to_image_space(raw_points, camera_pose, fx_d, fy_d, cx_d, cy_d)
            detections = augment_detections_with_frustum_points(frame_detections["data"], projected_points)

            # sort detections in order of median depth
            detections.sort(key=lambda e: get_median_depth(e["frustum_points"]))

            for detection in detections:
                points = np.array([point["point"] for point_string, point in detection["frustum_points"].items() if
                                   point_string not in removed_points])

                clustering = DBSCAN(eps=0.1, min_samples=30)
                clustering.fit(points)

                # remove outlier class
                cluster_ids = clustering.labels_.copy().tolist()
                if -1 in cluster_ids:
                    cluster_ids.remove(-1)

                # find label of largest cluster
                largest_cluster = mode(cluster_ids).mode[0]

                # select points which fall into the cluster
                mask = clustering.labels_ == largest_cluster
                cluster_points = points[mask]

                # remove points and find other clusters if any that the points belong to
                prev_clusters = []
                for point in cluster_points:
                    point_id = get_id(point)
                    removed_points.add(point_id)

                    if point_id in labelled_points:
                        prev_clusters.append(labelled_points[point_id])

                # if no point in the cluster is part of another object, or if supercluster does not have matching
                #   object class, we assign them to a new cluster
                if len(prev_clusters) == 0 or cluster_classes[mode(prev_clusters).mode[0]] != detection["detection"]["class"]:
                    no_points_labelled = True
                    for point in cluster_points:
                        point_id = get_id(point)
                        if point_id not in labelled_points:
                            no_points_labelled = False
                            labelled_points[point_id] = next_cluster_id

                    if not no_points_labelled:
                        cluster_classes[next_cluster_id] = detection["detection"]["class"]
                        next_cluster_id += 1

                else:
                    # if any points in the cluster are part of another object, don't create a new cluster and assign
                    #   all unlabelled points to supercluster instead.
                    supercluster_id = mode(prev_clusters).mode[0]
                    for point in cluster_points:
                        point_id = get_id(point)
                        if point_id not in labelled_points:
                            labelled_points[point_id] = supercluster_id

            entry_idx += 1
            logger.info('Processed entry {} / {} in {:.4f}s'.format(entry_idx, num_entries, time.time() - t0))
            logger.info(cluster_classes)

    with open(os.path.join(save_folder, 'cluster_classes'), 'w+') as f:
        f.write(f'{json.dumps(cluster_classes)}')

    with open(os.path.join(save_folder, 'labelled_points'), 'w+') as f:
        f.write(f'{json.dumps(labelled_points)}')


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
