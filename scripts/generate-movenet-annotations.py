import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
import random
from typing import Sequence

import numpy as np
from coco_lib.keypointdetection import KeypointDetectionDataset


def save_split(source: KeypointDetectionDataset, ids: Sequence[int],
               data_path: Path, output_path: Path,
               annotation_name: str, data_dir_name: str):
    dataset = deepcopy(source)
    dataset.images = []
    dataset.annotations = []

    output_data_path = output_path.joinpath(data_dir_name)
    os.makedirs(str(output_data_path), exist_ok=True)

    for i in ids:
        image_desc = source.images[i]
        dataset.images.append(image_desc)
        dataset.annotations.append(source.annotations[i])

        shutil.copy(str(data_path.joinpath(image_desc.file_name)), str(output_data_path.joinpath(image_desc.file_name)))

    dataset.save(output_path.joinpath("annotations").joinpath(annotation_name))


def convert(input_path: Path, output_path):
    print(f"processing {input_path.name}...")

    dataset: KeypointDetectionDataset = KeypointDetectionDataset.load(input_path)
    data = []

    for image, annotation in zip(dataset.images, dataset.annotations):
        w, h = image.width, image.height

        keypoints = np.vstack(np.split(np.array(annotation.keypoints, dtype=float), len(annotation.keypoints) // 3))
        keypoints[:, 0] /= w
        keypoints[:, 1] /= h

        cx = np.average(keypoints[:, 0])
        cy = np.average(keypoints[:, 1])

        bbox = np.vstack(np.split(np.array(annotation.bbox, dtype=float), len(annotation.bbox) // 2))
        bbox[:, 0] /= w
        bbox[:, 1] /= h

        other_keypoints = [[] for i in range(keypoints.shape[0])]

        data.append({
            "img_name": image.file_name,
            "keypoints": keypoints.flatten().tolist(),
            # z: 0 for no label, 1 for labeled but invisible, 2 for labeled and visible
            "center": [cx, cy],
            "bbox": bbox.flatten().tolist(),
            "other_centers": [],
            "other_keypoints": other_keypoints,  # lenth = num_keypoints
        })

    with open(str(output_path), "w") as file:
        json.dump(data, file)


def main():
    args = parse_args()

    output_path = Path(args.output)
    os.makedirs(str(output_path), exist_ok=True)

    for file_path in args.inputs:
        path = Path(file_path)
        output_path = path.parent.joinpath(f"{args.prefix}{path.name}")

        convert(path, output_path)

    print("done")


def parse_args():
    parser = argparse.ArgumentParser(prog="MoveNet Generate Dataset", description="Dataset without crop.")

    default_inputs = ["./data/annotations/person_keypoints_train2017.json",
                      "./data/annotations/person_keypoints_val2017.json"]

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--inputs", type=str, default=default_inputs, nargs="+", help="Input files to convert.")
    dataset_group.add_argument("--output", type=str, default="./data/annotations", help="Output path.")
    dataset_group.add_argument("--prefix", type=str, default="movenet_", help="Prefix of the new files.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
