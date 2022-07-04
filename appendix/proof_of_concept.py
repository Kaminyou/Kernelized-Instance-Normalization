import argparse
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from models.model import get_model
from torch.utils.data import DataLoader
from utils.dataset import XInferenceDataset
from utils.util import read_yaml_config, test_transforms


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calc_mean_std(feat, eps=1e-5):
    size = feat.shape
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(axis=2) + eps
    feat_std = np.sqrt(feat_var).flatten()
    feat_mean = feat.reshape(N, C, -1).mean(axis=2).flatten()
    return feat_mean, feat_std


def compute_img_distance(img_name_a, img_name_b):
    x_a, y_a = img_name_a.split("_")[:2]
    x_b, y_b = img_name_b.split("_")[:2]
    x_a = int(x_a)
    y_a = int(y_a)
    x_b = int(x_b)
    y_b = int(y_b)
    return ((x_a - x_b) ** 2 + (y_a - y_b) ** 2) ** (1 / 2)


def main():
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./data/example/config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./proof_of_concept/",
        help="Path to the output folder.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    config = read_yaml_config(args.config)
    model = get_model(
        config=config,
        model_name=config["MODEL_NAME"],
        normalization=config["INFERENCE_SETTING"]["NORMALIZATION"],
        isTrain=False,
    )

    test_dataset = XInferenceDataset(
        root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"],
        transform=test_transforms,
        return_anchor=True,
        thumbnail=config["INFERENCE_SETTING"]["THUMBNAIL"],
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])

    feature_map_dict = {}

    # calculate thumbnail features
    print("Extract thumbnail's features")
    X = test_dataset.get_thumbnail()
    Y_fake, feature_maps = model.analyze_feature_map(X)
    feature_map_dict["thumbnail"] = feature_maps

    # calculate others features
    print("Extract patches' features")
    for idx, data in enumerate(test_loader):
        print(f"Processing {idx}", end="\r")
        X, X_path = data["X_img"], data["X_path"]
        Y_fake, feature_maps = model.analyze_feature_map(X)
        feature_map_dict[Path(X_path[0]).stem] = feature_maps

    # precomputed
    print("Precomputing features' mean and var")
    thumbnail_feature_maps_precomputed = {}
    for feature_name, feature_map in feature_map_dict["thumbnail"].items():
        thumbnail_feature_map_mean, thumbnail_feature_map_std = calc_mean_std(
            feature_map
        )  # dim = C
        thumbnail_feature_maps_precomputed[
            f"{feature_name}_mean"
        ] = thumbnail_feature_map_mean
        thumbnail_feature_maps_precomputed[
            f"{feature_name}_std"
        ] = thumbnail_feature_map_std

    other_featuer_maps_precomputed = defaultdict(dict)
    for img_name, feature_maps in feature_map_dict.items():
        if img_name == "thumbnail":
            continue
        print(f"Processing ... {img_name}     ", end="\r")
        for feature_name, feature_map in feature_maps.items():
            other_feature_map_mean, other_feature_map_std = calc_mean_std(feature_map)
            other_featuer_maps_precomputed[img_name][
                f"{feature_name}_mean"
            ] = other_feature_map_mean
            other_featuer_maps_precomputed[img_name][
                f"{feature_name}_std"
            ] = other_feature_map_std

    # thumbnail with others
    print("Calculating thumbnail v.s. patches")
    cos_sim_dict = defaultdict(list)
    for img_name, featuer_maps_precomputed in other_featuer_maps_precomputed.items():
        print(f"Processing ... {img_name}     ", end="\r")
        for feature_name, feature_map_precomputed in featuer_maps_precomputed.items():
            cos_sim_value = cos_sim(
                thumbnail_feature_maps_precomputed[feature_name],
                feature_map_precomputed,
            )
            cos_sim_dict[feature_name].append(cos_sim_value)
    print("Drawing ...")
    for key1, key2 in zip(
        list(cos_sim_dict.keys())[0::2], list(cos_sim_dict.keys())[1::2]
    ):
        value1 = cos_sim_dict[key1]
        value2 = cos_sim_dict[key2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
        ax1.hist(value1, bins=100)
        ax1.set_title(key1)
        ax1.set_xlabel("Cosine similarity")
        ax1.set_ylabel("# of patches")

        ax2.hist(value2, bins=100, range=(-1, 1))
        ax2.set_title(key2)
        ax2.set_xlabel("Cosine similarity")
        ax2.set_ylabel("# of patches")
        if not "resnetblock" in key1:
            plt.savefig(
                os.path.join(args.output, f"thumbnail_with_patch_{key1.split('_')[0]}"),
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                os.path.join(
                    args.output, f"thumbnail_with_patch_{key1.rsplit('_', 2)[0]}"
                ),
                bbox_inches="tight",
            )
        plt.close("all")

    # others with others
    print("Calculating patches v.s. patches")
    cos_sim_dict = defaultdict(list)
    for i, (img_name_a, featuer_maps_precomputed_a) in enumerate(
        other_featuer_maps_precomputed.items()
    ):
        for j, (img_name_b, featuer_maps_precomputed_b) in enumerate(
            other_featuer_maps_precomputed.items()
        ):
            if i < j:
                print(f"Processing ... {img_name_a} {img_name_b}     ", end="\r")
                distance = compute_img_distance(img_name_a, img_name_b)
                for feature_name in featuer_maps_precomputed_a.keys():
                    cos_sim_value = cos_sim(
                        featuer_maps_precomputed_a[feature_name],
                        featuer_maps_precomputed_b[feature_name],
                    )
                    cos_sim_dict[feature_name].append([distance, cos_sim_value])

    print("Drawing ...")
    for key1, key2 in zip(
        list(cos_sim_dict.keys())[0::2], list(cos_sim_dict.keys())[1::2]
    ):
        value1 = np.array(cos_sim_dict[key1])
        value2 = np.array(cos_sim_dict[key2])

        # filter those are far away
        value1 = value1[value1[:, 0] <= 10]
        value2 = value2[value2[:, 0] <= 10]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))
        sns.boxplot(
            ax=ax1,
            x="distance",
            y="cos sim",
            data=pd.DataFrame(
                {
                    "distance": list(map(lambda x: round(x, 2), value1[:, 0])),
                    "cos sim": value1[:, 1],
                }
            ),
        )
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        # ax1.scatter(x=value1[:,0], y=value1[:,1], s=0.2)
        ax1.set_title(key1)
        ax1.set_xlabel("Distance")
        ax1.set_ylabel("Cosine similarity")

        sns.boxplot(
            ax=ax2,
            x="distance",
            y="cos sim",
            data=pd.DataFrame(
                {
                    "distance": list(map(lambda x: round(x, 2), value2[:, 0])),
                    "cos sim": value2[:, 1],
                }
            ),
        )
        # ax2.scatter(x=value2[:,0], y=value2[:,1], s=0.2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_title(key2)
        ax2.set_xlabel("Distance")
        ax2.set_ylabel("Cosine similarity")
        if not "resnetblock" in key1:
            plt.savefig(
                os.path.join(args.output, f"patch_with_patch_{key1.split('_')[0]}"),
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                os.path.join(args.output, f"patch_with_patch_{key1.rsplit('_', 2)[0]}"),
                bbox_inches="tight",
            )
        plt.close("all")

    # others with others (lineplot)
    colors = ["#4A5FAB", "#609F5C", "#E3C454", "#A27CBA", "#B85031"]
    ## block1 mean
    plt.figure(figsize=(8, 3.2))
    for key1 in list(cos_sim_dict.keys())[0::2]:
        if "block1" in key1:
            value1 = np.array(cos_sim_dict[key1])
            value1 = value1[value1[:, 0] <= 10]
            df = pd.DataFrame(
                {
                    "Distance (pixel)": value1[:, 0] * 512,
                    "Cosine similarity between μ(X)": value1[:, 1],
                }
            )
            sns.lineplot(
                data=df,
                x="Distance (pixel)",
                y="Cosine similarity between μ(X)",
                label="layer 1",
                color=colors[0],
            )
            break
    plt.grid()
    plt.savefig(
        os.path.join(
            args.output, f"patch_with_patch_lineplot_{key1.split('_')[0]}_mean"
        ),
        bbox_inches="tight",
    )
    plt.close("all")

    ## block1 std
    plt.figure(figsize=(8, 3.2))
    for key2 in list(cos_sim_dict.keys())[1::2]:
        if "block1" in key2:
            value2 = np.array(cos_sim_dict[key2])
            value2 = value2[value2[:, 0] <= 10]
            df = pd.DataFrame(
                {
                    "Distance (pixel)": value2[:, 0] * 512,
                    "Cosine similarity between σ(X)": value2[:, 1],
                }
            )
            sns.lineplot(
                data=df,
                x="Distance (pixel)",
                y="Cosine similarity between σ(X)",
                label="layer 1",
                color=colors[0],
            )
            break
    plt.grid()
    plt.savefig(
        os.path.join(
            args.output, f"patch_with_patch_lineplot_{key2.split('_')[0]}_std"
        ),
        bbox_inches="tight",
    )
    plt.close("all")

    ## other blocks mean
    plt.figure(figsize=(10, 4))
    idx = 1
    for key1 in list(cos_sim_dict.keys())[0::2]:
        if "resnetblock" in key1:
            continue
        if "block1" in key1:
            continue

        value1 = np.array(cos_sim_dict[key1])

        # filter those are far away
        value1 = value1[value1[:, 0] <= 10]
        df = pd.DataFrame(
            {
                "Distance (pixel)": value1[:, 0] * 512,
                "Cosine similarity between μ(X)": value1[:, 1],
            }
        )
        sns.lineplot(
            data=df,
            x="Distance (pixel)",
            y="Cosine similarity between μ(X)",
            label="layer " + key1.split("_")[0][-1],
            color=colors[idx],
        )
        idx += 1
    plt.grid()
    plt.savefig(
        os.path.join(args.output, f"patch_with_patch_lineplot_blocks_mean"),
        bbox_inches="tight",
    )
    plt.close("all")

    ## other blocks std
    plt.figure(figsize=(10, 4))
    idx = 1
    for key2 in list(cos_sim_dict.keys())[1::2]:
        if "resnetblock" in key2:
            continue
        if "block1" in key2:
            continue

        value2 = np.array(cos_sim_dict[key2])

        # filter those are far away
        value2 = value2[value2[:, 0] <= 10]
        df = pd.DataFrame(
            {
                "Distance (pixel)": value2[:, 0] * 512,
                "Cosine similarity between σ(X)": value2[:, 1],
            }
        )
        sns.lineplot(
            data=df,
            x="Distance (pixel)",
            y="Cosine similarity between σ(X)",
            label="layer " + key2.split("_")[0][-1],
            color=colors[idx],
        )
        idx += 1
    plt.grid()
    plt.savefig(
        os.path.join(args.output, f"patch_with_patch_lineplot_blocks_std"),
        bbox_inches="tight",
    )
    plt.close("all")


if __name__ == "__main__":
    main()
