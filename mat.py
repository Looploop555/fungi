import os
import pickle

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile

from process_data import load_csv

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_feature(feature_path=None):
    catgory_features = load_pkl(feature_path)
    return catgory_features


def load_csv_gt(file_path):
    head, data = load_csv(file_path)
    target = {}
    for line in data:
        category_id = line[head.index("category_id")]
        observation_id = line[head.index("observationID")]

        target.setdefault(category_id, []).append(observation_id)
    return target


def save_csv(result, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    line = ["observationId,predictions"]
    for key, value in result.items():
        line.append(f"{key},{' '.join(value)}")
    with open(save_path, "w") as f:
        f.write("\n".join(line))


def mattching(pred_features, catgory_features, topk=10):
    result = {}
    for name, feature in tqdm(pred_features.items(), desc="matching"):
        scores = []
        for key, value in catgory_features.items():
            scores.append([key, np.mean(np.dot(np.array(feature), np.array(value).T))])
        if topk is not None:
            scores.sort(key=lambda x: x[1], reverse=True)
            # print("%20s: %s" % (name, str(scores[:50])))
            result[name] = [item[0] for item in scores[:topk]]
        else:
            result[name] = scores
    return result


def merge_score(results, topk=10):
    result = {}
    for res in results:
        for key, value in res.items():
            if result.get(key) is None:
                result[key] = value
            else:
                result[key][1] = result[key][1] + value[1]
    for name in result.keys():
        value = sorted(result[name], key=lambda x: x[1], reverse=True)
        print("%20s: %s" % (name, str(value[:50])))
        result[name] = [item[0] for item in value[:topk]]
    return result


def eval(csv_path, result, topk=10):
    gt = load_csv_gt(csv_path)
    target = np.zeros(topk)
    valid_samples = 0

    for key, value in result.items():
        if key in gt:
            obs_ids = gt[key]
            valid_samples += len(obs_ids)

            if key in value:
                idx = value.index(key)
                if idx < topk:
                    target[idx:] += len(obs_ids)

    for i in range(topk):
        acc = target[i] / valid_samples if valid_samples > 0 else 0
        print(f"top-{i + 1}: {int(target[i])}, acc: {acc:.4f}")


if __name__ == "__main__":
    catgory_features = get_feature(feature_path="/data/new/train_feature.pkl")
    val_features = get_feature(feature_path="/data/new/val_feature.pkl")
    test_features = get_feature(feature_path="/data/new/test_feature.pkl")

    result = mattching(test_features, catgory_features, topk=10)
    save_csv(result, "/data/FUNGI.csv")