import osget_feature
import pickle
import shutil

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel, CLIPImageProcessor


from process_data import load_csv

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_csv_gt(file_path):
    head, data = load_csv(file_path)
    target = {}
    for line in data:
        index = head.index("observationID")
        target[line[index]] = line[head.index("category_id")]
    return target


def save_csv(result, save_path):
    line = ["observationId,predictions"]
    for key, value in result.items():
        line.append(f"{key},{' '.join(value)}")
    with open(save_path, "w") as f:
        f.write("\n".join(line))


def load_dinov2(model_path='/data/models/Dinov2G', device='cuda'):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, device_map="auto", ignore_mismatched_sizes=True).to(device)
    return model, processor




def dict2batch_images(pkl_path, image_dir, batch_size=32):
    pickle_file = load_pkl(pkl_path)
    batch_data = []
    item_batch = []
    for key, value in pickle_file.items():
        for i, item in enumerate(value["image_urls"]):
            item_batch.append((key, value["old_info"][i]["observationID"], os.path.join(image_dir, os.path.basename(item))))
            if len(item_batch) == batch_size:
                batch_data.append(item_batch)
                item_batch = []
    if len(item_batch) > 0:
        batch_data.append(item_batch)
    return batch_data


def vectorization_images(model, preprocess, image_batch, device='cuda', bf=False):
    batch_images = []
    for _, _, image_path in image_batch:
        image = Image.open(image_path).convert("RGB")
        batch_images.append(preprocess(image, return_tensors="pt"))
    if bf:
        inputs = torch.concatenate([image["pixel_values"] for image in batch_images], dim=0).to(device).bfloat16()
    else:
        inputs = torch.concatenate([image["pixel_values"] for image in batch_images], dim=0).to(device).half()
    with torch.no_grad():
        outputs = model(pixel_values=inputs)
        embeddings = outputs.last_hidden_state
    #     embeddings = embeddings.mean(dim=1)
    # embeddings /= embeddings.norm(dim=-1, keepdim=True)
    return embeddings.detach().to(torch.float16).cpu().numpy()


def get_feature(model, preprocess, batch_data, device='cuda', feature_path=None, flag=0):
    catgory_features = {}
    if feature_path is not None and os.path.exists(feature_path):
        catgory_features = load_pkl(feature_path)
    else:
        for batch_item in tqdm(batch_data, desc="feature vectorization"):
            features = vectorization_images(model, preprocess, batch_item, device, bf=True)
            for info, feature in zip(batch_item, features):
                catgory_features[info[flag]] = catgory_features.get(info[flag], [])
                catgory_features[info[flag]].append(feature)
    if feature_path is not None:
        save_pkl(catgory_features, feature_path)
    return catgory_features


def get_feature_dir(model, preprocess, batch_data, device='cuda', feature_path=None, flag=0):
    if feature_path is not None and os.path.exists(feature_path):
        for feature_id in os.listdir(feature_path):
            catgory_features[feature_id] = load_pkl(feature_path)
    else:
        for batch_item in tqdm(batch_data, desc="feature vectorization"):
            features = vectorization_images(model, preprocess, batch_item, device, bf=True)
            for info, feature in zip(batch_item, features):
                os.makedirs(os.path.join(feature_path, info[flag]), exist_ok=True)
                count_id = len(os.listdir(os.path.join(feature_path, info[flag])))
                save_pkl(feature, os.path.join(feature_path, info[flag], str(count_id))+".pkl")
        catgory_features = None
    return catgory_features


def merge_feature(feature_path):
    for feature_id in os.listdir(feature_path):
        features = []
        for feature in os.listdir(os.path.join(feature_path, feature_id)):
            features.append(load_pkl(os.path.join(feature_path, feature_id, feature)))
        save_pkl(features, os.path.join(feature_path, feature_id)+".pkl")
        shutil.rmtree(os.path.join(feature_path, feature_id))


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


def eval(csv_path, result, topk=10):
    gt = load_csv_gt(csv_path)
    target = np.zeros(topk)
    for key, value in result.items():
        if gt.get(key) is None:
            continue
        if gt[key] in value:
            target[value.index(gt[key]):] += 1
    for item in enumerate(target):
        print(f"top-{item[0]+1}: {int(item[1])}, acc: {item[1]/len(result):.4f}")


if __name__ == "__main__":
    device = 'cuda'
    train_pkl = "FungiTastic-FewShot-Train.pkl"
    test_pkl = "FungiTastic-FewShot-Test.pkl"
    val_pkl = 'FungiTastic-FewShot-Val.pkl'

    model, preprocess = load_dinov2(model_path="/data/models/dinov2G", device=device)
    train_batch = dict2batch_images(train_pkl, "/data/fungi/images/FungiTastic-FewShot/train/fullsize", 256)
    test_batch = dict2batch_images(test_pkl, "/data/fungi/images/FungiTastic-FewShot/test/fullsize", 256)
    val_batch = dict2batch_images(val_pkl, "/data/fungi/images/FungiTastic-FewShot/val/fullsize", 256)

    train_features = get_feature_dir(model, preprocess, train_batch, device=device,feature_path="/data/train_features", flag=0)
    test_features = get_feature_dir(model, preprocess, test_batch, device=device, feature_path="/data/test_features", flag=0)
    val_features = get_feature_dir(model, preprocess, val_batch, device=device, feature_path="/data/val_features", flag=0)
