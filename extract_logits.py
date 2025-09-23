import argparse
import torch
import os
import clip
import time
import datetime
import numpy as np
import pandas as pd

from modeling.utils import extract_vision_features, predict_from_features
from modeling.models import Adapter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')

from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if not os.path.isfile(img_path):
            print(f"[WARNING] File not found: {img_path}. Skipping this sample.")
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] {img_path} cannot be opened: {e}")
                image = Image.new("RGB", (224, 224), (255, 255, 255))
        if self.transforms:
            image = self.transforms(image)
        return image  # <-- chỉ trả về image


def set_loader(dataset_name, transforms, batch_size, image_paths=None):
    if image_paths is not None:
        dataset = CustomImageDataset(image_paths, transforms=transforms)
    else:
        raise NotImplementedError("Chỉ hỗ trợ truyền image_paths cho test dataset!")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def load_image_paths_from_hadoop():
    hadoop_output_dir = "E:/CLIP-Conformal-main/firesmoke_output"
    file_paths = [
        os.path.join(hadoop_output_dir, "part-00000"),
        os.path.join(hadoop_output_dir, "part-00001")
    ]
    image_paths = []
    for fp in file_paths:
        with open(fp, "r") as f:
            for line in f:
                path = line.strip().split()[0]  # lấy cột 1
                # Nếu thiếu đuôi .jpg thì tự động thêm vào
                if not path.endswith(".jpg"):
                    path += ".jpg"
                image_dir = "E:/CLIP-Conformal-main"
                full_path = os.path.join(image_dir, path) if not os.path.isabs(path) else path
                image_paths.append(full_path)
    return image_paths


classnames_dict = {
    "firesmoke": [
        "bothFireAndSmoke",
        "fire",
        "neitherFireNorSmoke",
        "smoke"
    ],
    "dtd": [
        "banded", "blotchy", "braided", "bubbly", "cobwebbed"
    ],
}

templates_dict = {
    "firesmoke": [
        "a photo of {}.",
        "an image of {}.",
        "a surveillance image of {}.",
        "a scene with {}."
    ],
    "dtd": [
        "a texture of {}.",
        "a photo of {} pattern.",
        "a surface that is {}."
    ],
}

def process(args):
    image_paths = load_image_paths_from_hadoop()
    print(f"Đã load {len(image_paths)} ảnh đầu vào từ Hadoop.")

    batch_size = args.bs
    backbone = args.backbone
    test_datasets = args.test_datasets

    # Load CLIP Model
    if backbone.split("-")[0] == "CLIP":
        model_clip, transforms = clip.load("-".join(backbone.split("-")[1:]))
        model_clip.to(device).float()
        model_clip.eval()
    elif backbone.split("-")[0] == "MetaCLIP":
        from transformers import AutoProcessor, AutoModel
        name_id_match = {"MetaCLIP-ViT-B/16": "facebook/metaclip-b16-fullcc2.5b",
                         "MetaCLIP-ViT-H/14": "facebook/metaclip-h14-fullcc2.5b"}
        transforms = AutoProcessor.from_pretrained(name_id_match[backbone]).image_processor
        model_clip = AutoModel.from_pretrained(name_id_match[backbone])
        model_clip.to(device).float()
        model_clip.eval()
    else:
        print("Architecture not supported...")
        return

    experiment = {"test": {}}
    for i in range(len(test_datasets)):
        experiment["test"][i] = {}
        experiment["test"][i]["domain"] = {test_datasets[i]}
        experiment["test"][i]["dataloader"] = set_loader(
            test_datasets[i],
            transforms=transforms,
            batch_size=batch_size,
            image_paths=image_paths
        )

    time_extraction = []
    for i_domain in range(0, len(experiment["test"])):
        dataset_name = test_datasets[i_domain]
        print("  Processing: [{dataset}]".format(dataset=dataset_name))

        model_clip = model_clip.to(device)

        classnames = classnames_dict.get(dataset_name, classnames_dict["firesmoke"])
        templates = templates_dict.get(dataset_name, templates_dict["firesmoke"])

        adapter = Adapter(model_clip,
                          classnames=classnames,
                          adapter="ZS",
                          templates=templates,
                          clip_id=backbone).to(device)

        time_adapt_i_1 = time.time()
        id = "./local_data/cache/" + dataset_name + "_" + backbone.lower().replace("/", "_")
        if not os.path.isfile(id + ".npz"):
            print("  Extracting features and saving in disk")
            feats_ds, refs_ds = extract_vision_features(model_clip, experiment["test"][i_domain]["dataloader"],
                                                        clip_id=backbone)
            print("  Extracting logits")
            logits_ds = predict_from_features(adapter, torch.tensor(feats_ds), bs=args.bs, act=False, epsilon=1.0)
            logits_ds = logits_ds.cpu().numpy()
            time_adapt_i_2 = time.time()
            print("  Saving in disk")
            np.savez(id, logits_ds=logits_ds, refs_ds=refs_ds)
        else:
            time_adapt_i_2 = time.time()
        time_adapt_i = time_adapt_i_2 - time_adapt_i_1
        time_extraction.append(time_adapt_i)
        print(str("Feature extraction time: " + str(datetime.timedelta(seconds=time_adapt_i))))
    print("Average time: " + str(datetime.timedelta(seconds=np.mean(time_extraction))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_datasets',
                        default='firesmoke',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--backbone', default='CLIP-ViT-B/16')
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--cache_features', default=False, type=lambda x: (str(x).lower() == 'true'))
    args, unknown = parser.parse_known_args()
    process(args=args)

if __name__ == "__main__":
    main()
