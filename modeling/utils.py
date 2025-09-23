import torch
import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_vision_features(model_clip, loader, augmentations=False, clip_id="CLIP"):

    reps = 10 if augmentations else 1

    feats_ds_rep = []
    for irep in range(reps):
        feats_ds = []
        for step, batch in enumerate(loader):
            print(f"  Batch {step + 1}/{len(loader)}", end="\r")

            # Nếu batch là tensor (chuẩn nhất)
            images = batch.to(device).to(torch.float32)

            # Forward predictions
            with torch.no_grad():
                if clip_id.split("-")[0] == "CLIP":
                    feats = model_clip.visual(images)
                elif clip_id.split("-")[0] == "MetaCLIP":
                    feats = model_clip.get_image_features(images)
                else:
                    raise NotImplementedError("Unknown clip_id")

            feats_ds.append(feats.detach().cpu().numpy())

        # Concatenate features
        feats_ds = np.concatenate(feats_ds, axis=0)
        feats_ds_rep.append(np.expand_dims(feats_ds, -1))

    # Concatenate augmentations
    feats_ds_rep = np.squeeze(np.concatenate(feats_ds_rep, axis=-1))

    # Không trả về label (refs_ds_rep = None)
    return feats_ds_rep, None

def predict_from_features(adapter, feats_ds, bs=512, act=True, epsilon=1.0):
    """
    Dự đoán từ features đã extract
    Args:
        adapter: model adapter
        feats_ds: numpy array hoặc torch tensor, shape (N, D)
        bs: batch size
        act: có dùng softmax không
        epsilon: temperature cho softmax
    Returns:
        preds: torch tensor, shape (N, num_classes)
    """
    preds = []
    idx = 0

    # Đảm bảo feats_ds là torch tensor
    if not torch.is_tensor(feats_ds):
        feats_ds = torch.tensor(feats_ds)

    total = feats_ds.shape[0]

    while idx < total:
        x = feats_ds[idx:idx+bs, :].to(device).to(torch.float32)

        with torch.no_grad():
            pred = adapter(x)
            if act:
                pred = torch.softmax(pred / epsilon, dim=-1)

        preds.append(pred.detach().cpu())

        idx += bs

    preds = torch.cat(preds, axis=0)

    return preds
