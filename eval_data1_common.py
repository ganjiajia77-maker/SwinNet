import os

import numpy as np
import torch
from PIL import Image


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def image_names(root, split):
    directory = os.path.join(root, split, "image")
    return sorted(name for name in os.listdir(directory)
                  if name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")))


def mask_path(root, split, image_name):
    directory = os.path.join(root, split, "mask")
    if not os.path.isdir(directory):
        directory = os.path.join(root, split, "label")
    base, _ = os.path.splitext(image_name)
    candidates = [image_name, base + ".png", base + ".tif",
                  base.replace("_sat", "_mask") + ".png",
                  base.replace("_image", "_mask") + ".png"]
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Cannot find mask for " + image_name)


def positions(length, tile, stride):
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    last = length - tile
    if values[-1] != last:
        values.append(last)
    return values


def tensor_from_pil(image):
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1)).float()


def predict_full(model, image, tile_size=512, stride=512, device="cuda"):
    width, height = image.size
    output = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    with torch.no_grad():
        for top in positions(height, tile_size, stride):
            for left in positions(width, tile_size, stride):
                crop = image.crop((left, top, min(left + tile_size, width), min(top + tile_size, height)))
                if crop.size != (tile_size, tile_size):
                    padded = Image.new("RGB", (tile_size, tile_size))
                    padded.paste(crop, (0, 0))
                    crop = padded
                prediction, _, _ = model(tensor_from_pil(crop).unsqueeze(0).to(device))
                prediction = prediction[0, 0].detach().cpu().numpy()
                h = min(tile_size, height - top)
                w = min(tile_size, width - left)
                output[top:top + h, left:left + w] += prediction[:h, :w]
                count[top:top + h, left:left + w] += 1.0
    return output / np.maximum(count, 1.0)


def load_model(args, device):
    from modeling.coanet import CoANet
    model = CoANet(num_classes=1, backbone=args.backbone, output_stride=args.out_stride,
                   sync_bn=False, freeze_bn=False)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def scores(probability, target, threshold):
    prediction = probability >= threshold
    target = target.astype(bool)
    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "iou": iou, "f1": f1, "precision": precision, "recall": recall}
