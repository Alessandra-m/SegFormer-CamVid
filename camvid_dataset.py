import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter

import config as C

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_class_dict(csv_path: Path = None):
    names, colors = [], []
    with open(csv_path or C.CLASS_DICT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        for row in reader:
            names.append(row["name"].strip())
            colors.append([int(row["r"]), int(row["g"]), int(row["b"])])
    return names, np.asarray(colors, dtype=np.uint8)


def build_label_mapping():
    names, palette = load_class_dict()
    row_to_train = np.empty(len(names), dtype=np.int64)
    id2label, label2id = {}, {}
    next_id = 0
    for row, name in enumerate(names):
        if C.IGNORE_CLASS_NAME is not None and name.lower() == C.IGNORE_CLASS_NAME.lower():
            row_to_train[row] = C.IGNORE_INDEX
        else:
            row_to_train[row] = next_id
            id2label[next_id] = name
            label2id[name] = next_id
            next_id += 1

    num_labels = next_id
    train_palette = np.zeros((num_labels, 3), dtype=np.uint8)
    for row, train_id in enumerate(row_to_train):
        if train_id != C.IGNORE_INDEX:
            train_palette[train_id] = palette[row]

    return dict(
        palette=palette,
        row_to_train=row_to_train,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
        train_palette=train_palette,
    )


def _color_key(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int64)
    return (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]


def rgb_to_index(mask_rgb: np.ndarray, palette: np.ndarray, row_to_train: np.ndarray) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    flat_key = _color_key(mask_rgb.reshape(-1, 3))
    pal_key = _color_key(palette)

    order = np.argsort(pal_key)
    sorted_keys = pal_key[order]
    pos = np.clip(np.searchsorted(sorted_keys, flat_key), 0, len(sorted_keys) - 1)
    matched = sorted_keys[pos] == flat_key

    out = np.full(flat_key.shape, C.IGNORE_INDEX, dtype=np.int64)
    out[matched] = row_to_train[order[pos][matched]]
    return out.reshape(h, w)


def colorize(index_map: np.ndarray, train_palette: np.ndarray) -> np.ndarray:
    out = np.zeros(index_map.shape + (3,), dtype=np.uint8)
    valid = index_map != C.IGNORE_INDEX
    out[valid] = train_palette[index_map[valid]]
    return out


def _list_pairs(split: str):
    img_dir = C.DATA_ROOT / C.SPLIT_DIRS[split][0]
    lbl_dir = C.DATA_ROOT / C.SPLIT_DIRS[split][1]
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    label_by_stem = {}
    for path in lbl_dir.glob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = path.stem
        if C.LABEL_SUFFIX and stem.endswith(C.LABEL_SUFFIX):
            stem = stem[: -len(C.LABEL_SUFFIX)]
        label_by_stem[stem] = path

    pairs = []
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = label_by_stem.get(img.stem)
        if lbl is None:
            raise FileNotFoundError(f"No mask for {img.name} (expected {img.stem}{C.LABEL_SUFFIX})")
        pairs.append((img, lbl))
    if not pairs:
        raise RuntimeError(f"Empty split '{split}' in {img_dir}")
    return pairs


class CamVidDataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, mapping, train=False):
        self.pairs = _list_pairs(split)
        self.processor = processor
        self.mapping = mapping
        self.train = train
        self.jitter = ColorJitter(**C.COLOR_JITTER) if train else None

    def __len__(self):
        return len(self.pairs)

    def _load_label_index(self, lbl_path):
        if C.MASK_MODE == "rgb":
            mask_rgb = np.array(Image.open(lbl_path).convert("RGB"))
            return rgb_to_index(mask_rgb, self.mapping["palette"], self.mapping["row_to_train"])
        if C.MASK_MODE == "index":
            return np.array(Image.open(lbl_path)).astype(np.int64)
        raise ValueError(f"Unknown MASK_MODE: {C.MASK_MODE}")

    def original(self, i):
        img_path, lbl_path = self.pairs[i]
        image = np.array(Image.open(img_path).convert("RGB"))
        return image, self._load_label_index(lbl_path)

    def _random_scale_crop(self, image, label):
        w, h = image.size
        scale = random.uniform(*C.SCALE_RANGE) * C.SHORT_SIDE / min(h, w)
        size = (round(w * scale), round(h * scale))
        image = np.array(image.resize(size, Image.BILINEAR))
        label = np.array(Image.fromarray(label.astype(np.uint8)).resize(size, Image.NEAREST))

        pad_h = max(C.CROP_SIZE - label.shape[0], 0)
        pad_w = max(C.CROP_SIZE - label.shape[1], 0)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
        label = np.pad(label, ((0, pad_h), (0, pad_w)), constant_values=C.IGNORE_INDEX)

        top = random.randint(0, label.shape[0] - C.CROP_SIZE)
        left = random.randint(0, label.shape[1] - C.CROP_SIZE)
        crop = np.s_[top:top + C.CROP_SIZE, left:left + C.CROP_SIZE]
        return Image.fromarray(image[crop]), label[crop].astype(np.int64)

    def _resize(self, image, label):
        h, w = C.EVAL_SIZE
        image = image.resize((w, h), Image.BILINEAR)
        label = np.array(Image.fromarray(label.astype(np.uint8)).resize((w, h), Image.NEAREST))
        return image, label.astype(np.int64)

    def __getitem__(self, i):
        img_path, lbl_path = self.pairs[i]
        image = Image.open(img_path).convert("RGB")
        label = self._load_label_index(lbl_path)

        if self.train:
            if C.TRAIN_AUG == "scale_crop":
                image, label = self._random_scale_crop(image, label)
            elif C.TRAIN_AUG == "resize":
                image, label = self._resize(image, label)
            else:
                raise ValueError(f"Unknown TRAIN_AUG: {C.TRAIN_AUG}")
            image = self.jitter(image)
            if random.random() < C.HFLIP_PROB:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label[:, ::-1].copy()
            enc = self.processor(image, do_resize=False, return_tensors="pt")
        else:
            enc = self.processor(image, return_tensors="pt")

        return {
            "pixel_values": enc["pixel_values"][0],
            "labels": torch.from_numpy(label),
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
