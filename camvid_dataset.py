from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import config as C


def load_class_dict(csv_path: Path = C.CLASS_DICT_CSV):
    """Читает class_dict.csv (колонки name,r,g,b).

    Возвращает:
        names   : list[str]              — имена классов в порядке файла
        palette : np.ndarray (K, 3) uint8 — RGB-цвет каждого класса
    """
    names, colors = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip().lower(): v for k, v in row.items()}
            names.append(row["name"].strip())
            colors.append([int(row["r"]), int(row["g"]), int(row["b"])])
    palette = np.asarray(colors, dtype=np.uint8)
    return names, palette


def build_label_mapping():
    """Строит отображение из строк class_dict.csv в train-id (и обратно).

    Игнорируемый класс (C.IGNORE_CLASS_NAME) получает train-id = IGNORE_INDEX (255)
    и исключается из num_labels; остальные классы перенумеровываются подряд 0..num-1.

    Возвращает dict с полями:
        palette      : (K,3) uint8 — исходные цвета всех K строк csv
        row_to_train : (K,) int    — train-id для каждой строки csv (или IGNORE_INDEX)
        id2label     : {train_id: name}
        label2id     : {name: train_id}
        num_labels   : int
        train_palette: (num_labels,3) uint8 — цвет для каждого train-id (для визуализации)
    """
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
    for row in range(len(names)):
        tid = row_to_train[row]
        if tid != C.IGNORE_INDEX:
            train_palette[tid] = palette[row]

    return dict(
        palette=palette,
        row_to_train=row_to_train,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
        train_palette=train_palette,
    )


def rgb_to_index(mask_rgb: np.ndarray, palette: np.ndarray, row_to_train: np.ndarray) -> np.ndarray:
    """Цветную маску (H,W,3) -> карту train-id (H,W) int64.

    Каждый пиксель сопоставляется со строкой палитры по точному совпадению RGB;
    непопавшие цвета (артефакты сжатия, «нет класса») -> IGNORE_INDEX.
    """
    h, w, _ = mask_rgb.shape
    flat = mask_rgb.reshape(-1, 3).astype(np.int64)
    # кодируем цвет одним числом r*256*256 + g*256 + b для быстрого сопоставления
    flat_key = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    pal_key = (palette[:, 0].astype(np.int64) << 16) | (palette[:, 1].astype(np.int64) << 8) | palette[:, 2]

    out = np.full(flat_key.shape, C.IGNORE_INDEX, dtype=np.int64)
    # сортированный поиск строки палитры по ключу цвета
    order = np.argsort(pal_key)
    sorted_keys = pal_key[order]
    pos = np.searchsorted(sorted_keys, flat_key)
    pos_clipped = np.clip(pos, 0, len(sorted_keys) - 1)
    matched = sorted_keys[pos_clipped] == flat_key
    rows = order[pos_clipped]                      # индекс строки csv для каждого пикселя
    out[matched] = row_to_train[rows[matched]]     # -> train-id (или IGNORE_INDEX для Void)
    return out.reshape(h, w)


def colorize(index_map: np.ndarray, train_palette: np.ndarray) -> np.ndarray:
    """Карту train-id (H,W) -> RGB (H,W,3) для визуализации. IGNORE_INDEX -> чёрный."""
    h, w = index_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = index_map != C.IGNORE_INDEX
    out[valid] = train_palette[index_map[valid]]
    return out


# ---------------------------------------------------------------------------
# Сопоставление картинок и масок
# ---------------------------------------------------------------------------
def _list_pairs(split: str):
    """Возвращает список (image_path, label_path) для сплита."""
    img_dir = C.DATA_ROOT / C.SPLIT_DIRS[split][0]
    lbl_dir = C.DATA_ROOT / C.SPLIT_DIRS[split][1]
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка картинок: {img_dir}")

    # маски индексируем по stem без суффикса _L, чтобы матчить независимо от расширения
    label_by_stem = {}
    for p in lbl_dir.glob("*"):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        stem = p.stem
        if C.LABEL_SUFFIX and stem.endswith(C.LABEL_SUFFIX):
            stem = stem[: -len(C.LABEL_SUFFIX)]
        label_by_stem[stem] = p

    pairs = []
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        lbl = label_by_stem.get(img.stem)
        if lbl is None:
            raise FileNotFoundError(f"Нет маски для {img.name} (ожидался {img.stem}{C.LABEL_SUFFIX})")
        pairs.append((img, lbl))
    if not pairs:
        raise RuntimeError(f"Пустой сплит '{split}' в {img_dir}")
    return pairs


# ---------------------------------------------------------------------------
# Dataset для HuggingFace Trainer
# ---------------------------------------------------------------------------
class CamVidDataset:
    """torch-совместимый Dataset. __getitem__ -> {"pixel_values": (3,H,W), "labels": (H,W)}.

    Параметры:
        split      : "train" | "val" | "test"
        processor  : SegformerImageProcessor (resize+нормализация; do_reduce_labels=False)
        mapping    : результат build_label_mapping()
        augment    : применять ли ColorJitter + hflip (обычно True только для train)
    """

    def __init__(self, split, processor, mapping, augment=False):
        self.pairs = _list_pairs(split)
        self.processor = processor
        self.mapping = mapping
        self.augment = augment
        self._jitter = None
        if augment:
            from torchvision.transforms import ColorJitter
            self._jitter = ColorJitter(**C.COLOR_JITTER)

    def __len__(self):
        return len(self.pairs)

    def _load_label_index(self, lbl_path):
        if C.MASK_MODE == "rgb":
            mask_rgb = np.array(Image.open(lbl_path).convert("RGB"))
            return rgb_to_index(mask_rgb, self.mapping["palette"], self.mapping["row_to_train"])
        elif C.MASK_MODE == "index":
            idx = np.array(Image.open(lbl_path))          # уже индексы классов
            return idx.astype(np.int64)
        raise ValueError(f"Неизвестный MASK_MODE: {C.MASK_MODE}")

    def original(self, i):
        """Исходные картинка (H,W,3) uint8 и маска train-id (H,W) без ресайза — для оценки в родном разрешении."""
        img_path, lbl_path = self.pairs[i]
        image = np.array(Image.open(img_path).convert("RGB"))
        return image, self._load_label_index(lbl_path)

    def __getitem__(self, i):
        img_path, lbl_path = self.pairs[i]
        image = Image.open(img_path).convert("RGB")
        label = self._load_label_index(lbl_path)          # (H,W) train-id

        if self.augment:
            if self._jitter is not None:
                image = self._jitter(image)
            if random.random() < C.HFLIP_PROB:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label[:, ::-1].copy()

        label_img = Image.fromarray(label.astype(np.uint8) if label.max() < 256 else label.astype(np.int32))
        enc = self.processor(image, segmentation_maps=label_img, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"][0],       # (3, H, W)
            "labels": enc["labels"][0],                   # (H, W) long
        }


def collate_fn(batch):
    """Склеивает элементы в батч для Trainer."""
    import torch
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
