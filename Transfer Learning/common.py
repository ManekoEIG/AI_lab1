"""Общие элементы части 2 (Transfer Learning): данные EuroSAT и модель EfficientNet-B0.

Модель разбита на две части:
  * bottom — stem + blocks[0..4] (всегда заморожены в обеих стратегиях);
  * top    — blocks[5], blocks[6], conv_head, bn2, global pooling;
  * head   — новый классификатор (Dropout + Linear 1280 -> 10).

Поскольку bottom заморожен в обеих стратегиях, его выходы (карты признаков
8x8x112) вычисляются один раз и кэшируются. Это даёт тот же результат, что и
прогон через замороженные слои на каждой эпохе, но позволяет обучать на CPU.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import timm

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("EUROSAT_DIR", os.path.join(ROOT, "data", "dataset_rgb"))
WEIGHTS = os.path.join(ROOT, "weights", "efficientnet_b0_ra-3dd342df.pth")
CACHE = os.path.join(ROOT, "cache")
OUT = os.path.join(ROOT, "output")
IMG_SIZE = 128          # снимки EuroSAT 64x64 увеличиваются в 2 раза
SEED = 42
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
CLASSES_RU = {
    "AnnualCrop": "Однолетние культуры", "Forest": "Лес",
    "HerbaceousVegetation": "Травянистая растит.", "Highway": "Шоссе",
    "Industrial": "Промзона", "Pasture": "Пастбище",
    "PermanentCrop": "Многолетние культуры", "Residential": "Жилая застройка",
    "River": "Река", "SeaLake": "Море/озеро",
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_backbone():
    """EfficientNet-B0, предобученная на ImageNet-1k (веса timm, top-1 77.7%)."""
    model = timm.create_model("efficientnet_b0", pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    return model


class Bottom(nn.Module):
    """Нижняя (всегда замороженная) часть сети: stem + blocks[0..4]."""

    def __init__(self, m):
        super().__init__()
        self.conv_stem, self.bn1 = m.conv_stem, m.bn1
        self.blocks = m.blocks[:5]

    def forward(self, x):
        return self.blocks(self.bn1(self.conv_stem(x)))


class TopWithHead(nn.Module):
    """Верхняя часть сети + новый классификатор."""

    def __init__(self, m, num_classes, dropout=0.2):
        super().__init__()
        self.blocks = m.blocks[5:]
        self.conv_head, self.bn2 = m.conv_head, m.bn2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.num_features, num_classes))

    def features(self, x):
        x = self.bn2(self.conv_head(self.blocks(x)))
        return torch.flatten(self.pool(x), 1)

    def forward(self, x):
        return self.head(self.features(x))


def freeze_bn_stats(module):
    """BatchNorm-слои в режиме eval: статистики ImageNet не перезаписываются."""
    for mod in module.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def count_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def load_cache():
    d = {}
    for split in ("train", "val", "test"):
        z = np.load(os.path.join(CACHE, f"{split}.npz"))
        d[split] = (torch.from_numpy(z["x"]), torch.from_numpy(z["y"]).long())
    return d
