"""Шаг 1. Подготовка данных: разбиение EuroSAT и кэширование выходов
замороженной части EfficientNet-B0.

Разбиение:
  * test  — официальный тестовый каталог репозитория (по 500 снимков на класс, 5000);
  * train / val — стратифицированное разбиение каталога train (22000) в пропорции 85/15.
Для обучающей выборки кэшируются два вида каждого снимка — исходный и
отражённый по горизонтали (аугментация).
"""
import os
import time
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (DATA_DIR, CACHE, OUT, IMG_SIZE, MEAN, STD, CLASSES, CLASSES_RU,
                    SEED, seed_everything, load_backbone, Bottom)

torch.set_num_threads(os.cpu_count())
seed_everything()
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def list_split(split):
    paths, labels = [], []
    for ci, c in enumerate(CLASSES):
        d = os.path.join(DATA_DIR, split, c)
        for f in sorted(os.listdir(d)):
            paths.append(os.path.join(d, f))
            labels.append(ci)
    return np.array(paths), np.array(labels)


tr_paths, tr_labels = list_split("train")
te_paths, te_labels = list_split("test")
tr_p, va_p, tr_y, va_y = train_test_split(tr_paths, tr_labels, test_size=0.15,
                                          stratify=tr_labels, random_state=SEED)

print("Classes:", CLASSES)
print(f"train={len(tr_p)}  val={len(va_p)}  test={len(te_paths)}")
print("{:<22}{:>7}{:>6}{:>6}".format("class", "train", "val", "test"))
for ci, c in enumerate(CLASSES):
    print("{:<22}{:>7}{:>6}{:>6}".format(c, (tr_y == ci).sum(), (va_y == ci).sum(), (te_labels == ci).sum()))

fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
for ci, ax in enumerate(axes.ravel()):
    ax.imshow(Image.open(tr_p[np.where(tr_y == ci)[0][0]]).convert("RGB"))
    ax.set_title(f"{CLASSES[ci]}\n({CLASSES_RU[CLASSES[ci]]})", fontsize=8.5)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eurosat_samples.png"), dpi=150)
plt.close()

tf = T.Compose([T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(), T.Normalize(MEAN, STD)])
bottom = Bottom(load_backbone()).eval()


@torch.no_grad()
def encode(paths, flip=False, bs=128):
    out = []
    t0 = time.time()
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])
        if flip:
            x = torch.flip(x, dims=[3])
        out.append(bottom(x).half())
    print(f"  encoded {len(paths)} imgs (flip={flip}) in {time.time() - t0:.0f}s")
    return torch.cat(out).numpy()


va = encode(va_p)
np.savez(os.path.join(CACHE, "val.npz"), x=va, y=va_y)
te = encode(te_paths)
np.savez(os.path.join(CACHE, "test.npz"), x=te, y=te_labels)
tr = np.concatenate([encode(tr_p), encode(tr_p, flip=True)])
np.savez(os.path.join(CACHE, "train.npz"), x=tr, y=np.concatenate([tr_y, tr_y]))
print("Cached feature map shape:", tr.shape[1:])
