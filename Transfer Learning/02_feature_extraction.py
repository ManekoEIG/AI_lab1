"""Шаг 2. Стратегия «Feature Extraction».

Все веса EfficientNet-B0 заморожены, обучается только новый выходной слой
(Dropout 0.2 + Linear 1280 -> 10). Так как вся базовая сеть заморожена,
1280-мерные признаки считаются один раз, после чего классификатор обучается
на них за секунды.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from metrics_utils import evaluate_and_save
from common import (OUT, CLASSES, seed_everything, load_backbone, TopWithHead,
                    load_cache, count_params)

torch.set_num_threads(os.cpu_count())
seed_everything()

EPOCHS, LR, BS = 30, 1e-3, 256

data = load_cache()
model = TopWithHead(load_backbone(), num_classes=len(CLASSES)).eval()
for p in model.parameters():
    p.requires_grad = False
for p in model.head.parameters():
    p.requires_grad = True
print(f"Total params: {count_params(model):,} | trainable: {count_params(model, True):,}")


@torch.no_grad()
def extract(x, bs=512):
    return torch.cat([model.features(x[i:i + bs].float()) for i in range(0, len(x), bs)])


t0 = time.time()
feats = {s: (extract(x), y) for s, (x, y) in data.items()}
print(f"Pooled 1280-d features computed in {time.time() - t0:.0f}s")

opt = torch.optim.Adam(model.head.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_acc, best_state = -1, None
Xtr, ytr = feats["train"]
Xva, yva = feats["val"]

t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.head.train()
    perm = torch.randperm(len(Xtr))
    tl, tc = 0.0, 0
    for i in range(0, len(Xtr), BS):
        idx = perm[i:i + BS]
        out = model.head(Xtr[idx])
        loss = loss_fn(out, ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tl += loss.item() * len(idx)
        tc += (out.argmax(1) == ytr[idx]).sum().item()
    model.head.eval()
    with torch.no_grad():
        vo = model.head(Xva)
        vl = loss_fn(vo, yva).item()
        va = (vo.argmax(1) == yva).float().mean().item()
    hist["train_loss"].append(tl / len(Xtr)); hist["train_acc"].append(tc / len(Xtr))
    hist["val_loss"].append(vl); hist["val_acc"].append(va)
    if va > best_acc:
        best_acc, best_ep = va, ep
        best_state = {k: v.clone() for k, v in model.head.state_dict().items()}
    print(f"epoch {ep:2d}  train_loss {tl/len(Xtr):.4f}  train_acc {tc/len(Xtr):.4f}  "
          f"val_loss {vl:.4f}  val_acc {va:.4f}")
train_time = time.time() - t0
print(f"Best val_acc {best_acc:.4f} at epoch {best_ep}; head training time {train_time:.1f}s")

model.head.load_state_dict(best_state)
torch.save(best_state, os.path.join(OUT, "fe_head.pt"))

model.eval()
with torch.no_grad():
    y_pred = model.head(feats["test"][0]).argmax(1).numpy()
evaluate_and_save("feature_extraction", "Feature Extraction", feats["test"][1].numpy(), y_pred,
                  hist, extra={"trainable_params": count_params(model, True),
                               "epochs": EPOCHS, "best_epoch": best_ep, "lr": LR,
                               "train_time_s": round(train_time, 1)})
