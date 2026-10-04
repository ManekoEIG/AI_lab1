"""Шаг 3. Стратегия «Fine-tuning».

Начальная точка — модель после Feature Extraction (обученная голова), чтобы
случайно инициализированный классификатор не «ломал» предобученные веса
большими градиентами на первых шагах. Затем размораживаются верхние слои
базовой сети и вся модель дообучается с низким learning rate.

Варианты (аргумент --unfreeze):
  top2  — blocks[5], blocks[6], conv_head (+bn2)   [основной эксперимент]
  top1  — только blocks[6], conv_head (+bn2)
BatchNorm-слои остаются в режиме eval (статистики ImageNet фиксированы,
обучаются только их масштаб и сдвиг).
"""
import os
import time
import argparse
import torch
import torch.nn as nn
from metrics_utils import evaluate_and_save
from common import (OUT, CLASSES, seed_everything, load_backbone, TopWithHead,
                    load_cache, count_params, freeze_bn_stats)

ap = argparse.ArgumentParser()
ap.add_argument("--unfreeze", default="top1", choices=["top1", "top2"])
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--tag", default="fine_tuning")
args = ap.parse_args()

torch.set_num_threads(os.cpu_count())
seed_everything()
BS = 64

data = load_cache()
model = TopWithHead(load_backbone(), num_classes=len(CLASSES))
model.head.load_state_dict(torch.load(os.path.join(OUT, "fe_head.pt")))

for p in model.parameters():
    p.requires_grad = False
to_train = [model.conv_head, model.bn2, model.head, model.blocks[-1]]
if args.unfreeze == "top2":
    to_train.append(model.blocks[-2])
for m in to_train:
    for p in m.parameters():
        p.requires_grad = True
print(f"Unfrozen: {args.unfreeze} | lr={args.lr} | epochs={args.epochs}")
print(f"Total params: {count_params(model):,} | trainable: {count_params(model, True):,}")

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
Xtr, ytr = data["train"]
steps = args.epochs * ((len(Xtr) + BS - 1) // BS)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def predict(x, bs=512):
    model.eval()
    return torch.cat([model(x[i:i + bs].float()) for i in range(0, len(x), bs)])


# Точность до дообучения (= Feature Extraction) на валидации
vo = predict(data["val"][0])
print(f"val_acc before fine-tuning: {(vo.argmax(1) == data['val'][1]).float().mean():.4f}")

hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_acc, best_state = -1, None
t0 = time.time()
for ep in range(1, args.epochs + 1):
    model.train()
    freeze_bn_stats(model)
    perm = torch.randperm(len(Xtr))
    tl, tc = 0.0, 0
    for i in range(0, len(Xtr), BS):
        idx = perm[i:i + BS]
        out = model(Xtr[idx].float())
        loss = loss_fn(out, ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tl += loss.item() * len(idx)
        tc += (out.argmax(1) == ytr[idx]).sum().item()
    vo = predict(data["val"][0])
    vl = loss_fn(vo, data["val"][1]).item()
    va = (vo.argmax(1) == data["val"][1]).float().mean().item()
    hist["train_loss"].append(tl / len(Xtr)); hist["train_acc"].append(tc / len(Xtr))
    hist["val_loss"].append(vl); hist["val_acc"].append(va)
    if va > best_acc:
        best_acc, best_ep = va, ep
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"epoch {ep:2d}  train_loss {tl/len(Xtr):.4f}  train_acc {tc/len(Xtr):.4f}  "
          f"val_loss {vl:.4f}  val_acc {va:.4f}  ({time.time()-t0:.0f}s)", flush=True)
train_time = time.time() - t0
print(f"Best val_acc {best_acc:.4f} at epoch {best_ep}; training time {train_time:.0f}s")

model.load_state_dict(best_state)
y_pred = predict(data["test"][0]).argmax(1).numpy()
title = {"top2": "Fine-tuning", "top1": "Fine-tuning (только blocks[6])"}[args.unfreeze]
if args.lr != 1e-4:
    title += f", lr={args.lr:g}"
evaluate_and_save(args.tag, title, data["test"][1].numpy(), y_pred, hist,
                  extra={"trainable_params": count_params(model, True), "epochs": args.epochs,
                         "best_epoch": best_ep, "lr": args.lr, "unfreeze": args.unfreeze,
                         "train_time_s": round(train_time, 1)})
