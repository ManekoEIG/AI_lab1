"""Шаг 4. Сравнение стратегий Feature Extraction и Fine-tuning."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, CLASSES

runs = [("feature_extraction", "Feature Extraction"),
        ("fine_tuning", "Fine-tuning (blocks[5..6], lr=1e-4, 8 ep.)"),
        ("ft_top1", "Fine-tuning (blocks[6], lr=1e-4, 4 ep.)"),
        ("ft_lr1e-3", "Fine-tuning (blocks[5..6], lr=1e-3, 3 ep.)")]
res = {}
for tag, name in runs:
    p = os.path.join(OUT, f"{tag}_metrics.json")
    if os.path.exists(p):
        res[tag] = (name, json.load(open(p)))

print("{:<44}{:>9}{:>11}{:>9}{:>9}{:>8}{:>12}{:>9}".format(
    "strategy", "Acc,%", "Prec,%", "Rec,%", "F1,%", "errors", "trainable", "time,s"))
for tag, (name, r) in res.items():
    print("{:<44}{:>9.2f}{:>11.2f}{:>9.2f}{:>9.2f}{:>8}{:>12,}{:>9.0f}".format(
        name, 100 * r["accuracy"], 100 * r["precision_macro"], 100 * r["recall_macro"],
        100 * r["f1_macro"], r["errors"], r["trainable_params"], r["train_time_s"]))

fe, ft = res["feature_extraction"][1], res["fine_tuning"][1]
print(f"\nError reduction FE -> FT: {fe['errors']} -> {ft['errors']} "
      f"({100 * (fe['errors'] - ft['errors']) / fe['errors']:.1f}% fewer errors)")

print("\nPer-class recall, %:")
print("{:<22}{:>8}{:>8}{:>8}".format("class", "FE", "FT", "delta"))
for i, c in enumerate(CLASSES):
    a, b = 100 * fe["per_class_recall"][i], 100 * ft["per_class_recall"][i]
    print("{:<22}{:>8.1f}{:>8.1f}{:>+8.1f}".format(c, a, b, b - a))

# Столбчатая диаграмма метрик
labels = ["Accuracy", "Precision", "Recall", "F1"]
keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7.5, 4))
for k, (tag, color) in enumerate([("feature_extraction", "#8fb3d9"), ("fine_tuning", "#1f5f99")]):
    vals = [100 * res[tag][1][m] for m in keys]
    bars = ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, label=res[tag][0], color=color)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x, labels)
lo = min(100 * res[t][1][m] for t in ("feature_extraction", "fine_tuning") for m in keys)
ax.set_ylim(np.floor(lo) - 2, 100)
ax.set_ylabel("%, тестовая выборка")
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "comparison_metrics.png"), dpi=150)
plt.close()

# Полнота по классам
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(CLASSES))
ax.bar(x - 0.18, [100 * v for v in fe["per_class_recall"]], 0.36, label="Feature Extraction", color="#8fb3d9")
ax.bar(x + 0.18, [100 * v for v in ft["per_class_recall"]], 0.36, label="Fine-tuning", color="#1f5f99")
ax.set_xticks(x, CLASSES, rotation=35, ha="right", fontsize=8.5)
ax.set_ylim(85, 100.5)
ax.set_ylabel("Recall, %")
ax.legend(fontsize=8.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "comparison_recall_per_class.png"), dpi=150)
plt.close()

# Кривые валидационной точности всех запусков
fig, ax = plt.subplots(figsize=(7.5, 4))
for tag, (name, r) in res.items():
    if tag == "feature_extraction":
        continue
    h = r["history"]["val_acc"]
    ax.plot(np.arange(0, len(h) + 1), [max(fe["history"]["val_acc"])] + h, marker="o", label=name)
# Прерванный запуск lr=1e-3 на 4 эпохи (метрики есть только в логе)
import re
ip = os.path.join(OUT, "03_ft_lr1e-3_interrupted.txt")
if os.path.exists(ip):
    h = [float(m) for m in re.findall(r"val_acc ([0-9.]+)", open(ip).read())]
    ax.plot(np.arange(0, len(h) + 1), [max(fe["history"]["val_acc"])] + h, marker="x", ls=":",
            color="crimson", label="Fine-tuning (blocks[5..6], lr=1e-3, 4 ep., прерван)")
ax.axhline(max(fe["history"]["val_acc"]), ls="--", color="gray", label="Feature Extraction (лучшая эпоха)")
ax.set_xlabel("эпоха дообучения (0 — старт с модели Feature Extraction)")
ax.set_ylabel("val accuracy")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "comparison_val_curves.png"), dpi=150)
plt.close()
