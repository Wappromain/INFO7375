"""
Lab 2: train the (10/ReLU)-(8/ReLU)^2-(4/ReLU)-(1/Sigmoid) network on the apple vs orange
training set I built in Lab 1.

How I train it: mini batch gradient descent with backpropagation, using binary cross entropy loss.
write every result (loss and accuracy curves, confusion matrices, predictions, trained weights,
and a text log) to lab2/results/.

Run it from the repository root:   python lab2/train.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # save plots straight to files, so I do not need a display window
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lab1"))
from network import DeepNetwork  # noqa: E402  (the network itself lives in Lab 1)

DATA = os.path.join(ROOT, "lab1", "data")
OUT = os.path.join(ROOT, "lab2", "results")
os.makedirs(OUT, exist_ok=True)

FEATURES = ["hue", "saturation", "brightness", "roundness", "texture"]
# learning_rate : how big a step I take on each weight update
# epochs        : the most full passes through the training set I allow
# batch_size    : how many samples I use for each weight update
# seed          : fixes the random starting weights and shuffling so results can be reproduced
# patience      : I stop early if validation loss has not improved for this many epochs
CONFIG = dict(learning_rate=0.05, epochs=400, batch_size=32, seed=1, patience=60)

# ---------- 1. load the Lab 1 data ----------
train = pd.read_csv(os.path.join(DATA, "train.csv"))
val = pd.read_csv(os.path.join(DATA, "validation.csv"))
real = pd.read_csv(os.path.join(DATA, "realworld_test.csv"))

X_train, d_train = train[FEATURES].to_numpy(float), train["label"].to_numpy(float)
X_val, d_val = val[FEATURES].to_numpy(float), val["label"].to_numpy(float)
X_real, d_real = real[FEATURES].to_numpy(float), real["label"].to_numpy(float)

# ---------- 2. standardise the inputs ----------
# My features live on very different scales (texture runs into the thousands, roundness sits near 1).
# rescale each one to mean 0 and standard deviation 1 so no feature dominates just because its
# numbers are bigger. compute the mean and std from the training set only, then apply the same
# numbers to validation and test data, so no information leaks from those sets into training.
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
scale = lambda X: (X - mean) / std
X_train, X_val, X_real = scale(X_train), scale(X_val), scale(X_real)

# ---------- 3. build the network ----------
net = DeepNetwork(n_inputs=len(FEATURES), seed=CONFIG["seed"])
log = ["Network", net.describe(len(FEATURES)), "", "Config: " + json.dumps(CONFIG), ""]
print("\n".join(log))


# ---------- 4. gradient check ----------
def gradient_check(net, X, d, eps=1e-5, n_checks=20, seed=0):
    """Before training, I confirm my backpropagation is correct. For a few random weights nudge the
    weight up and down by a tiny amount, measure how the loss changes, and compare that numerical
    slope with the gradient backpropagation gives. If they agree, the backprop maths is right."""
    rng = np.random.default_rng(seed)
    _, cache = net.forward(X)
    gW, _ = net.backward(cache, d)
    worst = 0.0
    for _ in range(n_checks):
        layer = rng.integers(len(net.W))
        i, j = (rng.integers(s) for s in net.W[layer].shape)
        original = net.W[layer][i, j]
        net.W[layer][i, j] = original + eps; plus = net.loss(net.predict_proba(X), d)
        net.W[layer][i, j] = original - eps; minus = net.loss(net.predict_proba(X), d)
        net.W[layer][i, j] = original
        numeric = (plus - minus) / (2 * eps)
        rel = abs(numeric - gW[layer][i, j]) / max(1e-8, abs(numeric) + abs(gW[layer][i, j]))
        worst = max(worst, rel)
    return worst


rel_err = gradient_check(net, X_train[:64], d_train[:64])
msg = f"Gradient check (backprop vs numerical), worst relative error: {rel_err:.2e} -> {'PASS' if rel_err < 1e-4 else 'CHECK'}"
print(msg); log += [msg, ""]


# ---------- 5. train ----------
def accuracy(net, X, d):
    return float((net.predict(X) == d).mean())


rng = np.random.default_rng(CONFIG["seed"])
history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
best = dict(val_loss=np.inf, epoch=0, W=None, b=None)

for epoch in range(1, CONFIG["epochs"] + 1):
    # shuffle every epoch so the network never sees the samples in the same order twice.
    order = rng.permutation(len(X_train))
    for start in range(0, len(order), CONFIG["batch_size"]):
        idx = order[start:start + CONFIG["batch_size"]]
        _, cache = net.forward(X_train[idx])               # 1) forward pass: compute y(p)
        gW, gb = net.backward(cache, d_train[idx])         # 2) backpropagate the error y(p) - d(p)
        net.update(gW, gb, CONFIG["learning_rate"])        # 3) adjust the weights and biases

    # After each epoch I record loss and accuracy on both sets so can plot the learning curves.
    history["train_loss"].append(net.loss(net.predict_proba(X_train), d_train))
    history["val_loss"].append(net.loss(net.predict_proba(X_val), d_val))
    history["train_acc"].append(accuracy(net, X_train, d_train))
    history["val_acc"].append(accuracy(net, X_val, d_val))

    # keep a copy of the weights from the epoch with the lowest validation loss. That is the
    # version that generalises best, before the network starts memorising the training set.
    if history["val_loss"][-1] < best["val_loss"]:
        best.update(val_loss=history["val_loss"][-1], epoch=epoch,
                    W=[w.copy() for w in net.W], b=[b.copy() for b in net.b])
    if epoch == 1 or epoch % 25 == 0:
        line = (f"epoch {epoch:>3}  train loss {history['train_loss'][-1]:.4f}  acc {history['train_acc'][-1]:.3f}"
                f"  |  val loss {history['val_loss'][-1]:.4f}  acc {history['val_acc'][-1]:.3f}")
        print(line); log.append(line)
    # Early stopping: if validation has not improved for `patience` epochs, more training won't help.
    if epoch - best["epoch"] >= CONFIG["patience"]:
        line = f"Early stop at epoch {epoch}: no validation improvement for {CONFIG['patience']} epochs"
        print(line); log.append(line)
        break

net.W, net.b = best["W"], best["b"]     # I restore the best weights before evaluating
log.append(f"Best epoch (lowest validation loss): {best['epoch']}")


# ---------- 6. evaluate ----------
def confusion(d, y):
    # Rows are the actual class, columns the predicted class: [[TN, FP], [FN, TP]], with orange as positive.
    return np.array([[np.sum((d == 0) & (y == 0)), np.sum((d == 0) & (y == 1))],
                     [np.sum((d == 1) & (y == 0)), np.sum((d == 1) & (y == 1))]])


results = {}
for name, X, d in [("train", X_train, d_train), ("validation", X_val, d_val), ("real world photos", X_real, d_real)]:
    y_pred = net.predict(X)
    results[name] = dict(loss=round(net.loss(net.predict_proba(X), d), 4),
                         accuracy=round(float((y_pred == d).mean()), 4),
                         confusion=confusion(d, y_pred).tolist())
log += ["", "Final results (best weights)"]
for name, r in results.items():
    log.append(f"{name:<18} loss {r['loss']:.4f}  accuracy {r['accuracy']:.3f}  confusion [[TN, FP], [FN, TP]] = {r['confusion']}")

# list each real world photo with the network's output so the individual decisions are visible.
real_out = real[["source", "variety", "label"]].copy()
real_out["y(p) = P(orange)"] = np.round(net.predict_proba(X_real), 4)
real_out["predicted"] = np.where(net.predict(X_real) == 1, "orange", "apple")
real_out["correct"] = (net.predict(X_real) == d_real)
real_out.to_csv(os.path.join(OUT, "realworld_predictions.csv"), index=False)
log += ["", "Predictions on the real world photos", real_out.drop(columns="source").to_string(index=False)]

# break validation accuracy down by variety to see which kinds of apple the network confuses with oranges.
val_out = val[["source", "variety", "label"]].copy()
val_out["y(p)"] = np.round(net.predict_proba(X_val), 4)
val_out["predicted"] = net.predict(X_val)
per_variety = val_out.assign(correct=val_out.predicted == val_out.label).groupby("variety")["correct"].mean().round(3)
log += ["", "Validation accuracy by variety", per_variety.to_string()]

print("\n".join(log[log.index("Final results (best weights)"):]))

# ---------- 7. save snapshots ----------
# save the feature mean and std with the weights, because any new input has to be scaled the same way.
net.save(os.path.join(OUT, "trained_weights.npz"), feature_mean=mean, feature_std=std)
pd.DataFrame(history).set_index(pd.RangeIndex(1, len(history["train_loss"]) + 1, name="epoch")) \
    .to_csv(os.path.join(OUT, "training_history.csv"))
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(dict(config=CONFIG, best_epoch=best["epoch"], gradient_check_rel_error=rel_err, results=results), f, indent=2)
with open(os.path.join(OUT, "training_log.txt"), "w") as f:
    f.write("\n".join(log) + "\n")

# Learning curves: if training loss keeps falling while validation loss rises, the network is overfitting.
TRAIN_C, VAL_C = "#2a6fdb", "#e8590c"
epochs = np.arange(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, key, title in [(axes[0], "loss", "Binary cross entropy loss"), (axes[1], "acc", "Accuracy")]:
    ax.plot(epochs, history[f"train_{key}"], color=TRAIN_C, label="training")
    ax.plot(epochs, history[f"val_{key}"], color=VAL_C, label="validation")
    ax.axvline(best["epoch"], color="#888", ls="--", lw=1, label=f"best epoch ({best['epoch']})")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
fig.suptitle("Training the (10/ReLU)-(8/ReLU)²-(4/ReLU)-(1/Sigmoid) network")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "training_curves.png"), dpi=150); plt.close(fig)

# Confusion matrices show which way the mistakes go (apples called oranges, or the reverse).
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, name in zip(axes, ["train", "validation"]):
    cm = np.array(results[name]["confusion"])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["apple", "orange"]); ax.set_yticks([0, 1], ["apple", "orange"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(f"{name}: accuracy {results[name]['accuracy']:.1%}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "confusion_matrices.png"), dpi=150); plt.close(fig)
print(f"\nSaved results to {OUT}")
