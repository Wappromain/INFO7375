"""
train.py
Builds, initializes, trains, and tests the network on the digit images.

Examples:
    python train.py
    python train.py --hidden 64 --lr 1.0 --epochs 3000
    python train.py --activation tanh --loss bce
    python train.py --train-dir my_images/train --test-dir my_images/test
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perceptron import (NeuralNetwork, Training, GradDescent,
                        load_labeled, load_unlabeled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--answer-key", default="data/test_answer_key.csv",
                    help="optional; used only to score the test predictions")
    ap.add_argument("--size", type=int, default=20, help="image side length")
    ap.add_argument("--hidden", type=int, nargs="+", default=[32])
    ap.add_argument("--activation", default="sigmoid", help="hidden layer activation")
    ap.add_argument("--output-activation", default="sigmoid")
    ap.add_argument("--init", default="xavier", choices=["xavier", "he", "random", "zeros"])
    ap.add_argument("--loss", default="mse", choices=["mse", "bce"])
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=10, help="mini batch size; 0 = full batch")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--target-loss", type=float, default=None)
    ap.add_argument("--augment", type=int, default=0,
                    help="add this many rotated/shifted copies of each training image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    size = (args.size, args.size)

    # 1. data
    X, Y, labels, classes = load_labeled(args.train_dir, size)
    X_test, test_files = load_unlabeled(args.test_dir, size)
    print(f"Training set: {X.shape[1]} images, {X.shape[0]} pixels each, classes {classes}")
    print(f"Test set: {X_test.shape[1]} unlabeled images")
    if args.augment:
        from perceptron import augment
        X, Y = augment(X, Y, copies=args.augment, image_size=size, seed=args.seed)
        print(f"Augmented training set: {X.shape[1]} images")

    # optional answer key so we can score the test predictions
    key = {}
    if args.answer_key and os.path.exists(args.answer_key):
        with open(args.answer_key) as f:
            key = {r["file"]: int(r["label"]) for r in csv.DictReader(f)}
    if key and not all(f in key for f in test_files):
        print("Answer key does not cover these test images; skipping test scoring.")
        key = {}
    Y_test = None
    if key:
        from perceptron import one_hot
        Y_test = one_hot(np.array([classes.index(str(key[f])) for f in test_files]), len(classes))

    # 2. model + parameter initialization
    net = NeuralNetwork(X.shape[0], args.hidden, len(classes),
                        hidden_activation=args.activation,
                        output_activation=args.output_activation,
                        init=args.init, seed=args.seed)
    print(net.summary())

    # 3. training
    trainer = Training(net, loss=args.loss,
                       optimizer=GradDescent(args.lr, args.momentum),
                       batch_size=args.batch or None, seed=args.seed)
    hist = trainer.fit(X, Y, epochs=args.epochs, target_loss=args.target_loss,
                       X_val=X_test, Y_val=Y_test, print_every=max(1, args.epochs // 10))
    net.save(os.path.join(args.out, "weights.npz"))

    # 4. testing
    probs = trainer.predict_proba(X_test)
    preds = np.argmax(probs, axis=0)
    rows = []
    for i, f in enumerate(test_files):
        rows.append({"file": f, "predicted": classes[preds[i]],
                     "confidence": f"{probs[preds[i], i]:.3f}",
                     "actual": key.get(f, "")})
    with open(os.path.join(args.out, "test_predictions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "predicted", "confidence", "actual"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nFinal training accuracy: {trainer.accuracy(X, Y):.1%}")
    if Y_test is not None:
        print(f"Test accuracy:           {trainer.accuracy(X_test, Y_test):.1%}")
        wrong = [r for r in rows if str(r["actual"]) != r["predicted"]]
        for r in wrong:
            print(f"  missed {r['file']}: said {r['predicted']} ({r['confidence']}), was {r['actual']}")

    # 5. plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"], color="#3b6fb6")
    ax[0].set_title(f"Training loss ({args.loss})")
    ax[0].set_xlabel("epoch"); ax[0].set_yscale("log"); ax[0].grid(alpha=0.3)
    ax[1].plot(hist["accuracy"], label="train", color="#3b6fb6")
    if hist["val_accuracy"]:
        ax[1].plot(hist["val_accuracy"], label="test", color="#d9822b")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "training_curves.png"), dpi=120)

    n = len(test_files)
    cols = 10
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols * 1.2, np.ceil(n / cols) * 1.45))
    for i, a in enumerate(axes.flat):
        a.axis("off")
        if i < n:
            a.imshow(X_test[:, i].reshape(size), cmap="gray", vmin=0, vmax=1)
            ok = (rows[i]["actual"] == "") or (str(rows[i]["actual"]) == rows[i]["predicted"])
            a.set_title(f"{rows[i]['predicted']} ({float(rows[i]['confidence']):.2f})",
                        fontsize=9, color="green" if ok else "red")
    fig.suptitle("Test images: predicted digit (confidence)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "test_predictions.png"), dpi=120)
    print(f"\nSaved weights, predictions and plots to {args.out}/")


if __name__ == "__main__":
    main()
