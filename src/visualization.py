# training_visualizer.py

import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    """
    Helper for plotting:
      - training & validation loss
      - training & validation accuracy
      - confusion matrix
    """

    def __init__(self):
        pass

    def plot_history(self, history, show=True, save_path=None):
        """
        history: tf.keras.callbacks.History object OR a dict with the same keys.
        """
        if hasattr(history, "history"):
            hist = history.history
        else:
            hist = history

        epochs = range(1, len(list(hist.values())[0]) + 1)

        # loss
        if "loss" in hist:
            plt.figure()
            plt.plot(epochs, hist.get("loss", []), label="loss")
            if "val_loss" in hist:
                plt.plot(epochs, hist["val_loss"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and validation loss")
            plt.legend()
            if save_path is not None:
                plt.savefig(save_path + "_loss.png", bbox_inches="tight")
            if show:
                plt.show()

        # accuracy
        if "accuracy" in hist:
            plt.figure()
            plt.plot(epochs, hist["accuracy"], label="accuracy")
            if "val_accuracy" in hist:
                plt.plot(epochs, hist["val_accuracy"], label="val_accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training and validation accuracy")
            plt.legend()
            if save_path is not None:
                plt.savefig(save_path + "_accuracy.png", bbox_inches="tight")
            if show:
                plt.show()

    def plot_confusion_matrix(
        self,
        cm,
        label_names=None,
        normalize=False,
        cmap="Blues",
        show=True,
        save_path=None
    ):
        """
        cm: confusion matrix (2D np.array)
        label_names: list of class names (optional)
        normalize: whether to show normalized values (per row)
        """
        cm = np.array(cm)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(f"cm must be a square 2D array, got shape {cm.shape}")

        if normalize:
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_sum[cm_sum == 0] = 1.0
            cm_display = cm / cm_sum
        else:
            cm_display = cm

        num_classes = cm.shape[0]
        if label_names is None or len(label_names) != num_classes:
            label_names = [str(i) for i in range(num_classes)]

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm_display, interpolation="nearest", cmap=cmap)
        plt.title("Confusion Matrix")
        plt.colorbar(im)

        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, label_names, rotation=45, ha="right")
        plt.yticks(tick_marks, label_names)

        fmt = ".2f" if normalize else "d"
        thresh = cm_display.max() / 2.0 if cm_display.size > 0 else 0.5

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j, i,
                    format(cm_display[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black"
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path + "_confusion_matrix.png", bbox_inches="tight")
        if show:
            plt.show()
