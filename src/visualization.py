# training_visualizer.py

import matplotlib.pyplot as plt


class TrainingVisualizer:
    """
    Simple helper for plotting training history:
      - loss / val_loss
      - accuracy / val_accuracy
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
